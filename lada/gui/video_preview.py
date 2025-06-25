import logging
import os
import pathlib
import tempfile
import threading
import queue
import time
import sys

import numpy as np
from gi.repository import Gtk, GObject, GLib, Gio, Gst, GstApp, Adw

from lada import RESTORATION_MODEL_NAMES_TO_FILES, DETECTION_MODEL_NAMES_TO_FILES
from lada.gui.timeline import Timeline
from lada.lib import audio_utils, video_utils, threading_utils
from lada.lib.frame_restorer import load_models, FrameRestorer
from lada import MODEL_WEIGHTS_DIR, LOG_LEVEL

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(filename=here / 'video_preview.ui')
class VideoPreview(Gtk.Widget):
    __gtype_name__ = 'VideoPreview'

    button_play_pause = Gtk.Template.Child()
    button_mute_unmute = Gtk.Template.Child()
    picture_video_preview = Gtk.Template.Child()
    widget_timeline: Timeline = Gtk.Template.Child()
    button_image_play_pause = Gtk.Template.Child()
    button_image_mute_unmute = Gtk.Template.Child()
    label_current_time = Gtk.Template.Child()
    label_cursor_time = Gtk.Template.Child()
    spinner_video_preview = Gtk.Template.Child()


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._passthrough = False
        self._mosaic_detection = False
        self._mosaic_restoration_model_name = 'basicvsrpp-1.2'
        self._mosaic_detection_model_name = 'v3.1-accurate'
        self._device = "cpu"
        self._video_preview_init_done = False
        self._max_clip_length = 180
        self._buffer_queue_min_thresh_time = 0
        self._buffer_queue_min_thresh_time_auto_min = 2.
        self._buffer_queue_min_thresh_time_auto_max = 10.
        self._buffer_queue_min_thresh_time_auto = self._buffer_queue_min_thresh_time_auto_min
        self._application: Adw.Application | None = None

        self.appsrc: GstApp | None = None
        self.video_sink = None
        self.audio_uridecodebin: Gst.UriDecodeBin | None = None
        self.audio_volume = None
        self.pipeline: Gst.Pipeline | None = None
        self.video_buffer_queue: Gst.Queue | None = None
        self.audio_buffer_queue: Gst.Queue | None = None
        self.pipeline_audio_elements = []
        self.eos = False

        self.appsource_thread: threading.Thread | None = None
        self.appsource_queue: queue.Queue = queue.Queue()
        self.appsource_thread_should_be_running: bool = False
        self.appsource_thread_stop_requested = False

        self.frame_restorer: FrameRestorer | None = None
        self.frame_restorer_lock: threading.Lock = threading.Lock()
        self.file_duration_ns = 0
        self.frame_duration_ns = None
        self.current_timestamp_ns = 0
        self.video_metadata: video_utils.VideoMetadata | None = None
        self.has_audio: bool = True
        self.models_cache: dict | None = None
        self.should_be_paused = False
        self.seek_in_progress = False
        self.waiting_for_data = False

        self.widget_timeline.connect('seek_requested', lambda widget, seek_position: self.seek_video(seek_position))
        self.widget_timeline.connect('cursor_position_changed', lambda widget, cursor_position: self.show_cursor_position(cursor_position))


    @GObject.Property()
    def passthrough(self):
        return self._passthrough

    @passthrough.setter
    def passthrough(self, value):
        if self._passthrough == value:
            return
        self._passthrough = value
        if self._video_preview_init_done:
            self.reset_appsource_worker()

    @GObject.Property()
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if self._device == value:
            return
        self._device = value
        self.models_cache = None
        if self._video_preview_init_done:
            self.reset_appsource_worker()

    @property
    def buffer_queue_min_thresh_time_auto(self):
        return self._buffer_queue_min_thresh_time_auto

    @buffer_queue_min_thresh_time_auto.setter
    def buffer_queue_min_thresh_time_auto(self, value):
        value = min(self._buffer_queue_min_thresh_time_auto_max, max(self._buffer_queue_min_thresh_time_auto_min, value))
        if self._buffer_queue_min_thresh_time_auto == value:
            return
        logger.info(f"adjusted buffer_queue_min_thresh_time_auto to {value}")
        self._buffer_queue_min_thresh_time_auto = value
        if self._video_preview_init_done:
            self.update_gst_buffers()

    @GObject.Property()
    def max_clip_length(self):
        return self._max_clip_length

    @max_clip_length.setter
    def max_clip_length(self, value):
        if self._max_clip_length == value:
            return
        self._max_clip_length = value
        if self._video_preview_init_done and self._buffer_queue_min_thresh_time == 0:
            self.buffer_queue_min_thresh_time_auto = float(self._max_clip_length / self.video_metadata.video_fps_exact)
            self.reset_appsource_worker()

    @GObject.Property()
    def buffer_queue_min_thresh_time(self):
        return self._buffer_queue_min_thresh_time

    @buffer_queue_min_thresh_time.setter
    def buffer_queue_min_thresh_time(self, value):
        if self._buffer_queue_min_thresh_time == value:
            return
        self._buffer_queue_min_thresh_time = value
        if self._video_preview_init_done:
            self.update_gst_buffers()

    @GObject.Property()
    def mosaic_restoration_model(self):
        return self._mosaic_restoration_model_name

    @mosaic_restoration_model.setter
    def mosaic_restoration_model(self, value):
        if self._mosaic_restoration_model_name == value:
            return
        self._mosaic_restoration_model_name = value
        if self._video_preview_init_done:
            self.reset_appsource_worker()

    @GObject.Property()
    def mosaic_detection_model(self):
        return self._mosaic_detection_model_name

    @mosaic_detection_model.setter
    def mosaic_detection_model(self, value):
        if self._mosaic_detection_model_name == value:
            return
        self._mosaic_detection_model_name = value
        if self._video_preview_init_done:
            self.reset_appsource_worker()

    @GObject.Property()
    def mosaic_detection(self):
        return self._mosaic_detection

    @mosaic_detection.setter
    def mosaic_detection(self, value):
        if self._mosaic_detection == value:
            return
        self._mosaic_detection = value
        if self._video_preview_init_done:
            self.reset_appsource_worker()

    @GObject.Property(type=Adw.Application)
    def application(self):
        return self._application

    @application.setter
    def application(self, value):
        self._application = value
        self._setup_shortcuts()

    @GObject.Signal(name="video-preview-reinit")
    def video_preview_init_start_signal(self):
        pass

    @GObject.Signal(name="video-preview-init-done")
    def video_preview_init_done_signal(self):
        pass

    @GObject.Signal(name="video-preview-close-done")
    def video_preview_close_done_signal(self):
        pass

    @GObject.Signal(name="video-export-finished")
    def video_export_finished_signal(self):
        pass

    @GObject.Signal(name="video-export-progress")
    def video_export_progress_signal(self, status: float):
        pass

    @Gtk.Template.Callback()
    def button_play_pause_callback(self, button_clicked):
        if not self._video_preview_init_done or self.seek_in_progress:
            return
        if self.eos:
            self.seek_video(0)
        pipe_state = self.pipeline.get_state(20 * Gst.MSECOND)
        if pipe_state.state == Gst.State.PLAYING:
            self.should_be_paused = True
            self.pipeline.set_state(Gst.State.PAUSED)
        elif pipe_state.state == Gst.State.PAUSED:
            self.should_be_paused = False
            self.pipeline.set_state(Gst.State.PLAYING)
        else:
            logger.warning(f"unhandled pipeline state in button_play_pause_callback: {pipe_state.nick_value}")

    @Gtk.Template.Callback()
    def button_mute_unmute_callback(self, button_clicked):
        if not (self.has_audio and self._video_preview_init_done):
            return
        muted = self.audio_volume.get_property("mute")
        self.set_mute_audio(self.audio_volume, not muted)

    def set_mute_audio(self, audio_volume, mute: bool):
        audio_volume.set_property("mute", mute)
        self.set_speaker_icon(mute)

    def set_speaker_icon(self, mute: bool):
        icon_name = "speaker-0-symbolic" if mute else "speaker-4-symbolic"
        self.button_image_mute_unmute.set_property("icon-name", icon_name)

    def update_gst_buffers(self):
        buffer_queue_min_thresh_time = self._buffer_queue_min_thresh_time if self._buffer_queue_min_thresh_time > 0 else self._buffer_queue_min_thresh_time_auto
        buffer_queue_max_thresh_time = buffer_queue_min_thresh_time * 2

        self.video_buffer_queue.set_property('max-size-time', buffer_queue_max_thresh_time * Gst.SECOND)
        self.video_buffer_queue.set_property('min-threshold-time', buffer_queue_min_thresh_time * Gst.SECOND)
        if self.has_audio:
            self.audio_buffer_queue.set_property('max-size-time', buffer_queue_max_thresh_time * Gst.SECOND)
            self.audio_buffer_queue.set_property('min-threshold-time', buffer_queue_min_thresh_time * Gst.SECOND)

    def seek_video(self, seek_position_ns):
        if self.seek_in_progress:
            return
        # pipeline.seek_simple is blocking. As we're stopping/starting our appsrc on seek this could potentially take a few seconds and freeze the UI
        def seek():
            self.eos = False
            self.seek_in_progress = True
            self.spinner_video_preview.set_visible(True)
            self.label_current_time.set_text(self.get_time_label_text(seek_position_ns))
            self.widget_timeline.set_property("playhead-position", seek_position_ns)
            # Pausing before seek seems to fix an issue where calling seek_simple() never returns.
            # I did not notice it on smaller/shorter files but on long files (>3h) I could reproduce this issue pretty consistently.
            # Shouldn't be necessary and I don't understand how it helps but apparently it does.
            self.pipeline.set_state(Gst.State.PAUSED)
            self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, seek_position_ns)
            logger.debug("returned from pipeline.seek_simple()")
            self.pipeline.set_state(Gst.State.PLAYING)
            self.seek_in_progress = False
            if not self.waiting_for_data:
                self.spinner_video_preview.set_visible(False)

        seek_thread = threading.Thread(target=seek, daemon=True)
        seek_thread.start()

    def show_cursor_position(self, cursor_position_ns):
        if cursor_position_ns > 0:
            self.label_cursor_time.set_visible(True)
            label_text = self.get_time_label_text(cursor_position_ns)
            self.label_cursor_time.set_text(label_text)
        else:
            self.label_cursor_time.set_visible(False)

    def close_video_file(self):
        if not self.pipeline:
            return
        def stop_pipeline():
            self.pipeline.set_state(Gst.State.NULL)
            self.stop_appsource_worker()
            self._video_preview_init_done = False
            self.emit('video-preview-close-done')
        threading.Thread(target=stop_pipeline).start()

    def open_video_file(self, file: Gio.File, mute_audio: bool):
        file_path = file.get_path()

        self.video_metadata = video_utils.get_video_meta_data(file_path)
        audio_pipeline_already_added = self.has_audio
        self.has_audio = audio_utils.get_audio_codec(self.video_metadata.video_file) is not None
        self.button_mute_unmute.set_sensitive(self.has_audio)
        self.set_speaker_icon(mute=not self.has_audio)

        self.frame_duration_ns = (1 / self.video_metadata.video_fps) * Gst.SECOND
        self.file_duration_ns = int((self.video_metadata.frames_count * self.frame_duration_ns))
        self._buffer_queue_min_thresh_time_auto_min = float(self._max_clip_length / self.video_metadata.video_fps_exact)
        self.buffer_queue_min_thresh_time_auto = self._buffer_queue_min_thresh_time_auto_min

        self.widget_timeline.set_property("duration", self.file_duration_ns)

        if self.pipeline:
            self.adjust_pipeline_with_new_source_file(audio_pipeline_already_added, mute_audio)
        else:
            self.init_pipeline(mute_audio)

        def start_pipeline():
            self.pipeline.set_state(Gst.State.PLAYING)
        threading.Thread(target=start_pipeline).start()

    def pause_if_currently_playing(self):
        if not self._video_preview_init_done:
            return
        pipe_state = self.pipeline.get_state(20 * Gst.MSECOND)
        if pipe_state.state == Gst.State.PLAYING:
            self.should_be_paused = True
            self.pipeline.set_state(Gst.State.PAUSED)

    def export_video(self, file_path, video_codec, crf):
        def run_export():
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            self._video_preview_init_done = False
            self._mosaic_detection = False
            self._passthrough = False
            self.stop_appsource_worker()
            self.setup_frame_restorer()

            progress_update_step_size = 100
            success = True
            video_tmp_file_output_path = os.path.join(tempfile.gettempdir(),f"{os.path.basename(os.path.splitext(file_path)[0])}.tmp{os.path.splitext(file_path)[1]}")
            try:
                self.frame_restorer.start(start_ns=0)

                with video_utils.VideoWriter(video_tmp_file_output_path, self.video_metadata.video_width,
                                             self.video_metadata.video_height, self.video_metadata.video_fps_exact,
                                             video_codec, time_base=self.video_metadata.time_base,
                                             crf=crf) as video_writer:
                    for frame_num, elem in enumerate(self.frame_restorer):
                        if elem is None:
                            success = False
                            logger.error("Error on export: frame restorer stopped prematurely")
                            break
                        (restored_frame, restored_frame_pts) = elem
                        video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)
                        if frame_num % progress_update_step_size == 0:
                            self.emit('video-export-progress', frame_num / self.video_metadata.frames_count)

            except Exception as e:
                success = False
                logger.error("Error on export", exc_info=e)
            finally:
                self.frame_restorer.stop()

            if success:
                audio_utils.combine_audio_video_files(self.video_metadata, video_tmp_file_output_path, file_path)
                self.emit('video-export-progress', 1.0)
                self.emit('video-export-finished')
            else:
                if os.path.exists(video_tmp_file_output_path):
                    os.remove(video_tmp_file_output_path)

        exporter_thread = threading.Thread(target=run_export)
        exporter_thread.start()

    def grab_focus(self):
        self.button_play_pause.grab_focus()

    def adjust_pipeline_with_new_source_file(self, audio_pipeline_already_added, mute_audio):
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={self.video_metadata.video_width},height={self.video_metadata.video_height},framerate={self.video_metadata.video_fps_exact.numerator}/{self.video_metadata.video_fps_exact.denominator}")
        self.appsrc.set_property('caps', caps)
        self.appsrc.set_property('duration', self.file_duration_ns)
        if self.has_audio:
            if audio_pipeline_already_added:
                self.audio_uridecodebin.set_property('uri', pathlib.Path(self.video_metadata.video_file).resolve().as_uri())
            else:
                self.pipeline_add_audio(mute_audio)
        else:
            self.pipeline_remove_audio()

    def autoplay_if_enough_data_buffered(self):
        self.waiting_for_data = False
        self.spinner_video_preview.set_visible(False)
        if not self.should_be_paused:
            self.pipeline.set_state(Gst.State.PLAYING)

    def autopause_if_not_enough_data_buffered(self):
        self.waiting_for_data = True
        self.spinner_video_preview.set_visible(True)
        self.pipeline.set_state(Gst.State.PAUSED)
        if self._buffer_queue_min_thresh_time == 0 and self._video_preview_init_done:
            self.buffer_queue_min_thresh_time_auto *= 1.5
            self.update_gst_buffers()

    def get_initial_buffer_queue_thresholds(self):
        buffer_queue_min_thresh_time = self._buffer_queue_min_thresh_time if self._buffer_queue_min_thresh_time > 0 else self._buffer_queue_min_thresh_time_auto
        buffer_queue_max_thresh_time = buffer_queue_min_thresh_time * 2
        return buffer_queue_min_thresh_time, buffer_queue_max_thresh_time

    def pipeline_add_video(self):
        appsrc = Gst.ElementFactory.make('appsrc', "numpy-source")
        # TODO: As we're using BGR format GStreamer expects to receive a 'buffer size = rstride (image) * height' where 'rstride = RU4 (width * 3)'
        # RU4 here means that it will round up to nearest number divisible by 4. (https://gstreamer.freedesktop.org/documentation/additional/design/mediatype-video-raw.html)
        # Most common video widths like 1920, 1280, 640 are divisible by 4 so no problem we can allocate a buffer according to numpy shape H*W*C
        # But if we receive a video with a width which isn't divisible by 4 (I saw this on a file with dimensions 854 x 480) then the pipeline would break as H*W*C is less then expected buffer size given above calculation.
        # For now let's just add some zero padding. Maybe there are ways to explicitly set the stride size or tell GStreamer about our zero padding but couldn't find anything...
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={self.video_metadata.video_width + self.video_metadata.video_width % 4},height={self.video_metadata.video_height},framerate={self.video_metadata.video_fps_exact.numerator}/{self.video_metadata.video_fps_exact.denominator}")
        appsrc.set_property('caps', caps)
        appsrc.set_property('is-live', False)
        appsrc.set_property('emit-signals', True)
        appsrc.set_property('stream-type', GstApp.AppStreamType.SEEKABLE)
        appsrc.set_property('format', Gst.Format.TIME)
        appsrc.set_property('duration', self.file_duration_ns)
        appsrc.connect('need-data', self.on_need_data)
        appsrc.connect('seek-data', self.on_seek_data)
        def on_eos(appsrc):
            self.autoplay_if_enough_data_buffered()
            return True
        appsrc.connect("end-of-stream", on_eos)
        self.pipeline.add(appsrc)

        buffer_queue_min_thresh_time, buffer_queue_max_thresh_time = self.get_initial_buffer_queue_thresholds()

        buffer_queue = Gst.ElementFactory.make('queue', None)
        buffer_queue.set_property('max-size-bytes', 0)
        buffer_queue.set_property('max-size-buffers', 0)
        buffer_queue.set_property('max-size-time', buffer_queue_max_thresh_time * Gst.SECOND)  # ns
        buffer_queue.set_property('min-threshold-time', buffer_queue_min_thresh_time * Gst.SECOND)

        buffer_queue.connect("underrun", lambda queue: self.autopause_if_not_enough_data_buffered())
        buffer_queue.connect("overrun", lambda queue: self.autoplay_if_enough_data_buffered())
        self.pipeline.add(buffer_queue)

        gtksink = Gst.ElementFactory.make('gtk4paintablesink', None)
        paintable = gtksink.get_property('paintable')
        # TODO: workaround for #62. On Windows using Nvidia GPU and OpenGL for the paintable when it's available causes messed up colors.
        #  I could not reproduce this on a VM without a Nvidia card.
        if paintable.props.gl_context and sys.platform != 'win32':
            video_sink = Gst.ElementFactory.make('glsinkbin', None)
            video_sink.set_property('sink', gtksink)
        else:
            video_sink = Gst.Bin.new()
            convert = Gst.ElementFactory.make('videoconvert', None)
            video_sink.add(convert)
            video_sink.add(gtksink)
            convert.link(gtksink)
            video_sink.add_pad(Gst.GhostPad.new('sink', convert.get_static_pad('sink')))
        self.pipeline.add(video_sink)

        appsrc.link(buffer_queue)
        buffer_queue.link(video_sink)

        self.appsrc = appsrc
        self.video_buffer_queue = buffer_queue
        self.picture_video_preview.set_paintable(paintable)
        self.video_sink = video_sink

    def pipeline_remove_audio(self):
        for audio_element in self.pipeline_audio_elements:
            audio_element.set_state(Gst.State.NULL)
            self.pipeline.remove(audio_element)
        self.audio_uridecodebin = None
        self.audio_volume = None
        self.audio_buffer_queue = None

    def pipeline_add_audio(self, mute_audio: bool):
        buffer_queue_min_thresh_time, buffer_queue_max_thresh_time = self.get_initial_buffer_queue_thresholds()

        audio_queue = Gst.ElementFactory.make('queue', None)
        audio_queue.set_property('max-size-bytes', 0)
        audio_queue.set_property('max-size-buffers', 0)
        audio_queue.set_property('max-size-time', buffer_queue_max_thresh_time * Gst.SECOND)  # ns
        audio_queue.set_property('min-threshold-time', buffer_queue_min_thresh_time * Gst.SECOND)
        self.pipeline.add(audio_queue)
        self.pipeline_audio_elements.append(audio_queue)

        audio_uridecodebin = Gst.ElementFactory.make('uridecodebin', None)
        audio_uridecodebin.set_property('uri', pathlib.Path(self.video_metadata.video_file).resolve().as_uri())

        def on_pad_added(decodebin, decoder_src_pad, audio_queue):
            caps = decoder_src_pad.get_current_caps()
            if not caps:
                caps = decoder_src_pad.query_caps()
            gststruct = caps.get_structure(0)
            gstname = gststruct.get_name()
            if gstname.startswith("audio"):
                sink_pad = audio_queue.get_static_pad("sink")
                decoder_src_pad.link(sink_pad)

        audio_uridecodebin.connect("pad-added", on_pad_added, audio_queue)
        self.pipeline.add(audio_uridecodebin)
        self.pipeline_audio_elements.append(audio_uridecodebin)

        audio_audioconvert = Gst.ElementFactory.make('audioconvert', None)
        self.pipeline.add(audio_audioconvert)
        self.pipeline_audio_elements.append(audio_audioconvert)

        audio_volume = Gst.ElementFactory.make('volume', None)
        self.set_mute_audio(audio_volume, mute_audio)
        self.pipeline.add(audio_volume)
        self.pipeline_audio_elements.append(audio_volume)

        audio_sink = Gst.ElementFactory.make('autoaudiosink', None)
        self.pipeline.add(audio_sink)
        self.pipeline_audio_elements.append(audio_sink)

        # note that we cannot link decodebin directly to audio_queue as pads are dynamically added and not available at this point
        # see on_pad_added()
        audio_queue.link(audio_audioconvert)
        audio_audioconvert.link(audio_volume)
        audio_volume.link(audio_sink)

        self.audio_uridecodebin = audio_uridecodebin
        self.audio_volume = audio_volume
        self.audio_buffer_queue = audio_queue

    def init_pipeline(self, mute_audio: bool):
        pipeline = Gst.Pipeline.new()

        def on_bus_msg(_, msg):
            match msg.type:
                case Gst.MessageType.EOS:
                    self.button_image_play_pause.set_property("icon-name", "media-playback-start-symbolic")
                    self.eos = True
                case Gst.MessageType.ERROR:
                    (err, _) = msg.parse_error()
                    logger.error(f"Error from {msg.src.get_path_string()}: {err}")
                case Gst.MessageType.STATE_CHANGED:
                    if msg.src == self.pipeline:
                        old_state, new_state, pending_state = msg.parse_state_changed()
                        if old_state == Gst.State.PAUSED and new_state == Gst.State.PLAYING:
                            self.button_image_play_pause.set_property("icon-name", "media-playback-pause-symbolic")
                        elif old_state == Gst.State.PLAYING and new_state == Gst.State.PAUSED:
                            self.button_image_play_pause.set_property("icon-name", "media-playback-start-symbolic")
                        if not self._video_preview_init_done and new_state == Gst.State.PLAYING:
                            self._video_preview_init_done = True
                            self.emit('video-preview-init-done')

                case Gst.MessageType.STREAM_STATUS:
                    pass
                case _:
                    # print("other message", msg.type)
                    pass
            return True

        bus = pipeline.get_bus()
        bus.add_watch(GLib.PRIORITY_DEFAULT, on_bus_msg)

        self.pipeline = pipeline
        self.pipeline_add_video()
        if self.has_audio:
            self.pipeline_add_audio(mute_audio)

        GLib.timeout_add(20, self.update_current_position)

    def on_seek_data(self, appsrc, offset_ns):
        if self.frame_restorer and self.video_metadata.video_file == self.frame_restorer.video_meta_data.video_file and offset_ns == self.current_timestamp_ns:
            # nothing to do, we're already at the desired position in the file
            return True
        logger.debug(f"called on_seek_data of appsrc with offset (sec): {offset_ns / Gst.SECOND}, current position (sec): {self.current_timestamp_ns / Gst.SECOND}")
        self.frame_restorer_lock.acquire()
        self.pipeline.set_state(Gst.State.PAUSED)
        self.stop_appsource_worker()
        self.start_appsource_worker(start_ns=offset_ns)
        self.frame_restorer_lock.release()
        return True

    def reset_appsource_worker(self):
        self.emit('video-preview-reinit')

        def reset():
            self._video_preview_init_done = False
            self.pipeline.set_state(Gst.State.PAUSED)
            self.stop_appsource_worker()
            self.setup_frame_restorer()
            self.start_appsource_worker(start_ns=self.current_timestamp_ns)
            self.seek_video(self.current_timestamp_ns)
            self.pipeline.set_state(Gst.State.PLAYING)

        exporter_thread = threading.Thread(target=reset)
        exporter_thread.start()


    def start_appsource_worker(self, start_ns):
        self.appsource_thread_stop_requested = False
        self.appsource_thread_should_be_running = True
        self.current_timestamp_ns = start_ns

        self.setup_frame_restorer()
        self.frame_restorer.start(start_ns=int(start_ns))

        self.appsource_thread = threading.Thread(target=self._appsource_worker)
        self.appsource_thread.start()

    def stop_appsource_worker(self):
        start = time.time()
        if not self.frame_restorer:
            logger.debug(f"appsource worker: stopped, took {time.time() - start}")
            return
        self.appsource_thread_stop_requested = True
        self.appsource_thread_should_be_running = False

        self.frame_restorer.stop()

        # unblock consumer
        threading_utils.put_closing_queue_marker(self.appsource_queue, "appsource_queue")
        threading_utils.put_closing_queue_marker(self.frame_restorer.get_frame_restoration_queue(), "frame_restorer_thread_queue")

        if self.appsource_thread:
            self.appsource_thread.join()
            self.appsource_thread = None

        # garbage collection
        threading_utils.empty_out_queue(self.appsource_queue, "appsource_queue")
        threading_utils.put_closing_queue_marker(self.frame_restorer.get_frame_restoration_queue(), "frame_restorer_thread_queue")
        logger.debug(f"appsource worker: stopped, took {time.time() - start}")

    def _appsource_worker(self):
        logger.debug("appsource worker: started")
        eof = False
        while self.appsource_thread_should_be_running:
            self.appsource_queue.get()
            if self.appsource_thread_stop_requested:
                logger.debug("appsource worker: consumer unblocked")
            eof = self._appsource_read_next_frame()
            self.appsource_queue.task_done()
        if eof:
            logger.debug("appsource worker: stopped itself, EOF")

    def _appsource_read_next_frame(self) -> bool:
        result = self.frame_restorer.get_frame_restoration_queue().get()
        if self.appsource_thread_stop_requested:
            logger.debug("appsource worker: frame_restoration_queue consumer unblocked")
        if result is None:
            self.appsource_thread_should_be_running = False
            if not self.appsource_thread_stop_requested:
                self.appsrc.emit("end-of-stream")
                return True
            return False
        else:
            frame, frame_pts = result

        frame_timestamp_ns = int((frame_pts * self.video_metadata.time_base) * Gst.SECOND)

        width = frame.shape[1]
        # TODO: see reasoning for this zero padding in TODO where we specify appsrc Caps
        if width % 4 != 0:
            frame = np.pad(frame, ((0, 0), (0, width % 4), (0, 0)), mode='constant', constant_values=0)

        data = frame.tobytes()

        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.frame_duration_ns
        buf.pts = frame_timestamp_ns
        buf.offset = video_utils.offset_ns_to_frame_num(frame_timestamp_ns, self.video_metadata.video_fps_exact)
        self.appsrc.emit('push-buffer', buf)
        self.current_timestamp_ns = frame_timestamp_ns
        return False

    def on_need_data(self, src, length):
        self.appsource_queue.put("work worker, work!")

    def update_current_position(self):
        res, position = self.pipeline.query_position(Gst.Format.TIME)
        if res and position >= 0:
            label_text = self.get_time_label_text(position)
            self.label_current_time.set_text(label_text)
            self.widget_timeline.set_property("playhead-position", position)
        return True

    def get_time_label_text(self, time_ns):
        if not time_ns or time_ns == -1:
            return '00:00:00'
        else:
            seconds = int(time_ns / Gst.SECOND)
            minutes = int(seconds / 60)
            hours = int(minutes / 60)
            seconds = seconds % 60
            minutes = minutes % 60
            hours, minutes, seconds = int(hours), int(minutes), int(seconds)
            time = f"{minutes}:{seconds:02d}" if hours == 0 else f"{hours}:{minutes:02d}:{seconds:02d}"
            return time

    def setup_frame_restorer(self):
        if self.models_cache is None or self.models_cache["mosaic_restoration_model_name"] != self._mosaic_restoration_model_name or self.models_cache["mosaic_detection_model_name"] != self._mosaic_detection_model_name:
            logger.info(f"model {self._mosaic_restoration_model_name} not found in cache. Loading...")
            mosaic_restoration_model_path = RESTORATION_MODEL_NAMES_TO_FILES[self._mosaic_restoration_model_name]
            mosaic_detection_path = DETECTION_MODEL_NAMES_TO_FILES[self._mosaic_detection_model_name]
            mosaic_detection_model, mosaic_restoration_model, mosaic_restoration_model_preferred_pad_mode = load_models(
                self._device, self._mosaic_restoration_model_name, mosaic_restoration_model_path, None,
                mosaic_detection_path
            )

            self.models_cache = dict(mosaic_restoration_model_name=self._mosaic_restoration_model_name,
                                     mosaic_detection_model_name=self._mosaic_detection_model_name,
                                     mosaic_detection_model=mosaic_detection_model,
                                     mosaic_restoration_model=mosaic_restoration_model,
                                     mosaic_restoration_model_preferred_pad_mode=mosaic_restoration_model_preferred_pad_mode)

        if self._passthrough:
            self.frame_restorer = PassthroughFrameRestorer(self.video_metadata.video_file)
        else:
            self.frame_restorer = FrameRestorer(self._device, self.video_metadata.video_file, True, self._max_clip_length, self._mosaic_restoration_model_name,
                                                self.models_cache["mosaic_detection_model"], self.models_cache["mosaic_restoration_model"], self.models_cache["mosaic_restoration_model_preferred_pad_mode"],
                                                mosaic_detection=self._mosaic_detection)

    def _setup_shortcuts(self):
        self._application.shortcuts.register_group("preview", "Preview")
        self._application.shortcuts.add("preview", "toggle-mute-unmute", "m", lambda *args: self.button_mute_unmute_callback(self.button_mute_unmute), "Mute/Unmute")
        self._application.shortcuts.add("preview", "toggle-play-pause", "<Alt>space", lambda *args: self.button_play_pause_callback(self.button_play_pause), "Play/Pause")

    def close(self):
        def shutdown():
            if self.audio_volume:
                self.audio_volume.set_property("mute", True)
            self.stop_appsource_worker()
        shutdown_thread = threading.Thread(target=shutdown)
        shutdown_thread.start()

class PassthroughFrameRestorer:
    def __init__(self, video_file):
        self.video_file = video_file
        self.video_reader: video_utils.VideoReader | None = None
        self.frame_restoration_queue = None
        self.stopped = False

    def start(self, start_ns=0):
        self.video_reader = video_utils.VideoReader(self.video_file)
        self.video_reader = self.video_reader.__enter__()
        if start_ns >= 0:
            self.video_reader.seek(start_ns)
        self.frame_restoration_queue = PassthroughFrameRestorer.PassthroughQueue(self)

    def stop(self):
        self.stopped = True
        self.video_reader.__exit__(None, None, None)

    def get_frame_restoration_queue(self):
        return self.frame_restoration_queue

    class PassthroughQueue:
        def __init__(self, frame_restorer):
            self.video_frames_generator = frame_restorer.video_reader.frames()
            self.frame_restorer = frame_restorer

        def get(self, block=True, timeout=None):
            if self.frame_restorer.stopped:
                return None
            try:
                return next(self.video_frames_generator)
            except StopIteration:
                return None

        def put(self, item, block=True, timeout=None):
            pass
