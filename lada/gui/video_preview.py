import logging
import os
import pathlib
import tempfile
import threading
import queue
import numpy as np
from gi.repository import Gtk, GObject, GdkPixbuf, GLib, Gio, Gst, GstApp, Adw

from lada.gui.config import MODEL_NAMES_TO_FILES
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
        self._mosaic_cleaning = False
        self._mosaic_detection = False
        self._mosaic_restoration_model_name = 'basicvsrpp-generic-1.1'
        self._device = "cpu"
        self._video_preview_init_done = False
        self._max_clip_length = 180
        self._buffer_queue_min_thresh_time = 0
        self._buffer_queue_min_thresh_time_auto_min = 2.
        self._buffer_queue_min_thresh_time_auto_max = 10.
        self._buffer_queue_min_thresh_time_auto = self._buffer_queue_min_thresh_time_auto_min
        self._application: Adw.Application | None = None

        self.appsrc: GstApp | None = None
        self.audio_uridecodebin: Gst.UriDecodeBin | None = None
        self.audio_volume = None
        self.pipeline: Gst.Pipeline | None = None
        self.video_buffer_queue: Gst.Queue | None = None
        self.audio_buffer_queue: Gst.Queue | None = None
        self.pipeline_audio_elements = []

        self.frame_restorer_thread: threading.Thread | None = None
        self.frame_restorer_thread_queue: queue.Queue = queue.Queue()
        self.worker_should_be_running: bool = False

        self.frame_restorer: FrameRestorer | None = None
        self.frame_restorer_lock: threading.Lock = threading.Lock()
        self.file_duration_ns = 0
        self.frame_duration_ns = None
        self.current_timestamp_ns = 0
        self.video_metadata: video_utils.VideoMetadata | None = None
        self.has_audio: bool = True
        self.models_cache: dict | None = None
        self.should_be_paused = False

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
            self.reset_frame_restorer()

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
            self.reset_frame_restorer()

    @property
    def buffer_queue_min_thresh_time_auto(self):
        return self._buffer_queue_min_thresh_time_auto

    @buffer_queue_min_thresh_time_auto.setter
    def buffer_queue_min_thresh_time_auto(self, value):
        value = min(self._buffer_queue_min_thresh_time_auto_max, max(self._buffer_queue_min_thresh_time_auto_min, value))
        if self._buffer_queue_min_thresh_time_auto == value:
            return
        print("adjusted buffer_queue_min_thresh_time_auto to", value)
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
            self.reset_frame_restorer()

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
            self.reset_frame_restorer()

    @GObject.Property()
    def mosaic_cleaning(self):
        return self._mosaic_cleaning

    @mosaic_cleaning.setter
    def mosaic_cleaning(self, value):
        if self._mosaic_cleaning == value:
            return
        self._mosaic_cleaning = value
        if self._video_preview_init_done:
            self.reset_frame_restorer()

    @GObject.Property()
    def mosaic_detection(self):
        return self._mosaic_detection

    @mosaic_detection.setter
    def mosaic_detection(self, value):
        if self._mosaic_detection == value:
            return
        self._mosaic_detection = value
        if self._video_preview_init_done:
            self.reset_frame_restorer()

    @GObject.Property(type=Adw.Application)
    def application(self):
        return self._application

    @application.setter
    def application(self, value):
        self._application = value
        self._setup_shortcuts()

    @GObject.Signal(name="video-preview-init-done")
    def video_preview_init_done_signal(self):
        pass

    @GObject.Signal(name="video-export-finished")
    def video_export_finished_signal(self):
        pass

    @GObject.Signal(name="video-export-progress")
    def video_export_progress_signal(self, status: float):
        pass

    @Gtk.Template.Callback()
    def button_play_pause_callback(self, button_clicked):
        if not self._video_preview_init_done:
            return
        pipe_state = self.pipeline.get_state(20 * Gst.MSECOND)
        if pipe_state.state == Gst.State.PLAYING:
            self.should_be_paused = True
            self.pipeline.set_state(Gst.State.PAUSED)
        elif pipe_state.state == Gst.State.PAUSED:
            self.should_be_paused = False
            self.pipeline.set_state(Gst.State.PLAYING)
        else:
            print("unhandled pipeline state in button_play_pause_callback", pipe_state.nick_value)

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
        self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, seek_position_ns)

    def show_cursor_position(self, cursor_position_ns):
        if cursor_position_ns > 0:
            self.label_cursor_time.set_visible(True)
            label_text = self.get_time_label_text(cursor_position_ns)
            self.label_cursor_time.set_text(label_text)
        else:
            self.label_cursor_time.set_visible(False)

    def open_video_file(self, file: Gio.File, mute_audio: bool):
        file_path = file.get_path()

        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.stop_frame_restorer_worker()
            self._video_preview_init_done = False

        self.video_metadata = video_utils.get_video_meta_data(file_path)
        audio_pipeline_already_added = self.has_audio
        self.has_audio = audio_utils.get_audio_codec(self.video_metadata.video_file) is not None
        self.button_mute_unmute.set_sensitive(self.has_audio)
        self.set_speaker_icon(mute=not self.has_audio)

        self.setup_frame_restorer(start_ns=0)
        self.start_frame_restorer_worker()

        self.frame_duration_ns = (1 / self.video_metadata.video_fps) * Gst.SECOND
        self.file_duration_ns = int((self.video_metadata.frames_count * self.frame_duration_ns))
        self._buffer_queue_min_thresh_time_auto_min = float(self._max_clip_length / self.video_metadata.video_fps_exact)
        self.buffer_queue_min_thresh_time_auto = self._buffer_queue_min_thresh_time_auto_min

        self.widget_timeline.set_property("duration", self.file_duration_ns)

        if self.pipeline:
            self.adjust_pipeline_with_new_source_file(audio_pipeline_already_added, mute_audio)
        else:
            self.init_pipeline(mute_audio)

        self.pipeline.set_state(Gst.State.PLAYING)

    def export_video(self, file_path, video_codec, crf):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        self._video_preview_init_done = False
        self._mosaic_detection = False
        self.setup_frame_restorer(start_ns=0)

        def run_export():
            video_tmp_file_output_path = os.path.join(tempfile.gettempdir(),f"{os.path.basename(os.path.splitext(file_path)[0])}.tmp{os.path.splitext(file_path)[1]}")
            video_writer = video_utils.VideoWriter(video_tmp_file_output_path, self.video_metadata.video_width,
                                       self.video_metadata.video_height, self.video_metadata.video_fps_exact,
                                       time_base=self.video_metadata.time_base, codec=video_codec, crf=crf)

            progress_update_step_size = 100
            for frame_num, (restored_frame, restored_frame_pts) in enumerate(self.frame_restorer):
                video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)
                if frame_num % progress_update_step_size == 0:
                    self.emit('video-export-progress', frame_num / self.video_metadata.frames_count)

            video_writer.release()

            audio_utils.combine_audio_video_files(self.video_metadata.video_file, video_tmp_file_output_path, file_path)
            self.emit('video-export-progress', 1.0)
            self.emit('video-export-finished')

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
                self.audio_uridecodebin.set_property('uri', 'file://' + self.video_metadata.video_file)
            else:
                self.pipeline_add_audio(mute_audio)
        else:
            self.pipeline_remove_audio()

    def autoplay_if_enough_data_buffered(self):
        if not self.should_be_paused:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.spinner_video_preview.set_visible(False)

    def autopause_if_not_enough_data_buffered(self):
        self.pipeline.set_state(Gst.State.PAUSED)
        self.spinner_video_preview.set_visible(True)
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

        def on_running(queue):
            current_level_time = queue.get_property('current-level-time')
            if not self._video_preview_init_done and current_level_time > 0:
                self._video_preview_init_done = True
                self.emit('video-preview-init-done')

        buffer_queue.connect("underrun", lambda queue: self.autopause_if_not_enough_data_buffered())
        buffer_queue.connect("overrun", lambda queue: self.autoplay_if_enough_data_buffered())
        buffer_queue.connect('running', on_running)
        self.pipeline.add(buffer_queue)

        gtksink = Gst.ElementFactory.make('gtk4paintablesink', None)
        paintable = gtksink.get_property('paintable')
        if paintable.props.gl_context:
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
        audio_uridecodebin.set_property('uri', 'file://' + self.video_metadata.video_file)

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
                    print("eos")
                    self.button_image_play_pause.set_property("icon-name", "media-playback-start-symbolic")
                case Gst.MessageType.ERROR:
                    (err, _) = msg.parse_error()
                    print(f'Error from {msg.src.get_path_string()}: {err}')
                case Gst.MessageType.STATE_CHANGED:
                    old_state, new_state, pending_state = msg.parse_state_changed()
                    if old_state == Gst.State.PAUSED and new_state == Gst.State.PLAYING:
                        self.button_image_play_pause.set_property("icon-name", "media-playback-pause-symbolic")
                    elif old_state == Gst.State.PLAYING and new_state == Gst.State.PAUSED:
                        self.button_image_play_pause.set_property("icon-name", "media-playback-start-symbolic")
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
        if offset_ns == self.current_timestamp_ns:
            return True
        print(f"called on_seek_data of appsrc with offset (sec): {offset_ns / Gst.SECOND}, current position (sec): {self.current_timestamp_ns / Gst.SECOND}")
        self.pipeline.set_state(Gst.State.PAUSED)
        self.frame_restorer_lock.acquire()
        self.stop_frame_restorer_worker()
        self.setup_frame_restorer(start_ns=offset_ns)
        self.start_frame_restorer_worker()
        self.frame_restorer_lock.release()
        return True

    def reset_frame_restorer(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.frame_restorer_lock.acquire()
        self.stop_frame_restorer_worker()
        self.setup_frame_restorer(start_ns=self.current_timestamp_ns)
        self.start_frame_restorer_worker()
        self.frame_restorer_lock.release()

    def start_frame_restorer_worker(self):
        self.worker_should_be_running = True
        self.frame_restorer_thread = threading.Thread(target=self.frame_restorer_worker)
        self.frame_restorer_thread.start()

    def stop_frame_restorer_worker(self):
        self.worker_should_be_running = False

        # unblock consumer
        threading_utils.put_closing_queue_marker(self.frame_restorer_thread_queue, "frame_restorer_thread_queue")

        self.frame_restorer_thread.join()
        self.frame_restorer_thread = None

        # garbage collection
        threading_utils.empty_out_queue(self.frame_restorer_thread_queue, "frame_restorer_thread_queue")
        logger.debug("VideoPreview frame restorer worker: stopped")

    def frame_restorer_worker(self):
        logger.debug("VideoPreview frame restorer worker: started")
        while self.worker_should_be_running:
            self.frame_restorer_thread_queue.get()
            self.read_next_frame()
            self.frame_restorer_thread_queue.task_done()

    def read_next_frame(self):
        try:
            result = next(self.frame_restorer)
            if result is None:
                assert self.frame_restorer.stop_requested, "Illegal state: Received None (stop marker) from frame restorer but it wasn't requested to stop"
                return
            frame, frame_pts = result
        except StopIteration:
            self.appsrc.emit("end-of-stream")
            return

        width = frame.shape[1]
        # TODO: see reasoning for this zero padding in TODO where we specify appsrc Caps
        if width % 4 != 0:
            frame = np.pad(frame, ((0, 0), (0, width % 4), (0, 0)), mode='constant', constant_values=0)

        data = frame.tostring()

        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.frame_duration_ns
        buf.pts = int(self.current_timestamp_ns)
        buf.offset = video_utils.offset_ns_to_frame_num(int(self.current_timestamp_ns), self.video_metadata.video_fps_exact)
        self.appsrc.emit('push-buffer', buf)
        self.current_timestamp_ns += self.frame_duration_ns

    def on_need_data(self, src, length):
        self.frame_restorer_thread_queue.put("work worker, work!")

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

    def setup_frame_restorer(self, start_ns=0):
        if self.models_cache is None or self.models_cache["mosaic_restoration_model_name"] != self._mosaic_restoration_model_name:
            print(f"model {self._mosaic_restoration_model_name} not found in cache. Loading...")
            mosaic_restoration_model_path = MODEL_NAMES_TO_FILES[self._mosaic_restoration_model_name]
            mosaic_detection_model, mosaic_restoration_model, mosaic_edge_detection_model, mosaic_restoration_model_preferred_pad_mode = load_models(
                self._device, self._mosaic_restoration_model_name, mosaic_restoration_model_path, None,
                os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v2.pt'),None
            )

            self.models_cache = dict(mosaic_restoration_model_name=self._mosaic_restoration_model_name,
                                     mosaic_detection_model=mosaic_detection_model,
                                     mosaic_restoration_model=mosaic_restoration_model,
                                     mosaic_edge_detection_model=mosaic_edge_detection_model,
                                     mosaic_restoration_model_preferred_pad_mode=mosaic_restoration_model_preferred_pad_mode)

        if self.frame_restorer:
            self.frame_restorer.stop()
        if self._passthrough:
            self.frame_restorer = PassthroughFrameRestorer(self.video_metadata.video_file)
        else:
            self.frame_restorer = FrameRestorer(self._device, self.video_metadata.video_file, True, self._max_clip_length, self._mosaic_restoration_model_name,
                                                self.models_cache["mosaic_detection_model"], self.models_cache["mosaic_restoration_model"], self.models_cache["mosaic_edge_detection_model"], self.models_cache["mosaic_restoration_model_preferred_pad_mode"],
                                                mosaic_detection=self._mosaic_detection, mosaic_cleaning=self._mosaic_cleaning)
        self.frame_restorer.start(start_ns=int(start_ns))

        self.current_timestamp_ns = start_ns

    def _setup_shortcuts(self):
        self._application.shortcuts.register_group("preview", "Preview")
        self._application.shortcuts.add("preview", "toggle-mute-unmute", "m", lambda *args: self.button_mute_unmute_callback(self.button_mute_unmute), "Mute/Unmute")
        self._application.shortcuts.add("preview", "toggle-play-pause", "<Alt>space", lambda *args: self.button_play_pause_callback(self.button_play_pause), "Play/Pause")


class PassthroughFrameRestorer:
    def __init__(self, video_file):
        self.video_file = video_file
        self.video_reader: video_utils.VideoReader | None = None
        self.video_frames_generator = None
        self.stop_requested = False

    def start(self, start_ns=0):
        self.video_reader = video_utils.VideoReader(self.video_file)
        self.video_reader.__enter__()
        if start_ns == 0:
            self.video_reader.seek(start_ns)
        self.video_frames_generator = self.video_reader.frames()

    def stop(self):
        self.video_reader.__exit__(None, None, None)
        self.stop_requested = True

    def __iter__(self):
        return self

    def __next__(self):
        """
        returns None if being called while FrameRestorer is being stopped
        """
        return next(self.video_frames_generator)