import logging
import pathlib
import threading
import sys

from gi.repository import Gtk, GObject, GLib, Gio, Gst, GstApp

from lada.gui.shortcuts import ShortcutsManager
from lada.gui.timeline import Timeline
from lada.lib import audio_utils, video_utils
from lada import LOG_LEVEL
from lada.gui.gstreamer_pipeline_appsrc import LadaAppSrc
from lada.gui.frame_restorer_provider import FrameRestorerProvider, FrameRestorerOptions, FRAME_RESTORER_PROVIDER

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
    box_playback_controls = Gtk.Template.Child()
    box_video_preview = Gtk.Template.Child()


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._frame_restorer_options: FrameRestorerOptions | None = None
        self._video_preview_init_done = False
        self._buffer_queue_min_thresh_time = 0
        self._buffer_queue_min_thresh_time_auto_min = 2.
        self._buffer_queue_min_thresh_time_auto_max = 10.
        self._buffer_queue_min_thresh_time_auto = self._buffer_queue_min_thresh_time_auto_min
        self._shortcuts_manager: ShortcutsManager | None = None

        self.video_sink = None
        self.audio_uridecodebin: Gst.UriDecodeBin | None = None
        self.audio_volume = None
        self.pipeline: Gst.Pipeline | None = None
        self.video_buffer_queue: Gst.Queue | None = None
        self.audio_buffer_queue: Gst.Queue | None = None
        self.pipeline_audio_elements = []
        self.eos = False

        self.lada_appsrc: LadaAppSrc | None = None
        self.frame_restorer_provider: FrameRestorerProvider = FRAME_RESTORER_PROVIDER
        self.file_duration_ns = 0
        self.frame_duration_ns = None
        self.video_metadata: video_utils.VideoMetadata | None = None
        self.has_audio: bool = True
        self.should_be_paused = False
        self.seek_in_progress = False
        self.waiting_for_data = False

        self.widget_timeline.connect('seek_requested', lambda widget, seek_position: self.seek_video(seek_position))
        self.widget_timeline.connect('cursor_position_changed', lambda widget, cursor_position: self.show_cursor_position(cursor_position))

    @GObject.Property()
    def frame_restorer_options(self):
        return self._frame_restorer_options

    @frame_restorer_options.setter
    def frame_restorer_options(self, value: FrameRestorerOptions):
        if self._frame_restorer_options == value:
            return
        if self._video_preview_init_done and self._buffer_queue_min_thresh_time == 0 and self._frame_restorer_options.max_clip_length != value.max_clip_length:
            self.buffer_queue_min_thresh_time_auto = float(value.max_clip_length / value.video_metadata.video_fps_exact)
        self._frame_restorer_options = value
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
    def buffer_queue_min_thresh_time(self):
        return self._buffer_queue_min_thresh_time

    @buffer_queue_min_thresh_time.setter
    def buffer_queue_min_thresh_time(self, value):
        if self._buffer_queue_min_thresh_time == value:
            return
        self._buffer_queue_min_thresh_time = value
        if self._video_preview_init_done:
            self.update_gst_buffers()

    @GObject.Property(type=ShortcutsManager)
    def shortcuts_manager(self):
        return self._shortcuts_manager

    @shortcuts_manager.setter
    def shortcuts_manager(self, value):
        self._shortcuts_manager = value
        self._setup_shortcuts()

    def on_fullscreen_activity(self, fullscreen_activity: bool):
        if fullscreen_activity:
            self.box_playback_controls.set_visible(True)
            self.button_play_pause.grab_focus()
        else:
            self.box_playback_controls.set_visible(False)

    def on_fullscreened(self, fullscreened: bool):
        if fullscreened:
            self.box_playback_controls.set_visible(False)
            self.box_video_preview.set_css_classes(["fullscreen-preview"])
        else:
            self.box_playback_controls.set_visible(True)
            self.button_play_pause.grab_focus()
            self.box_video_preview.remove_css_class("fullscreen-preview")

    @GObject.Signal(name="video-preview-reinit")
    def video_preview_init_start_signal(self):
        pass

    @GObject.Signal(name="video-preview-init-done")
    def video_preview_init_done_signal(self):
        pass

    @GObject.Signal(name="video-preview-close-done")
    def video_preview_close_done_signal(self):
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
        buffer_queue_min_thresh_time, buffer_queue_max_thresh_time = self.get_gst_buffer_bounds()

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
            if self.audio_volume:
                self.audio_volume.set_property("mute", True)
            self.pipeline.set_state(Gst.State.NULL)
            self.lada_appsrc.stop()
            self._video_preview_init_done = False
            self.emit('video-preview-close-done')
        threading.Thread(target=stop_pipeline).start()

    def open_video_file(self, file: Gio.File, mute_audio: bool):
        file_path = file.get_path()

        self.video_metadata = video_utils.get_video_meta_data(file_path)
        self._frame_restorer_options = self._frame_restorer_options.with_video_metadata(self.video_metadata)
        audio_pipeline_already_added = self.has_audio
        self.has_audio = audio_utils.get_audio_codec(self.video_metadata.video_file) is not None
        self.button_mute_unmute.set_sensitive(self.has_audio)
        self.set_speaker_icon(mute=not self.has_audio)

        self.frame_duration_ns = (1 / self.video_metadata.video_fps) * Gst.SECOND
        self.file_duration_ns = int((self.video_metadata.frames_count * self.frame_duration_ns))
        self._buffer_queue_min_thresh_time_auto_min = float(self._frame_restorer_options.max_clip_length / self.video_metadata.video_fps_exact)
        self.buffer_queue_min_thresh_time_auto = self._buffer_queue_min_thresh_time_auto_min

        self.widget_timeline.set_property("duration", self.file_duration_ns)

        self.frame_restorer_provider.init(self._frame_restorer_options)

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

    def grab_focus(self):
        self.button_play_pause.grab_focus()

    def adjust_pipeline_with_new_source_file(self, audio_pipeline_already_added, mute_audio):
        self.lada_appsrc.reinit(self.video_metadata)
        if self.has_audio:
            if audio_pipeline_already_added:
                self.audio_volume.set_property("mute", mute_audio)
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

    def get_gst_buffer_bounds(self):
        buffer_queue_min_thresh_time = self._buffer_queue_min_thresh_time if self._buffer_queue_min_thresh_time > 0 else self._buffer_queue_min_thresh_time_auto
        buffer_queue_max_thresh_time = buffer_queue_min_thresh_time * 2
        return buffer_queue_min_thresh_time, buffer_queue_max_thresh_time

    def pipeline_add_video(self):
        self.lada_appsrc = LadaAppSrc(self.video_metadata, self.frame_restorer_provider, self.autoplay_if_enough_data_buffered)
        appsrc = self.lada_appsrc.appsrc
        self.pipeline.add(appsrc)

        buffer_queue_min_thresh_time, buffer_queue_max_thresh_time = self.get_gst_buffer_bounds()

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
        buffer_queue_min_thresh_time, buffer_queue_max_thresh_time = self.get_gst_buffer_bounds()

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

        GLib.timeout_add(20, self.update_current_position) # todo: remove when timeline is not visible (export view)

    def reset_appsource_worker(self):
        self.emit('video-preview-reinit')

        def reinit():
            self._video_preview_init_done = False
            self.pipeline.set_state(Gst.State.PAUSED)
            self.frame_restorer_provider.init(self._frame_restorer_options)
            self.lada_appsrc.reinit(self.video_metadata)

            # seeking flush to flush pipeline / clean out buffers
            res, position = self.pipeline.query_position(Gst.Format.TIME)
            if res and position >= 0:
                self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, position)

            self.pipeline.set_state(Gst.State.PLAYING)

        exporter_thread = threading.Thread(target=reinit)
        exporter_thread.start()

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

    def _setup_shortcuts(self):
        self._shortcuts_manager.register_group("preview", "Preview")
        self._shortcuts_manager.add("preview", "toggle-mute-unmute", "m", lambda *args: self.button_mute_unmute_callback(self.button_mute_unmute), "Mute/Unmute")
        self._shortcuts_manager.add("preview", "toggle-play-pause", "<Alt>space", lambda *args: self.button_play_pause_callback(self.button_play_pause), "Play/Pause")

    def close(self, block=False):
        def shutdown():
            if self.audio_volume:
                self.audio_volume.set_property("mute", True)
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            if self.lada_appsrc:
                self.lada_appsrc.stop()
        if block:
            shutdown()
        else:
            shutdown_thread = threading.Thread(target=shutdown)
            shutdown_thread.start()