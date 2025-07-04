import logging
import os
import pathlib
import threading

from gi.repository import Gtk, GObject, GLib, Gio, Gst, GstApp, Adw

from lada.gui.config import Config
from lada.gui.config_sidebar import ConfigSidebar
from lada.gui.fullscreen_mouse_activity_controller import FullscreenMouseActivityController
from lada.gui.gstreamer_pipeline_manager import PipelineManager, PipelineState
from lada.gui.shortcuts import ShortcutsManager
from lada.gui.timeline import Timeline
from lada.lib import audio_utils, video_utils
from lada import LOG_LEVEL
from lada.gui.frame_restorer_provider import FrameRestorerProvider, FrameRestorerOptions, FRAME_RESTORER_PROVIDER

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(filename=here / 'video_preview_view.ui')
class VideoPreviewView(Gtk.Widget):
    __gtype_name__ = 'VideoPreviewView'

    button_play_pause = Gtk.Template.Child()
    button_mute_unmute = Gtk.Template.Child()
    picture_video_preview: Gtk.Picture = Gtk.Template.Child()
    widget_timeline: Timeline = Gtk.Template.Child()
    button_image_play_pause = Gtk.Template.Child()
    button_image_mute_unmute = Gtk.Template.Child()
    label_current_time = Gtk.Template.Child()
    label_cursor_time = Gtk.Template.Child()
    box_playback_controls = Gtk.Template.Child()
    box_video_preview = Gtk.Template.Child()
    button_export_video = Gtk.Template.Child()
    toggle_button_preview_video = Gtk.Template.Child()
    spinner_overlay = Gtk.Template.Child()
    banner_no_gpu = Gtk.Template.Child()
    config_sidebar: ConfigSidebar = Gtk.Template.Child()
    header_bar: Adw.HeaderBar = Gtk.Template.Child()
    button_toggle_fullscreen: Gtk.Button = Gtk.Template.Child()
    stack_video_preview: Gtk.Stack = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._frame_restorer_options: FrameRestorerOptions | None = None
        self._video_preview_init_done = False
        self._buffer_queue_min_thresh_time = 0
        self._buffer_queue_min_thresh_time_auto_min = 2.
        self._buffer_queue_min_thresh_time_auto_max = 10.
        self._buffer_queue_min_thresh_time_auto = self._buffer_queue_min_thresh_time_auto_min
        self._shortcuts_manager: ShortcutsManager | None = None
        self._window_title: str | None = None

        self.eos = False

        self.frame_restorer_provider: FrameRestorerProvider = FRAME_RESTORER_PROVIDER
        self.file_duration_ns = 0
        self.frame_duration_ns = None
        self.opened_file: Gio.File | None = None
        self.video_metadata: video_utils.VideoMetadata | None = None
        self.has_audio: bool = True
        self.should_be_paused = False
        self.seek_in_progress = False
        self.waiting_for_data = False

        self._config: Config | None = None

        self.widget_timeline.connect('seek_requested', lambda widget, seek_position: self.seek_video(seek_position))
        self.widget_timeline.connect('cursor_position_changed', lambda widget, cursor_position: self.show_cursor_position(cursor_position))

        self.fullscreen_mouse_activity_controller = None

        self.pipeline_manager: PipelineManager | None = None

        self.stack_video_preview.set_visible_child_name("spinner")

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
        if self._config.get_property('device') == 'cpu':
            self.banner_no_gpu.set_revealed(True)
        self.setup_config_signal_handlers()

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

    @GObject.Property(type=str)
    def window_title(self):
        return self._window_title

    @window_title.setter
    def window_title(self, value):
        self._window_title = value

    @GObject.Signal(name="toggle-fullscreen-requested")
    def toggle_fullscreen_requested(self):
        pass

    @GObject.Signal(name="video-export-requested")
    def video_export_requested_signal(self, source_file: Gio.File, save_file: Gio.File):
        pass

    @Gtk.Template.Callback()
    def toggle_button_preview_video_callback(self, button_clicked):
        if self._video_preview_init_done:
            self.frame_restorer_options = self._frame_restorer_options.with_passthrough(not self._frame_restorer_options.passthrough)

    @Gtk.Template.Callback()
    def button_toggle_fullscreen_callback(self, button_clicked):
        self.emit("toggle-fullscreen-requested")

    @Gtk.Template.Callback()
    def button_export_video_callback(self, button_clicked):
        self.show_export_dialog()

    @Gtk.Template.Callback()
    def button_play_pause_callback(self, button_clicked):
        if not self._video_preview_init_done or self.seek_in_progress:
            return

        if self.pipeline_manager.state == PipelineState.PLAYING:
            self.should_be_paused = True
            self.pipeline_manager.pause()
        elif self.pipeline_manager.state == PipelineState.PAUSED:
            self.should_be_paused = False
            if self.eos:
                self.seek_video(0)
            self.pipeline_manager.play()
        else:
            logger.warning(f"unhandled pipeline state in button_play_pause_callback: {self.pipeline_manager.state}")

    @Gtk.Template.Callback()
    def button_mute_unmute_callback(self, button_clicked):
        if not (self.has_audio and self._video_preview_init_done):
            return
        self.pipeline_manager.muted = not self.pipeline_manager.muted

    @property
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

    def setup_config_signal_handlers(self):
        def on_preview_mode(*args):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_mosaic_detection(self._config.preview_mode == 'mosaic-detection')
        self._config.connect("notify::preview-mode", on_preview_mode)

        def on_device(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_device(self._config.device)
        self._config.connect("notify::device", on_device)

        def on_mosaic_restoration_model(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_mosaic_restoration_model_name(self._config.mosaic_restoration_model)
        self._config.connect("notify::mosaic-restoration-model", on_mosaic_restoration_model)

        def on_mosaic_detection_model(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_mosaic_detection_model_name(self._config.mosaic_detection_model)
        self._config.connect("notify::mosaic-detection-model", on_mosaic_detection_model)

        self._config.connect("notify::preview-buffer-duration", lambda object, spec: self.set_property('buffer-queue-min-thresh-time', object.get_property(spec.name)))

        def on_max_clip_duration(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_max_clip_length(self._config.max_clip_duration)
        self._config.connect("notify::max-clip-duration", on_max_clip_duration)

    def set_mute_audio(self, audio_volume, mute: bool):
        audio_volume.set_property("mute", mute)
        self.set_speaker_icon(mute)

    def set_speaker_icon(self, mute: bool):
        icon_name = "speaker-0-symbolic" if mute else "speaker-4-symbolic"
        self.button_image_mute_unmute.set_property("icon-name", icon_name)

    def update_gst_buffers(self):
        buffer_queue_min_thresh_time, buffer_queue_max_thresh_time = self.get_gst_buffer_bounds()
        self.pipeline_manager.update_gst_buffers(buffer_queue_min_thresh_time, buffer_queue_max_thresh_time)

    def seek_video(self, seek_position_ns):
        if self.seek_in_progress:
            return
        # pipeline.seek_simple is blocking. As we're stopping/starting our appsrc on seek this could potentially take a few seconds and freeze the UI
        def seek():
            self.eos = False
            self.seek_in_progress = True
            self.spinner_overlay.set_visible(True)
            self.label_current_time.set_text(self.get_time_label_text(seek_position_ns))
            self.widget_timeline.set_property("playhead-position", seek_position_ns)
            self.pipeline_manager.seek(seek_position_ns)
            self.seek_in_progress = False
            if not self.waiting_for_data:
                self.spinner_overlay.set_visible(False)

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
        if not self.pipeline_manager:
            return
        self.pipeline_manager.close_video_file()
        self._video_preview_init_done = False

    def open_file(self, file: Gio.File):
        self._show_spinner()
        self.button_export_video.set_sensitive(True)
        file_changed = self.opened_file is not None

        def run():
            if file_changed:
                self.close_video_file()
            GLib.idle_add(lambda: self._open_file(file))

        threading.Thread(target=run).start()


    def _open_file(self, file: Gio.File):
        self.opened_file = file
        self.frame_restorer_options = FrameRestorerOptions(self.config.mosaic_restoration_model, self.config.mosaic_detection_model, video_utils.get_video_meta_data(self.opened_file.get_path()), self.config.device, self.config.max_clip_duration, self.config.preview_mode == 'mosaic-detection', False)
        file_path = file.get_path()

        self.video_metadata = video_utils.get_video_meta_data(file_path)
        self._frame_restorer_options = self._frame_restorer_options.with_video_metadata(self.video_metadata)
        self.has_audio = audio_utils.get_audio_codec(self.video_metadata.video_file) is not None
        self.button_mute_unmute.set_sensitive(self.has_audio)
        self.set_speaker_icon(mute=not self.has_audio)

        self.frame_duration_ns = (1 / self.video_metadata.video_fps) * Gst.SECOND
        self.file_duration_ns = int((self.video_metadata.frames_count * self.frame_duration_ns))
        self._buffer_queue_min_thresh_time_auto_min = float(self._frame_restorer_options.max_clip_length / self.video_metadata.video_fps_exact)
        self.buffer_queue_min_thresh_time_auto = self._buffer_queue_min_thresh_time_auto_min

        self.widget_timeline.set_property("duration", self.file_duration_ns)

        self.frame_restorer_provider.init(self._frame_restorer_options)

        if self.pipeline_manager:
            self.pipeline_manager.adjust_pipeline_with_new_source_file(self.video_metadata)
        else:
            buffer_queue_min_thresh_time, buffer_queue_max_thresh_time = self.get_gst_buffer_bounds()
            self.pipeline_manager = PipelineManager(self.video_metadata, self.frame_restorer_provider, buffer_queue_min_thresh_time, buffer_queue_max_thresh_time, self.config.mute_audio)
            self.pipeline_manager.init_pipeline()
            self.picture_video_preview.set_paintable(self.pipeline_manager.paintable)
            self.pipeline_manager.connect("eos", self.on_eos)
            self.pipeline_manager.connect("waiting-for-data", lambda obj, waiting_for_data: self.on_waiting_for_data(waiting_for_data))
            self.pipeline_manager.connect("notify::state", lambda object, spec: self.on_pipeline_state(object.get_property(spec.name)))
            GLib.timeout_add(20, self.update_current_position)  # todo: remove when timeline is not visible (export view)

        threading.Thread(target=self.pipeline_manager.play).start()

    def on_eos(self, *args):
        self.eos = True
        self.button_image_play_pause.set_property("icon-name", "media-playback-start-symbolic")

    def on_pipeline_state(self, state: PipelineState):
        if state == PipelineState.PLAYING:
            self.button_image_play_pause.set_property("icon-name", "media-playback-pause-symbolic")
        elif state == PipelineState.PAUSED:
            self.button_image_play_pause.set_property("icon-name", "media-playback-start-symbolic")
        if not self._video_preview_init_done and state == PipelineState.PLAYING:
            self._video_preview_init_done = True
            self._show_video_preview()

    def pause_if_currently_playing(self):
        if not self._video_preview_init_done:
            return
        if self.pipeline_manager.state == PipelineState.PLAYING:
            self.should_be_paused = True
            self.pipeline_manager.pause()

    def grab_focus(self):
        self.button_play_pause.grab_focus()

    def on_waiting_for_data(self, waiting_for_data):
        self.waiting_for_data = waiting_for_data
        self.spinner_overlay.set_visible(waiting_for_data)
        if waiting_for_data:
            self.pipeline_manager.pause()
            if self._buffer_queue_min_thresh_time == 0 and self._video_preview_init_done:
                self.buffer_queue_min_thresh_time_auto *= 1.5
                self.update_gst_buffers()
        else:
            if not self.should_be_paused:
                self.pipeline_manager.play()

    def get_gst_buffer_bounds(self):
        buffer_queue_min_thresh_time = self._buffer_queue_min_thresh_time if self._buffer_queue_min_thresh_time > 0 else self._buffer_queue_min_thresh_time_auto
        buffer_queue_max_thresh_time = buffer_queue_min_thresh_time * 2
        return buffer_queue_min_thresh_time, buffer_queue_max_thresh_time

    def reset_appsource_worker(self):
        self._show_spinner()

        def reinit():
            self._video_preview_init_done = False
            self.pipeline_manager.pause()
            self.frame_restorer_provider.init(self._frame_restorer_options)
            self.pipeline_manager.reinit_appsrc()
            self.pipeline_manager.play()

        reinit_thread = threading.Thread(target=reinit)
        reinit_thread.start()

    def update_current_position(self):
        position = self.pipeline_manager.get_position_ns()
        if position is not None:
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

    def on_fullscreen_activity(self, fullscreen_activity: bool):
        if fullscreen_activity:
            self.header_bar.set_visible(True)
            self.set_cursor_from_name("default")
            self.box_playback_controls.set_visible(True)
            self.button_play_pause.grab_focus()
        else:
            self.header_bar.set_visible(False)
            self.set_cursor_from_name("none")
            self.box_playback_controls.set_visible(False)

    def on_fullscreened(self, fullscreened: bool):
        if fullscreened:
            self.fullscreen_mouse_activity_controller = FullscreenMouseActivityController(self)
            self.banner_no_gpu.set_revealed(False)
            self.button_toggle_fullscreen.set_property("icon-name", "view-restore-symbolic")
            self.box_video_preview.set_css_classes(["fullscreen-preview"])
        else:
            self.header_bar.set_visible(True)
            self.set_cursor_from_name("default")
            self.button_toggle_fullscreen.set_property("icon-name", "view-fullscreen-symbolic")
            self.box_playback_controls.set_visible(True)
            self.button_play_pause.grab_focus()
            self.box_video_preview.remove_css_class("fullscreen-preview")
            if self._config.get_property('device') == 'cpu':
                self.banner_no_gpu.set_revealed(True)
        self.fullscreen_mouse_activity_controller.on_fullscreened(fullscreened)
        self.fullscreen_mouse_activity_controller.connect("notify::fullscreen-activity", lambda object, spec: self.on_fullscreen_activity(object.get_property(spec.name)))

    def _show_spinner(self, *args):
        self.config_sidebar.set_property("disabled", True)
        self.toggle_button_preview_video.set_property("sensitive", False)
        self.stack_video_preview.set_visible_child_name("spinner")

    def _show_video_preview(self, *args):
        self.config_sidebar.set_property("disabled", False)
        self.toggle_button_preview_video.set_property("sensitive", True)
        self.stack_video_preview.set_visible_child_name("video-player")
        self.grab_focus()

    def _setup_shortcuts(self):
        def on_shortcut_preview_toggle(*args):
            if self.stack_video_preview.get_visible_child_name() == "video-player":
                self.toggle_button_preview_video_callback(self.toggle_button_preview_video)

        self._shortcuts_manager.register_group("files", "Files")
        self._shortcuts_manager.add("files", "export-file", "e", lambda *args: self.show_export_dialog(), "Export recovered video")

        self._shortcuts_manager.register_group("preview", "Preview")
        self._shortcuts_manager.add("preview", "toggle-mute-unmute", "m", lambda *args: self.button_mute_unmute_callback(self.button_mute_unmute), "Mute/Unmute")
        self._shortcuts_manager.add("preview", "toggle-play-pause", "<Alt>space", lambda *args: self.button_play_pause_callback(self.button_play_pause), "Play/Pause")
        self._shortcuts_manager.add("preview", "toggle-preview", "p", on_shortcut_preview_toggle, "Enable/Disable preview mode")
        self._shortcuts_manager.add("preview", "toggle-fullscreen", "<Ctrl>f", lambda *args: self.emit("toggle-fullscreen-requested"), "Enable/Disable fullscreen")

    def show_export_dialog(self):
        if not self.opened_file:
            return
        self.pause_if_currently_playing()
        file_dialog = Gtk.FileDialog()
        video_file_filter = Gtk.FileFilter()
        video_file_filter.add_mime_type("video/*")
        file_dialog.set_default_filter(video_file_filter)
        file_dialog.set_title("Save restored video file")
        file_dialog.set_initial_folder(self.opened_file.get_parent())
        file_dialog.set_initial_name(f"{os.path.splitext(self.opened_file.get_basename())[0]}.restored.mp4")
        file_dialog.save(callback=lambda dialog, result: self.emit("video-export-requested", self.opened_file, dialog.save_finish(result)))

    def close(self, block=False):
        if not self.pipeline_manager:
            return
        if block:
            self.pipeline_manager.close_video_file()
        else:
            shutdown_thread = threading.Thread(target=self.pipeline_manager.close_video_file)
            shutdown_thread.start()