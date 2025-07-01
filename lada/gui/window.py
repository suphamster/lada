import os.path
import pathlib

from gi.repository import Adw, Gtk, Gio, GObject
from lada.gui.config import Config
from lada.gui.config_sidebar import ConfigSidebar
from lada.gui.file_selection_view import FileSelectionView
from lada.gui.frame_restorer_provider import FrameRestorerOptions
from lada.gui.fullscreen_mouse_activity_controller import FullscreenMouseActivityController
from lada.gui.video_export_view import VideoExportView
from lada.gui.video_preview import VideoPreview
from lada.lib import video_utils

here = pathlib.Path(__file__).parent.resolve()

@Gtk.Template(filename=here / 'window.ui')
class MainWindow(Adw.ApplicationWindow):
    __gtype_name__ = 'MainWindow'

    file_selection_view: FileSelectionView = Gtk.Template.Child()
    video_export_view: VideoExportView = Gtk.Template.Child()
    button_export_video = Gtk.Template.Child()
    toggle_button_preview_video = Gtk.Template.Child()
    widget_video_preview: VideoPreview = Gtk.Template.Child()
    spinner_video_preview = Gtk.Template.Child()
    stack = Gtk.Template.Child()
    stack_video_preview = Gtk.Template.Child()
    banner_no_gpu = Gtk.Template.Child()
    shortcut_controller = Gtk.Template.Child()
    config_sidebar: ConfigSidebar = Gtk.Template.Child()
    header_bar: Adw.HeaderBar = Gtk.Template.Child()
    button_toggle_fullscreen: Gtk.Button = Gtk.Template.Child()
    toast_overlay: Adw.ToastOverlay = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._frame_restorer_options: FrameRestorerOptions | None = None

        def on_passthrough(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_passthrough(object.get_property(spec.name))
        self.widget_video_preview.connect("notify::passthrough",on_passthrough)

        self._opened_file: Gio.File | None = None
        self.preview_close_handler_id = None

        self.fullscreen_mouse_activity_controller = None
        self.connect("notify::fullscreened", lambda object, spec: self.on_fullscreened(object.get_property(spec.name)))

        application = self.get_application()

        application.shortcuts_manager.register_group("preview", "Preview")
        def on_shortcut_preview_toggle(*args):
            if self.stack.get_visible_child_name() == "page_main" and self.stack_video_preview.get_visible_child() == self.widget_video_preview:
                self.toggle_button_preview_video_callback(self.toggle_button_preview_video)
        application.shortcuts_manager.add("preview", "toggle-preview", "p", on_shortcut_preview_toggle, "Enable/Disable preview mode")
        application.shortcuts_manager.add("preview", "toggle-fullscreen", "<Ctrl>f", self.toggle_fullscreen, "Enable/Disable fullscreen")

        self.connect("close-request", self.close)

        self.file_selection_view.connect("file-selected", lambda obj, file: self.open_file(file))
        self.video_export_view.connect("video-export-requested", lambda obj, file: self.on_video_export_requested(file))
        self.video_export_view.connect("video-export-dialog-opened", lambda *args: self.widget_video_preview.pause_if_currently_playing())
        
        self._config: Config | None = None

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
        if self._config.get_property('device') == 'cpu':
            self.banner_no_gpu.set_revealed(True)
        self.setup_config_signal_handlers()

    @GObject.Property(type=Gio.File)
    def opened_file(self):
        return self._opened_file

    @opened_file.setter
    def opened_file(self, value):
        self._opened_file = value

    @Gtk.Template.Callback()
    def button_export_video_callback(self, button_clicked):
        self.video_export_view.show_export_dialog()

    @Gtk.Template.Callback()
    def toggle_button_preview_video_callback(self, button_clicked):
        assert self._frame_restorer_options, "InvalidState: Preview/Passthrough button clicked but FrameRestorerOptions is null. The button should only be clickable if has been opened."
        self.frame_restorer_options = self._frame_restorer_options.with_passthrough(not self._frame_restorer_options.passthrough)

    @Gtk.Template.Callback()
    def button_toggle_fullscreen_callback(self, button_clicked):
        self.toggle_fullscreen()

    @property
    def frame_restorer_options(self):
        return self._frame_restorer_options

    @frame_restorer_options.setter
    def frame_restorer_options(self, value):
        self._frame_restorer_options = value
        if self.widget_video_preview:
            self.widget_video_preview.set_property('frame-restorer-options', self._frame_restorer_options)

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

        self._config.connect("notify::preview-buffer-duration", lambda object, spec: self.widget_video_preview.set_property('buffer-queue-min-thresh-time', object.get_property(spec.name)))

        def on_max_clip_duration(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_max_clip_length(self._config.max_clip_duration)
        self._config.connect("notify::max-clip-duration", on_max_clip_duration)

    def toggle_fullscreen(self, *args):
        if self.is_fullscreen():
            self.unfullscreen()
        else:
            self.fullscreen()

    def on_video_export_requested(self, file: Gio.File):
        self.stack.set_visible_child_name("file-export")
        self.widget_video_preview.close(block=True)
        self.video_export_view.start_export(file)

    def on_fullscreen_activity(self, fullscreen_activity: bool):
        if fullscreen_activity:
            self.header_bar.set_visible(True)
            self.set_cursor_from_name("default")
        else:
            self.header_bar.set_visible(False)
            self.set_cursor_from_name("none")
        self.widget_video_preview.on_fullscreen_activity(fullscreen_activity)

    def on_fullscreened(self, fullscreened: bool):
        if not self.stack.get_visible_child_name() == "page_main":
            return
        if fullscreened:
            self.fullscreen_mouse_activity_controller = FullscreenMouseActivityController(self)
            self.header_bar.set_visible(False)
            self.set_cursor_from_name("none")
            self.button_toggle_fullscreen.set_property("icon-name", "view-restore-symbolic")
        else:
            self.header_bar.set_visible(True)
            self.set_cursor_from_name("default")
            self.button_toggle_fullscreen.set_property("icon-name", "view-fullscreen-symbolic")
        self.widget_video_preview.on_fullscreened(fullscreened)
        self.fullscreen_mouse_activity_controller.on_fullscreened(fullscreened)
        self.fullscreen_mouse_activity_controller.connect("notify::fullscreen-activity", lambda object, spec: self.on_fullscreen_activity(object.get_property(spec.name)))

    def _show_spinner(self, *args):
        self.config_sidebar.set_property("disabled", True)
        self.toggle_button_preview_video.set_property("sensitive", False)
        self.stack_video_preview.set_visible_child(self.spinner_video_preview)

    def _show_video_preview(self, *args):
        self.config_sidebar.set_property("disabled", False)
        self.toggle_button_preview_video.set_property("sensitive", True)
        self.stack_video_preview.set_visible_child(self.widget_video_preview)
        self.widget_video_preview.grab_focus()

    def open_file(self, file: Gio.File):
        self.switch_to_main_view()
        self._show_spinner()
        file_changed = self._opened_file is not None

        if file_changed:
            def preview_open_file(*args):
                if self.preview_close_handler_id:
                    self.widget_video_preview.disconnect(self.preview_close_handler_id)
                    self.preview_close_handler_id = None
                self._open_file(file)

            self.preview_close_handler_id = self.widget_video_preview.connect("video-preview-close-done", preview_open_file)
            self.widget_video_preview.close_video_file()
        else:
            self.widget_video_preview.connect("video-preview-init-done", self._show_video_preview)
            self.widget_video_preview.connect("video-preview-reinit", self._show_spinner)
            self._open_file(file)

    def _open_file(self, file: Gio.File):
        self.opened_file = file
        self.set_title(os.path.basename(file.get_path()))
        self.config_sidebar.set_property("disabled", True)
        self.toggle_button_preview_video.set_property("sensitive", False)
        try:
            self.frame_restorer_options = FrameRestorerOptions(self.config.mosaic_restoration_model, self.config.mosaic_detection_model, video_utils.get_video_meta_data(self._opened_file.get_path()), self.config.device, self.config.max_clip_duration, self.config.preview_mode == 'mosaic-detection', False)
            self.widget_video_preview.open_video_file(self.opened_file, self.config.mute_audio)
        except Exception as e:
            self.toast_overlay.add_toast(Adw.Toast.new(f"Error opening file: {e}"))
            raise e

    def switch_to_main_view(self):
        self.stack.set_visible_child_name("page_main")
        self.button_export_video.set_sensitive(True)

    def close(self, *args):
        self.widget_video_preview.close()