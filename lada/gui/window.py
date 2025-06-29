import os.path
import pathlib
from threading import Thread

from gi.repository import Adw, Gtk, Gio, Gdk
import lada.gui.video_preview
import lada.gui.video_export
from lada.gui.config import CONFIG
from lada.gui.config_sidebar import ConfigSidebar
from lada.gui.frame_restorer_provider import FrameRestorerOptions
from lada.gui.fullscreen_mouse_activity_controller import FullscreenMouseActivityController
from lada.lib import video_utils

here = pathlib.Path(__file__).parent.resolve()

@Gtk.Template(filename=here / 'window.ui')
class MainWindow(Adw.ApplicationWindow):
    __gtype_name__ = 'MainWindow'

    button_open_file = Gtk.Template.Child()
    button_export_video = Gtk.Template.Child()
    toggle_button_preview_video = Gtk.Template.Child()
    widget_video_preview = Gtk.Template.Child()
    widget_video_export = Gtk.Template.Child()
    spinner_video_preview = Gtk.Template.Child()
    stack = Gtk.Template.Child()
    stack_video_preview = Gtk.Template.Child()
    banner_no_gpu = Gtk.Template.Child()
    shortcut_controller = Gtk.Template.Child()
    config_sidebar: ConfigSidebar = Gtk.Template.Child()
    header_bar: Adw.HeaderBar = Gtk.Template.Child()
    button_toggle_fullscreen: Gtk.Button = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._frame_restorer_options: FrameRestorerOptions | None = None

        # init drag-drop files
        drop_target = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        def on_connect_drop(drop_target, file: Gio.File, x, y):
            self.open_file(file)
        drop_target.connect("drop", on_connect_drop)
        self.stack.add_controller(drop_target)

        if self.config_sidebar.get_property('device') == 'cpu':
            self.banner_no_gpu.set_revealed(True)

        def on_preview_mode(*args):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_mosaic_detection(CONFIG.preview_mode == 'mosaic-detection')
        self.config_sidebar.connect("notify::preview-mode", on_preview_mode)

        def on_passthrough(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_passthrough(object.get_property(spec.name))
        self.widget_video_preview.connect("notify::passthrough",on_passthrough)

        def on_device(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_device(CONFIG.device)
        self.config_sidebar.connect("notify::device", on_device)

        def on_mosaic_restoration_model(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_mosaic_restoration_model_name(CONFIG.mosaic_restoration_model)
        self.config_sidebar.connect("notify::mosaic-restoration-model", on_mosaic_restoration_model)

        def on_mosaic_detection_model(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_mosaic_detection_model_name(CONFIG.mosaic_detection_model)
        self.config_sidebar.connect("notify::mosaic-detection-model", on_mosaic_detection_model)

        self.config_sidebar.connect("notify::preview-buffer-duration", lambda object, spec: self.widget_video_preview.set_property('buffer-queue-min-thresh-time', object.get_property(spec.name)))

        def on_max_clip_duration(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_max_clip_length(CONFIG.max_clip_duration)
        self.config_sidebar.connect("notify::max-clip-duration", on_max_clip_duration)

        self.opened_file: Gio.File = None
        self.preview_close_handler_id = None

        self.fullscreen_mouse_activity_controller = None
        self.connect("notify::fullscreened", lambda object, spec: self.on_fullscreened(object.get_property(spec.name)))

        application = self.get_application()

        application.shortcuts.register_group("files", "Files")
        def on_shortcut_open_file(*args):
            self.show_open_dialog()
        application.shortcuts.add("files", "open-file", "o", on_shortcut_open_file, "Open a video file")
        def on_shortcut_export_file(*args):
            if self.stack.get_visible_child_name() == "page_main" and self.stack_video_preview.get_visible_child() == self.widget_video_preview:
                self.show_export_dialog()
        application.shortcuts.add("files", "export-file", "e", on_shortcut_export_file, "Export recovered video")

        application.shortcuts.register_group("preview", "Preview")
        def on_shortcut_preview_toggle(*args):
            if self.stack.get_visible_child_name() == "page_main" and self.stack_video_preview.get_visible_child() == self.widget_video_preview:
                self.toggle_button_preview_video_callback(self.toggle_button_preview_video)
        application.shortcuts.add("preview", "toggle-preview", "p", on_shortcut_preview_toggle, "Enable/Disable preview mode")
        application.shortcuts.add("preview", "toggle-fullscreen", "<Ctrl>f", self.toggle_fullscreen, "Enable/Disable fullscreen")

        self.connect("close-request", self.close)

    @Gtk.Template.Callback()
    def button_open_file_callback(self, button_clicked):
        self.show_open_dialog()

    @Gtk.Template.Callback()
    def button_export_video_callback(self, button_clicked):
        self.show_export_dialog()

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

    def toggle_fullscreen(self, *args):
        if self.is_fullscreen():
            self.unfullscreen()
        else:
            self.fullscreen()

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

    def show_open_dialog(self):
        file_dialog = Gtk.FileDialog()
        video_file_filter = Gtk.FileFilter()
        video_file_filter.add_mime_type("video/*")
        file_dialog.set_default_filter(video_file_filter)
        file_dialog.set_title("Select a video file")
        file_dialog.open(callback=lambda dialog, result: self.open_file(dialog.open_finish(result)))

    def show_export_dialog(self):
        self.widget_video_preview.pause_if_currently_playing()
        file_dialog = Gtk.FileDialog()
        video_file_filter = Gtk.FileFilter()
        video_file_filter.add_mime_type("video/*")
        file_dialog.set_default_filter(video_file_filter)
        file_dialog.set_title("Save restored video file")
        file_dialog.set_initial_folder(self.opened_file.get_parent())
        file_dialog.set_initial_name(f"{os.path.splitext(self.opened_file.get_basename())[0]}.restored.mp4")
        file_dialog.save(callback=lambda dialog, result: self.start_export(dialog.save_finish(result)))

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
        file_changed = self.opened_file is not None

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
        if not CONFIG.loaded: CONFIG.load_config()
        self.frame_restorer_options = FrameRestorerOptions(CONFIG.mosaic_restoration_model, CONFIG.mosaic_detection_model, video_utils.get_video_meta_data(self.opened_file.get_path()), CONFIG.device, CONFIG.max_clip_duration, CONFIG.preview_mode == 'mosaic-detection', False)
        self.widget_video_preview.open_video_file(self.opened_file, CONFIG.mute_audio)

    def start_export(self, file: Gio.File):
        self.stack.set_visible_child_name("file-export")
        def run():
            self.widget_video_preview.close(block=True)
            if not CONFIG.loaded: CONFIG.load_config()
            self.frame_restorer_options = FrameRestorerOptions(CONFIG.mosaic_restoration_model, CONFIG.mosaic_detection_model, video_utils.get_video_meta_data(self.opened_file.get_path()), CONFIG.device, CONFIG.max_clip_duration, False, False)
            self.widget_video_export.export_video(file.get_path(), CONFIG.export_codec, CONFIG.export_crf, self._frame_restorer_options)
        Thread(target=run).start()

    def switch_to_main_view(self):
        self.stack.set_visible_child_name("page_main")
        self.button_export_video.set_sensitive(True)

    def close(self, *args):
        self.widget_video_preview.close()