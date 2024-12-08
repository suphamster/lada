import os.path
import pathlib

from gi.repository import Adw, Gtk, Gio, Gdk
import lada.gui.video_preview
from lada.gui.config_sidebar import ConfigSidebar

here = pathlib.Path(__file__).parent.resolve()

VIDEO_MIME_TYPES = [
    'video/mp4',
    'video/x-matroska',
]

@Gtk.Template(filename=here / 'window.ui')
class MainWindow(Adw.ApplicationWindow):
    __gtype_name__ = 'MainWindow'

    button_open_file = Gtk.Template.Child()
    button_export_video = Gtk.Template.Child()
    toggle_button_preview_video = Gtk.Template.Child()
    widget_video_preview = Gtk.Template.Child()
    spinner_video_preview = Gtk.Template.Child()
    stack = Gtk.Template.Child()
    stack_video_preview = Gtk.Template.Child()
    progress_bar_file_export = Gtk.Template.Child()
    status_page_export_video = Gtk.Template.Child()
    banner_no_gpu = Gtk.Template.Child()
    shortcut_controller = Gtk.Template.Child()
    config_sidebar: ConfigSidebar = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # init drag-drop files
        drop_target = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        def on_connect_drop(drop_target, file: Gio.File, x, y):
            self.open_file(file)
        drop_target.connect("drop", on_connect_drop)
        self.stack.add_controller(drop_target)

        if self.config_sidebar.get_property('device') == 'cpu':
            self.banner_no_gpu.set_revealed(True)

        self.widget_video_preview.set_property('mosaic-detection', self.config_sidebar.get_property('preview-mode') == 'mosaic-detection')
        self.config_sidebar.connect("notify::preview-mode", lambda object, spec: self.widget_video_preview.set_property('mosaic-detection', object.get_property(spec.name) == 'mosaic-detection'))

        self.widget_video_preview.set_property('device', self.config_sidebar.get_property('device'))
        self.config_sidebar.connect("notify::device", lambda object, spec: self.widget_video_preview.set_property('device', object.get_property(spec.name)))

        self.widget_video_preview.set_property('mosaic-restoration-model', self.config_sidebar.get_property('mosaic-restoration-model'))
        self.config_sidebar.connect("notify::mosaic-restoration-model", lambda object, spec: self.widget_video_preview.set_property('mosaic-restoration-model', object.get_property(spec.name)))

        self.widget_video_preview.set_property('buffer-queue-min-thresh-time', self.config_sidebar.get_property('preview-buffer-duration'))
        self.config_sidebar.connect("notify::preview-buffer-duration", lambda object, spec: self.widget_video_preview.set_property('buffer-queue-min-thresh-time', object.get_property(spec.name)))

        self.widget_video_preview.set_property('max-clip-length', self.config_sidebar.get_property('max-clip-duration'))
        self.config_sidebar.connect("notify::max-clip-duration", lambda object, spec: self.widget_video_preview.set_property('max-clip-length', object.get_property(spec.name)))

        self.widget_video_preview.set_property('mosaic-cleaning', self.config_sidebar.get_property('mosaic-pre-cleaning'))
        self.config_sidebar.connect("notify::mosaic-pre-cleaning", lambda object, spec: self.widget_video_preview.set_property('mosaic-cleaning', object.get_property(spec.name)))

        self.opened_file: Gio.File = None

        application = self.get_application()
        application.shortcuts.register_group("files", "Files")
        application.shortcuts.add("files", "open-file", "o", lambda *args: self.show_open_dialog(), "Open a video file")
        application.shortcuts.add("files", "export-file", "e", lambda *args: self.show_export_dialog(), "Export recovered video")
        application.shortcuts.register_group("preview", "Preview")
        application.shortcuts.add("preview", "toggle-preview", "p", lambda *args: self.toggle_button_preview_video_callback(self.toggle_button_preview_video), "Enable/Disable preview mode")

    @Gtk.Template.Callback()
    def button_open_file_callback(self, button_clicked):
        self.show_open_dialog()

    @Gtk.Template.Callback()
    def button_export_video_callback(self, button_clicked):
        self.show_export_dialog()

    @Gtk.Template.Callback()
    def toggle_button_preview_video_callback(self, button_clicked):
        preview_active = button_clicked.get_property("active")
        self.widget_video_preview.set_property('passthrough', not preview_active)

    def show_open_dialog(self):
        file_dialog = Gtk.FileDialog()
        video_file_filter = Gtk.FileFilter()
        video_file_filter.add_mime_type("video/*")
        file_dialog.set_default_filter(video_file_filter)
        file_dialog.set_title("Select a video file")
        file_dialog.open(callback=lambda dialog, result: self.open_file(dialog.open_finish(result)))

    def show_export_dialog(self):
        file_dialog = Gtk.FileDialog()
        video_file_filter = Gtk.FileFilter()
        video_file_filter.add_mime_type("video/*")
        file_dialog.set_default_filter(video_file_filter)
        file_dialog.set_title("Save restored video file")
        file_dialog.set_initial_folder(self.opened_file.get_parent())
        file_dialog.set_initial_name(f"{os.path.splitext(self.opened_file.get_basename())[0]}.restored.mp4")
        file_dialog.save(callback=lambda dialog, result: self.start_export(dialog.save_finish(result)))

    def open_file(self, file: Gio.File):
        self.opened_file = file
        self.set_title(os.path.basename(file.get_path()))
        self.switch_to_main_view()

        if self.stack_video_preview.get_visible_child() == self.widget_video_preview:
            self.stack_video_preview.set_visible_child(self.spinner_video_preview)

        def show_video_preview(obj):
            self.stack_video_preview.set_visible_child(self.widget_video_preview)
            self.widget_video_preview.grab_focus()

        self.widget_video_preview.connect("video-preview-init-done", show_video_preview)
        self.widget_video_preview.open_video_file(file, self.config_sidebar.get_property("mute_audio"))

    def start_export(self, file: Gio.File):
        self.stack.set_visible_child_name("file-export")

        def show_video_export_success(obj):
            print("finished exporting")
            self.status_page_export_video.set_title("Finished video restoration!")
            self.status_page_export_video.set_icon_name("check-round-outline2-symbolic")

        def on_video_export_progress(obj, progress):
            self.progress_bar_file_export.set_fraction(progress)

        self.widget_video_preview.connect("video-export-finished", show_video_export_success)
        self.widget_video_preview.connect("video-export-progress", on_video_export_progress)
        self.widget_video_preview.export_video(file.get_path(), self.config_sidebar.get_property("export_codec"), self.config_sidebar.get_property("export_crf"))

    def switch_to_main_view(self):
        self.stack.set_visible_child_name("page_main")
        self.button_export_video.set_sensitive(True)