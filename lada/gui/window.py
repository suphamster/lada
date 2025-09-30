import pathlib
import threading

from gi.repository import Adw, Gtk, Gio, GLib, GObject

from lada.gui.config.config import Config
from lada.gui.fileselection.file_selection_view import FileSelectionView
from lada.gui.export.export_view import ExportView
from lada.gui.preview.preview_view import PreviewView

here = pathlib.Path(__file__).parent.resolve()

@Gtk.Template(filename=here / 'window.ui')
class MainWindow(Adw.ApplicationWindow):
    __gtype_name__ = 'MainWindow'

    file_selection_view: FileSelectionView = Gtk.Template.Child()
    export_view: ExportView = Gtk.Template.Child()
    preview_view: PreviewView = Gtk.Template.Child()
    view_stack: Adw.ViewStack = Gtk.Template.Child()
    stack: Gtk.Stack = Gtk.Template.Child()
    shortcut_controller = Gtk.Template.Child()

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._config: Config | None

        self.set_title("Lada")

        self.files = []

        self.connect("close-request", self.close)
        self.file_selection_view.connect("files-selected", lambda obj, files: self.on_files_selected(files))
        self.preview_view.connect("toggle-fullscreen-requested", lambda *args: self.on_toggle_fullscreen())
        self.connect("notify::fullscreened", lambda object, spec: self.on_fullscreened(object.get_property(spec.name)))

        self.export_view.props.view_stack = self.view_stack
        self.export_view.connect("video-export-requested", lambda obj, source_file, save_file: self.on_video_export_requested(source_file, save_file))
        self.preview_view.props.view_stack = self.view_stack

    def on_video_export_requested(self, source_file: Gio.File, save_file: Gio.File):
        self.stack.props.visible_child_name = "main"
        self.view_stack.props.visible_child_name = "export"
        def run():
            self.preview_view.close(block=True)
            GLib.idle_add(lambda: self.export_view.start_export(source_file, save_file))
        threading.Thread(target=run).start()

    def on_files_selected(self, files: list[Gio.File]):
        self.stack.props.visible_child_name = "main"
        self.view_stack.props.visible_child_name = "preview" if self._config.initial_view == "preview" else "export"
        self.files = files
        self.preview_view.add_files(files)
        if self.view_stack.props.visible_child_name == "preview":
            self.preview_view.play_file(0)
        self.export_view.open_files(files)

    def on_fullscreened(self, fullscreened: bool):
        if self.stack.props.visible_child_name == "main" and self.view_stack.props.visible_child_name == "preview":
            self.preview_view.on_fullscreened(fullscreened)

    def on_toggle_fullscreen(self):
        if self.is_fullscreen():
            self.unfullscreen()
        else:
            self.fullscreen()

    def close(self, *args):
        self.preview_view.close()