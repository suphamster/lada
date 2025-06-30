import pathlib

from gi.repository import Adw, Gtk, Gio, Gdk, GObject
from lada.gui.frame_restorer_provider import FrameRestorerOptions
from lada.gui.shortcuts import ShortcutsManager

here = pathlib.Path(__file__).parent.resolve()

@Gtk.Template(filename=here / 'file_selection_view.ui')
class FileSelectionView(Gtk.Widget):
    __gtype_name__ = 'FileSelectionView'

    button_open_file: Gtk.Button = Gtk.Template.Child()
    status_page: Adw.StatusPage = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._frame_restorer_options: FrameRestorerOptions | None = None

        # init drag-drop files
        drop_target = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        drop_target.connect("drop", lambda _drop_target, file, x, y: self.emit("file-selected", file))
        self.add_controller(drop_target)

        self._shortcuts_manager: ShortcutsManager | None = None
        self._window_title: str | None = None

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

    @Gtk.Template.Callback()
    def button_open_file_callback(self, button_clicked):
        self.show_open_dialog()

    @GObject.Signal(name="file-selected")
    def file_selected_signal(self, file: Gio.File):
        pass

    def _setup_shortcuts(self):
        self._shortcuts_manager.register_group("files", "Files")
        self._shortcuts_manager.add("files", "open-file", "o", lambda *args: self.show_open_dialog(), "Open a video file")

    def show_open_dialog(self):
        file_dialog = Gtk.FileDialog()
        video_file_filter = Gtk.FileFilter()
        video_file_filter.add_mime_type("video/*")
        file_dialog.set_default_filter(video_file_filter)
        file_dialog.set_title("Select a video file")
        file_dialog.open(callback=lambda _file_dialog, result: self.emit("file-selected", file_dialog.open_finish(result)))
