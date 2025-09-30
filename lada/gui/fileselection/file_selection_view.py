import logging
import pathlib

from gi.repository import Adw, Gtk, Gio, GObject, GLib, Gdk
from gettext import gettext as _
from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.frame_restorer_provider import FrameRestorerOptions
from lada.gui.shortcuts import ShortcutsManager

here = pathlib.Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(filename=here / 'file_selection_view.ui')
class FileSelectionView(Gtk.Widget):
    __gtype_name__ = 'FileSelectionView'

    button_open_file: Gtk.Button = Gtk.Template.Child()
    status_page: Adw.StatusPage = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._frame_restorer_options: FrameRestorerOptions | None = None
        self._shortcuts_manager: ShortcutsManager | None = None
        self._window_title: str | None = None

        self._setup_drop()

        logo_image = Gtk.Image.new_from_resource("/io/github/ladaapp/lada/icons/128x128/lada-logo-gray.png")
        self.status_page.set_paintable(logo_image.get_paintable())

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

    @GObject.Signal(name="files-selected", arg_types=(GObject.TYPE_PYOBJECT,))
    def files_selected_signal(self, files: list[Gio.File]):
        pass

    def _setup_drop(self):
        drop_target = Gtk.DropTarget.new(Gio.File, actions=Gdk.DragAction.COPY)
        drop_target.set_gtypes((Gdk.FileList,))
        def on_file_drop(_drop_target, files: list[Gio.File], x, y):
            filtered_files = utils.filter_video_files(files)
            if len(filtered_files) > 0:
                self.emit("files-selected", filtered_files)
        drop_target.connect("drop", on_file_drop)
        self.add_controller(drop_target)

    def _setup_shortcuts(self):
        self._shortcuts_manager.register_group("files", "Files")
        self._shortcuts_manager.add("files", "open-file", "o", lambda *args: self.show_open_dialog(), "Open a video file")

    def show_open_dialog(self):
        file_dialog = Gtk.FileDialog()
        video_file_filter = Gtk.FileFilter()
        video_file_filter.add_mime_type("video/*")
        file_dialog.set_default_filter(video_file_filter)
        file_dialog.set_title(_("Select one or multiple video files"))
        def on_open_multiple(_file_dialog, result):
            try:
                selected_files = _file_dialog.open_multiple_finish(result)
                if len(selected_files) > 0:
                    self.emit("files-selected", selected_files)
            except GLib.Error as error:
                if error.message == "Dismissed by user":
                    logger.debug("FileDialog cancelled: Dismissed by user")
                else:
                    logger.error(f"Error opening file: {error.message}")
        file_dialog.open_multiple(callback=on_open_multiple)
