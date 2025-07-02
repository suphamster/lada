import os
import pathlib
import threading

from gi.repository import Adw, Gtk, Gio, Gdk, GLib
from lada.gui.file_selection_view import FileSelectionView
from lada.gui.video_export_view import VideoExportView
from lada.gui.video_preview_view import VideoPreviewView

here = pathlib.Path(__file__).parent.resolve()

@Gtk.Template(filename=here / 'window.ui')
class MainWindow(Adw.ApplicationWindow):
    __gtype_name__ = 'MainWindow'

    file_selection_view: FileSelectionView = Gtk.Template.Child()
    video_export_view: VideoExportView = Gtk.Template.Child()
    video_preview_view: VideoPreviewView = Gtk.Template.Child()
    stack = Gtk.Template.Child()
    shortcut_controller = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        drop_target = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        drop_target.connect("drop", self.on_file_drop)
        self.add_controller(drop_target)

        self.connect("close-request", self.close)
        self.file_selection_view.connect("file-selected", lambda obj, file: self.on_file_selected(file))
        self.video_preview_view.connect("video-export-requested", lambda obj, source_file, save_file: self.on_video_export_requested(source_file, save_file))
        self.video_preview_view.connect("toggle-fullscreen-requested", lambda *args: self.on_toggle_fullscreen())
        self.connect("notify::fullscreened", lambda object, spec: self.on_fullscreened(object.get_property(spec.name)))

    def on_video_export_requested(self, source_file: Gio.File, save_file: Gio.File):
        self.stack.set_visible_child_name("file-export")
        def run():
            self.video_preview_view.close(block=True)
            GLib.idle_add(lambda: self.video_export_view.start_export(source_file, save_file))
        threading.Thread(target=run).start()

    def on_file_selected(self, file: Gio.File):
        self.stack.set_visible_child_name("video-preview")
        self.set_title(os.path.basename(file.get_path()))
        self.video_preview_view.open_file(file)

    def on_file_drop(self, _drop_target, file, x, y):
        name = self.stack.get_visible_child_name()
        if name in ("file-selection", "video-preview") or name == 'file-export' and not self.video_export_view.export_in_progress:
            self.on_file_selected(file)

    def on_fullscreened(self, fullscreened: bool):
        if self.stack.get_visible_child_name() == "video-preview":
            self.video_preview_view.on_fullscreened(fullscreened)

    def on_toggle_fullscreen(self):
        if self.is_fullscreen():
            self.unfullscreen()
        else:
            self.fullscreen()

    def close(self, *args):
        self.video_preview_view.close()