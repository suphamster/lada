from gi.repository import GObject, Gio

from lada.gui.export.export_item_row import ExportItemState

class ExportItemData(GObject.Object):
    __gtype_name__ = 'ExportItemData'

    def __init__(self, orig_file: Gio.File, restored_file: Gio.File):
        super().__init__()

        self._progress: float = 0.0
        self._orig_file: Gio.File = orig_file
        self._restored_file: Gio.File = restored_file
        self._state: ExportItemState = ExportItemState.QUEUED

    @GObject.Property(type=float)
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, value):
        self._progress = value

    @GObject.Property(type=ExportItemState, default=ExportItemState.QUEUED)
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @GObject.Property(type=Gio.File)
    def orig_file(self):
        return self._orig_file

    @GObject.Property(type=Gio.File)
    def restored_file(self):
        return self._restored_file

    @restored_file.setter
    def restored_file(self, value):
        self._restored_file = value

    def __repr__(self):
        return f"{{{self._orig_file.get_basename()}, {self._restored_file.get_basename()}, {self._state}, {self._progress}}}"
