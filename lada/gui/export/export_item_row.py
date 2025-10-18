import logging
import pathlib
import threading

from gi.repository import Adw, Gtk, Gio, GObject, GLib

from lada import LOG_LEVEL
from lada.gui import utils
from lada.lib import video_utils

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class ExportItemState(GObject.GEnum):
    QUEUED = 0
    PROCESSING = 1
    FINISHED = 2
    FAILED = 3

MIN_VISIBLE_PROGRESS_FRACTION = 0.01

def get_video_metadata_string(file: Gio.File):
    meta_data = video_utils.get_video_meta_data(file.get_path())
    return _("Duration: {duration}, Resolution: {resolution}, FPS: {fps}").format(
        duration=_format_duration(meta_data.duration),
        resolution=f"{meta_data.video_width}x{meta_data.video_height}",
        fps=f"{meta_data.video_fps:.2f}")

def _format_duration(duration_s):
    if not duration_s or duration_s == -1:
        return ""
    seconds = int(duration_s)
    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    seconds = seconds % 60
    minutes = minutes % 60
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    time = f"{minutes}:{seconds:02d}" if hours == 0 else f"{hours}:{minutes:02d}:{seconds:02d}"
    return time

@Gtk.Template(string=utils.translate_ui_xml(here / 'export_item_row.ui'))
class ExportItemRow(Adw.PreferencesRow):
    __gtype_name__ = "ExportItemRow"

    progressbar: Gtk.ProgressBar = Gtk.Template.Child()
    button_open: Gtk.Button = Gtk.Template.Child()
    button_remove: Gtk.Button = Gtk.Template.Child()

    def __init__(self, original_file: Gio.File, restored_file: Gio.File, **kwargs) -> None:
        super().__init__(**kwargs)
        self._restored_file = restored_file
        self._attach_file_launcher_to_open_button()

        self.original_file = original_file
        self.set_title(original_file.get_basename())
        self._progress: float = 0.0
        self._state: ExportItemState = ExportItemState.QUEUED
        self._subtitle = ""
        self._time_remaining = None

        def update_title_with_video_metadata():
            subtitle = get_video_metadata_string(original_file)
            GLib.idle_add(lambda: self.set_property("subtitle", subtitle))
        threading.Thread(target=update_title_with_video_metadata).start()

    @GObject.Property(type=float)
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, value: float):
        self._progress = value
        self.progressbar.set_fraction(max(MIN_VISIBLE_PROGRESS_FRACTION, value) if self._state in [ExportItemState.PROCESSING, ExportItemState.FAILED] else value)

    @GObject.Property(type=str)
    def time_remaining(self):
        return self._time_remaining

    @time_remaining.setter
    def time_remaining(self, value: str):
        if not value:
            return
        self._time_remaining = value
        self.progressbar.set_text(_("Processing {done_percent}%, Time remaining: {remaining_time}").format(done_percent=int(self.progressbar.get_fraction() * 100), remaining_time=value))

    @GObject.Property(type=ExportItemState, default=ExportItemState.QUEUED)
    def state(self):
        return self._state

    @state.setter
    def state(self, value: ExportItemState):
        self._state = value
        if value == ExportItemState.FINISHED:
            self.progressbar.add_css_class("finished")
            self.button_open.set_visible(True)
            self.button_remove.set_visible(True)
            self.progressbar.set_text(_("Finished"))
            self.progressbar.set_show_text(True)
        elif value == ExportItemState.QUEUED:
            self.button_open.set_visible(False)
            self.button_remove.set_visible(True)
            self.progressbar.set_show_text(False)
        elif value == ExportItemState.PROCESSING:
            self.button_open.set_visible(False)
            self.button_remove.set_visible(False)
            self.progressbar.set_fraction(MIN_VISIBLE_PROGRESS_FRACTION)
            self.progressbar.set_text(_("Processing {done_percent}%, Time remaining: Estimating…").format(done_percent=int(self.progressbar.get_fraction() * 100)))
            self.progressbar.set_show_text(True)
        elif value == ExportItemState.FAILED:
            self.button_open.set_visible(False)
            self.button_remove.set_visible(True)
            self.progressbar.add_css_class("failed")
            self.progressbar.set_text(_("Failed"))
            self.progressbar.set_show_text(True)
        else:
            logger.error("Unhandled enum state", value)

    @GObject.Property(type=str)
    def subtitle(self):
        return self._subtitle

    @subtitle.setter
    def subtitle(self, value: str):
        self._subtitle = value

    @GObject.Property(type=Gio.File)
    def restored_file(self):
        return self._restored_file

    @restored_file.setter
    def restored_file(self, value: Gio.File):
        if self._restored_file.get_path() != value.get_path():
            self._restored_file = value
            self._attach_file_launcher_to_open_button()

    def _attach_file_launcher_to_open_button(self):
        file_launcher = Gtk.FileLauncher(
            always_ask=False,
            file=self._restored_file,
        )

        self.button_open.connect("clicked", lambda _: file_launcher.launch())

    @Gtk.Template.Callback()
    def button_remove_callback(self, button):
        self.emit("remove-requested")

    @GObject.Signal(name="remove-requested")
    def video_export_requested_signal(self):
        pass