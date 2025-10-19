import logging
import pathlib
import threading

from gi.repository import Gtk, Gio, GObject, GLib
from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.export import export_utils
from lada.gui.export.export_item_data import ExportItemData
from lada.gui.export.export_utils import MIN_VISIBLE_PROGRESS_FRACTION, ExportItemState

here = pathlib.Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'export_single_file_status_page.ui'))
class ExportSingleFileStatusPage(Gtk.Widget):
    __gtype_name__ = 'ExportSingleFileStatusPage'

    status_page = Gtk.Template.Child()
    progress_bar_file_export_status_page: Gtk.ProgressBar = Gtk.Template.Child()
    button_open: Gtk.Button = Gtk.Template.Child()
    button_show_error: Gtk.Button = Gtk.Template.Child()
    button_start_export: Gtk.Button = Gtk.Template.Child()
    label_meta_data: Gtk.Label = Gtk.Template.Child()
    label_file_name: Gtk.Label = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.item: ExportItemData | None = None

    @GObject.Signal(name="start-export-requested")
    def start_export_requested_signal(self):
        pass

    @Gtk.Template.Callback()
    def button_start_export_callback(self, button_clicked):
        assert self.item.state == ExportItemState.QUEUED

        self.emit("start-export-requested")

    @Gtk.Template.Callback()
    def button_show_error_status_page_callback(self, button_clicked):
        assert self.item.state == ExportItemState.FAILED

        export_utils.open_error_dialog(self, self.item.orig_file.get_basename(), self.item.error_details)

    def show_video_export_started(self, save_file: Gio.File):
        self.status_page.set_title(_("Restoring video…"))
        self.status_page.set_icon_name("cafe-symbolic")
        self.progress_bar_file_export_status_page.set_fraction(MIN_VISIBLE_PROGRESS_FRACTION)
        self.progress_bar_file_export_status_page.set_visible(True)
        self.progress_bar_file_export_status_page.set_show_text(True)
        self.progress_bar_file_export_status_page.set_text(_("Processing {done_percent}%, Time remaining: Estimating…").format(done_percent=int(self.progress_bar_file_export_status_page.get_fraction() * 100)))
        self.button_start_export.set_visible(False)
        file_launcher = Gtk.FileLauncher(
            always_ask=False,
            file=save_file
        )
        self.button_open.connect("clicked", lambda _: file_launcher.launch())

    def on_video_export_finished(self):
        self.status_page.set_title(_("Finished video restoration!"))
        self.status_page.set_icon_name("check-round-outline2-symbolic")
        self.progress_bar_file_export_status_page.set_visible(False)
        self.button_open.set_visible(True)

    def on_video_export_failed(self):
        self.status_page.set_title(_("Restoration failed"))
        self.status_page.set_icon_name("exclamation-mark-symbolic")
        self.progress_bar_file_export_status_page.set_visible(False)
        self.button_show_error.set_visible(True)

    def on_video_export_progress(self, progress:float, time_remaining: str):
        self.progress_bar_file_export_status_page.set_fraction(max(MIN_VISIBLE_PROGRESS_FRACTION, progress))
        self.progress_bar_file_export_status_page.set_text(_("Processing {done_percent}%, Time remaining: {remaining_time}").format(done_percent=int(self.progress_bar_file_export_status_page.get_fraction() * 100), remaining_time=time_remaining))


    def on_add_file(self, item: ExportItemData):
        self.item = item
        self.status_page.set_title(_("Export video"))
        self.status_page.set_icon_name("arrow-pointing-away-from-line-right-symbolic")
        self.progress_bar_file_export_status_page.set_visible(False)
        self.button_start_export.set_visible(True)
        self.button_show_error.set_visible(False)
        self.button_open.set_visible(False)
        self.label_meta_data.set_visible(True)
        self.label_file_name.set_label(self.item.orig_file.get_basename())

        def update_label_with_video_metadata():
            label = export_utils.get_video_metadata_string(self.item.orig_file)
            GLib.idle_add(lambda: self.label_meta_data.set_label(label))

        threading.Thread(target=update_label_with_video_metadata).start()

    def set_button_start_restore_label(self, value: str):
        self.button_start_export.set_label(value)