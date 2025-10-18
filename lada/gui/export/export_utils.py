from gi.repository import Gtk, Adw, GObject, Gio


from lada.lib import video_utils
from lada.lib import VideoMetadata

MIN_VISIBLE_PROGRESS_FRACTION = 0.01

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

def get_video_metadata_string(file: Gio.File):
    meta_data = video_utils.get_video_meta_data(file.get_path())
    return _("Duration: {duration}, Resolution: {resolution}, FPS: {fps}").format(
        duration=_format_duration(meta_data.duration),
        resolution=f"{meta_data.video_width}x{meta_data.video_height}",
        fps=f"{meta_data.video_fps:.2f}")

def open_error_dialog(parent: Gtk.Widget, filename:str, details:str|None):
    extra_child = None
    if details:
        textview = Gtk.TextView()
        PADDING = 6
        textview.set_left_margin(PADDING)
        textview.set_right_margin(PADDING)
        textview.set_top_margin(PADDING)
        textview.set_bottom_margin(PADDING)
        textbuffer = textview.get_buffer()
        textbuffer.set_text(details.strip())

        scrolledwindow = Gtk.ScrolledWindow()
        scrolledwindow.props.hexpand = True
        scrolledwindow.props.vexpand = True
        scrolledwindow.set_child(textview)
        extra_child = scrolledwindow

    dialog = Adw.AlertDialog(
        heading=_("Restoration failed"),
        body=_("Error while processing <span><tt>{filename}</tt></span>").format(filename=filename),
        close_response="okay",
        extra_child=extra_child,
        body_use_markup=True
    )

    def on_response_selected(_dialog, task):
        _dialog.choose_finish(task)

    dialog.add_response("okay", _("Okay"))

    dialog.choose(parent, None, on_response_selected)

class ExportItemState(GObject.GEnum):
    QUEUED = 0
    PROCESSING = 1
    FINISHED = 2
    FAILED = 3

class RemainingProcessingTimeEstimator:
    def __init__(self, video_metadata: VideoMetadata):
        self.frame_processing_durations_buffer = []
        self.video_metadata = video_metadata
        self.frame_processing_durations_buffer_min_len = min(video_metadata.frames_count - 1, int(video_metadata.video_fps * 15))
        self.frame_processing_durations_buffer_max_len = min(video_metadata.frames_count - 1, int(video_metadata.video_fps * 120))

    def _convert_seconds_remaining_to_text_label(self, seconds):
        minutes = int(seconds / 60)
        hours = int(minutes / 60)
        seconds = seconds % 60
        minutes = minutes % 60
        hours, minutes, seconds = int(hours), int(minutes), int(seconds)
        if hours == 0 and minutes == 0:
            return f"{seconds:02d}"
        elif hours == 0:
            return f"{minutes}:{seconds:02d}"
        else:
            return f"{hours}:{minutes:02d}:{seconds:02d}"

    def _get_mean_processing_duration(self):
        return sum(self.frame_processing_durations_buffer) / len(self.frame_processing_durations_buffer)

    def add_processing_duration(self, duration):
        if len(self.frame_processing_durations_buffer) >= self.frame_processing_durations_buffer_max_len:
            self.frame_processing_durations_buffer.pop(0)
        self.frame_processing_durations_buffer.append(duration)

    def get_time_remaining(self, frame_num) -> str:
        if len(self.frame_processing_durations_buffer) < self.frame_processing_durations_buffer_min_len:
            return _("Estimating…")
        frames_remaining = self.video_metadata.frames_count - (frame_num + 1)
        seconds_remaining = frames_remaining * self._get_mean_processing_duration()
        return self._convert_seconds_remaining_to_text_label(seconds_remaining)