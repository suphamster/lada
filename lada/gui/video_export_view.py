import logging
import os
import pathlib
import tempfile
import threading

from gi.repository import Gtk, GObject, Gio

from lada.gui.config import CONFIG
from lada.gui.shortcuts import ShortcutsManager
from lada.lib import audio_utils, video_utils
from lada import LOG_LEVEL
from lada.gui.frame_restorer_provider import FrameRestorerOptions, FRAME_RESTORER_PROVIDER

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(filename=here / 'video_export_view.ui')
class VideoExportView(Gtk.Widget):
    __gtype_name__ = 'VideoExportView'

    status_page = Gtk.Template.Child()
    progress_bar_file_export = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._shortcuts_manager: ShortcutsManager | None = None
        self._window_title: str | None = None
        self._opened_file: Gio.File | None = None

        self.connect("video-export-finished", self.show_video_export_success)
        self.connect("video-export-progress", self.on_video_export_progress)

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

    @GObject.Property(type=Gio.File)
    def opened_file(self):
        return self._opened_file

    @opened_file.setter
    def opened_file(self, value):
        self._opened_file = value

    @GObject.Signal(name="video-export-dialog-opened")
    def video_export_dialog_opened_signal(self):
        pass

    @GObject.Signal(name="video-export-requested")
    def video_export_requested_signal(self, file: Gio.File):
        pass

    @GObject.Signal(name="video-export-finished")
    def video_export_finished_signal(self):
        pass

    @GObject.Signal(name="video-export-progress")
    def video_export_progress_signal(self, status: float):
        pass

    def _setup_shortcuts(self):
        self._shortcuts_manager.register_group("files", "Files")
        self._shortcuts_manager.add("files", "export-file", "e", lambda *args: self.show_export_dialog(), "Export recovered video")

    def show_video_export_success(self, obj):
        self.status_page.set_title("Finished video restoration!")
        self.status_page.set_icon_name("check-round-outline2-symbolic")
        self.progress_bar_file_export.set_fraction(1.0)

    def on_video_export_progress(self, obj, progress):
        self.progress_bar_file_export.set_fraction(progress)

    def show_export_dialog(self):
        file_dialog = Gtk.FileDialog()
        video_file_filter = Gtk.FileFilter()
        video_file_filter.add_mime_type("video/*")
        file_dialog.set_default_filter(video_file_filter)
        file_dialog.set_title("Save restored video file")
        file_dialog.set_initial_folder(self._opened_file.get_parent())
        file_dialog.set_initial_name(f"{os.path.splitext(self._opened_file.get_basename())[0]}.restored.mp4")
        file_dialog.save(callback=lambda dialog, result: self.emit("video-export-requested", dialog.save_finish(result)))
        self.emit("video-export-dialog-opened")

    def export_video(self, output_file_path: str, video_codec, crf, frame_restorer_options: FrameRestorerOptions):
        def run_export():
            video_metadata = frame_restorer_options.video_metadata
            frame_restorer_provider = FRAME_RESTORER_PROVIDER
            frame_restorer_provider.init(frame_restorer_options)
            frame_restorer = frame_restorer_provider.get()

            progress_update_step_size = 100
            success = True
            video_tmp_file_output_path = os.path.join(tempfile.gettempdir(),f"{os.path.basename(os.path.splitext(video_metadata.video_file)[0])}.tmp{os.path.splitext(video_metadata.video_file)[1]}")
            try:
                frame_restorer.start(start_ns=0)

                with video_utils.VideoWriter(video_tmp_file_output_path, video_metadata.video_width,
                                             video_metadata.video_height, video_metadata.video_fps_exact,
                                             video_codec, time_base=video_metadata.time_base,
                                             crf=crf) as video_writer:
                    for frame_num, elem in enumerate(frame_restorer):
                        if elem is None:
                            success = False
                            logger.error("Error on export: frame restorer stopped prematurely")
                            break
                        (restored_frame, restored_frame_pts) = elem
                        video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)
                        if frame_num % progress_update_step_size == 0:
                            self.emit('video-export-progress', frame_num / video_metadata.frames_count)

            except Exception as e:
                success = False
                logger.error("Error on export", exc_info=e)
            finally:
                frame_restorer.stop()

            if success:
                audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, output_file_path)
                self.emit('video-export-progress', 1.0)
                self.emit('video-export-finished')
            else:
                if os.path.exists(video_tmp_file_output_path):
                    os.remove(video_tmp_file_output_path)

        exporter_thread = threading.Thread(target=run_export)
        exporter_thread.start()

    def start_export(self, file: Gio.File):
        if not CONFIG.loaded: CONFIG.load_config()
        frame_restorer_options = FrameRestorerOptions(CONFIG.mosaic_restoration_model, CONFIG.mosaic_detection_model, video_utils.get_video_meta_data(self._opened_file.get_path()), CONFIG.device, CONFIG.max_clip_duration, False, False)
        self.export_video(file.get_path(), CONFIG.export_codec, CONFIG.export_crf, frame_restorer_options)
