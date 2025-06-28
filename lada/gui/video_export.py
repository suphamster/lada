import logging
import os
import pathlib
import tempfile
import threading

from gi.repository import Gtk, GObject

from lada.lib import audio_utils, video_utils
from lada import LOG_LEVEL
from lada.gui.frame_restorer_provider import FrameRestorerOptions, FRAME_RESTORER_PROVIDER

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(filename=here / 'video_export.ui')
class VideoExport(Gtk.Widget):
    __gtype_name__ = 'VideoExport'

    progress_bar_file_export = Gtk.Template.Child()
    status_page_export_video = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect("video-export-finished", self.show_video_export_success)
        self.connect("video-export-progress", self.on_video_export_progress)

    @GObject.Signal(name="video-export-finished")
    def video_export_finished_signal(self):
        pass

    @GObject.Signal(name="video-export-progress")
    def video_export_progress_signal(self, status: float):
        pass

    def show_video_export_success(self, obj):
        self.status_page_export_video.set_title("Finished video restoration!")
        self.status_page_export_video.set_icon_name("check-round-outline2-symbolic")
        self.progress_bar_file_export.set_fraction(1.0)

    def on_video_export_progress(self, obj, progress):
        self.progress_bar_file_export.set_fraction(progress)

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