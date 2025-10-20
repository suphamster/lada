import logging
import os
import pathlib
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from fractions import Fraction

from gi.repository import Gtk, GObject, Gio, Adw, GLib

from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.config.config import Config
from lada.gui.config.no_gpu_banner import NoGpuBanner
from lada.gui.export import export_utils
from lada.gui.export.export_item_data import ExportItemData, ExportItemDataProgress, ExportItemState
from lada.gui.export.export_multiple_files_page import ExportMultipleFilesPage
from lada.gui.export.export_single_file_page import ExportSingleFileStatusPage
from lada.gui.frame_restorer_provider import FrameRestorerOptions, FRAME_RESTORER_PROVIDER
from lada.lib import audio_utils, video_utils

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@dataclass
class ResumeInformation:
    frame_pts: int
    time_base: Fraction
    frame_num : int

    def get_resume_timestamp_ns(self):
        SECOND = 1_000_000_000
        return int((self.frame_pts * self.time_base) * SECOND)

@Gtk.Template(string=utils.translate_ui_xml(here / 'export_view.ui'))
class ExportView(Gtk.Widget):
    __gtype_name__ = 'ExportView'

    status_page: ExportSingleFileStatusPage = Gtk.Template.Child()
    multiple_files_page: ExportMultipleFilesPage = Gtk.Template.Child()
    button_start_export: Gtk.Button = Gtk.Template.Child()
    button_cancel_export: Gtk.Button = Gtk.Template.Child()
    button_resume_export: Gtk.Button = Gtk.Template.Child()
    button_pause_export: Gtk.Button = Gtk.Template.Child()
    stack: Gtk.Stack = Gtk.Template.Child()
    view_switcher: Adw.ViewSwitcher = Gtk.Template.Child()
    config_sidebar = Gtk.Template.Child()
    button_add_files: Gtk.Button = Gtk.Template.Child()
    banner_no_gpu: NoGpuBanner = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._view_stack: Adw.ViewStack | None = None
        self._config: Config | None = None
        self.in_progress_idx: int | None = None
        self.single_file = True
        self.stop_requested = False
        self.pause_requested = False
        self.resume_info: ResumeInformation | None = None
        self.video_writer: video_utils.VideoWriter | None = None
        self.progress_calculator: export_utils.ProgressCalculator | None = None

        self.connect("video-export-finished", self.on_video_export_finished)
        self.connect("video-export-failed", self.on_video_export_failed)
        self.connect("video-export-progress", self.on_video_export_progress)
        self.connect("video-export-resumed", self.on_video_export_resumed)
        self.connect("video-export-paused", self.on_video_export_paused)
        self.connect("video-export-stopped", self.on_video_export_stopped)

        self.model =  Gio.ListStore(item_type=ExportItemData)
        self.multiple_files_page.bind(self.model)

        def on_files_added(obj, files):
            self.button_add_files.set_sensitive(True)
            self.add_files(files)
        self.connect("files-added", on_files_added)

        self.status_page.connect("start-export-requested", self.on_button_start_export_clicked)
        self.status_page.connect("stop-export-requested", self.on_button_cancel_export_clicked)
        self.status_page.connect("pause-export-requested", self.on_button_pause_export_clicked)
        self.status_page.connect("resume-export-requested", self.on_button_resume_export_clicked)

        self.multiple_files_page.connect("show-error-requested", self.on_show_error_requested)
        self.multiple_files_page.connect("remove-item-requested", self.on_remove_item_requested)

        drop_target = utils.create_video_files_drop_target(lambda files: self.emit("files-added", files))
        self.add_controller(drop_target)

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
        self._config.connect("notify::export-directory", self.on_config_changed)
        self._config.connect("notify::file-name-pattern", self.on_config_changed)
        self.set_restore_button_label()

    @GObject.Property(type=Adw.ViewStack)
    def view_stack(self):
        return self._view_stack

    @view_stack.setter
    def view_stack(self, value: Adw.ViewStack):
        self._view_stack = value

    def add_files(self, added_files: list[Gio.File]):
        assert len(added_files) > 0

        for original_file in added_files:
            if any([original_file.get_path() == item.original_file.get_path() for item in self.model]):
                # duplicate
                continue
            if self._config.export_directory:
                restored_file = self.get_restored_file_path(original_file, self._config.export_directory)
            else:
                # We don't know the output directory yet. This guess needs to be updated after the user set one via FilePicker
                restored_file = self.get_restored_file_path(original_file, added_files[0].get_parent().get_path())
            export_item = ExportItemData(original_file, restored_file)
            self.model.append(export_item)

        self.single_file = len(self.model) == 1

        if self.single_file:
            self.stack.set_visible_child_name("single-file")
            self.status_page.on_add_file(self.model[0])
        else:
            self.stack.set_visible_child_name("multiple-files")
            self.update_export_buttons()

    def update_export_buttons(self):
        if self.single_file:
            return
        count_queued_items = sum([item.state == ExportItemState.QUEUED for item in self.model])
        is_in_progress = self.in_progress_idx is not None
        is_paused = self.resume_info is not None
        is_any_queued_items = count_queued_items > 0
        self.button_start_export.set_visible(not is_in_progress and is_any_queued_items)
        self.button_pause_export.set_visible(is_in_progress and not is_paused)
        self.button_resume_export.set_visible(is_paused)
        self.button_cancel_export.set_visible(is_in_progress)

    @GObject.Signal(name="video-export-finished")
    def video_export_finished_signal(self):
        pass

    @GObject.Signal(name="video-export-failed", arg_types=(GObject.TYPE_STRING,))
    def video_export_failed_signal(self, error_message: str):
        pass

    @GObject.Signal(name="video-export-paused",)
    def video_export_paused_signal(self):
        pass

    @GObject.Signal(name="video-export-resumed",)
    def video_export_resumed_signal(self):
        pass

    @GObject.Signal(name="video-export-stopped",)
    def video_export_stopped_signal(self):
        pass

    @GObject.Signal(name="video-export-progress", arg_types=(ExportItemDataProgress,))
    def video_export_progress_signal(self, progress):
        pass

    @GObject.Signal(name="video-export-requested")
    def video_export_requested_signal(self, save_file: Gio.File):
        pass

    @GObject.Signal(name="files-added", arg_types=(GObject.TYPE_PYOBJECT,))
    def files_opened_signal(self, files: list[Gio.File]):
        pass

    @Gtk.Template.Callback()
    def on_button_start_export_clicked(self, *args):
        if self._config.export_directory:
            item = self.model[self.get_next_queued_item_idx()]
            self.emit("video-export-requested", item.restored_file)
        else:
            self.show_export_dialog()

    @Gtk.Template.Callback()
    def button_add_files_callback(self, button_clicked):
        self.button_add_files.set_sensitive(False)
        callback = lambda files: self.emit("files-added", files)
        dismissed_callback = lambda *args: self.button_add_files.set_sensitive(True)
        utils.show_open_files_dialog(callback, dismissed_callback)

    @Gtk.Template.Callback()
    def on_button_cancel_export_clicked(self, button_clicked):
        self.stop_requested = True
        self.button_pause_export.set_sensitive(False)
        self.button_cancel_export.set_sensitive(False)

    @Gtk.Template.Callback()
    def on_button_pause_export_clicked(self, button_clicked):
        assert self.resume_info is None

        self.button_pause_export.set_sensitive(False)
        self.button_cancel_export.set_sensitive(False)
        self.pause_requested = True

    @Gtk.Template.Callback()
    def on_button_resume_export_clicked(self, button_clicked):
        assert self.resume_info is not None
        self.button_resume_export.set_sensitive(False)
        self.button_cancel_export.set_sensitive(False)

        self.pause_requested = False
        assert self.in_progress_idx is not None
        item = self.model[self.in_progress_idx]
        self._start_export(item.original_file, item.restored_file)

    def on_show_error_requested(self, obj, idx):
        model_item = self.model[idx]
        export_utils.open_error_dialog(self, model_item.original_file.get_basename(), model_item.error_details)

    def on_remove_item_requested(self, obj, idx):
        self.model.remove(idx)
        self.update_export_buttons()

    def on_config_changed(self, *args):
        if self._config.export_directory:
            for model_item in self.model:
                restored_file = self.get_restored_file_path(model_item.original_file, self._config.export_directory)
                model_item.restored_file = restored_file
        self.set_restore_button_label()

    def set_restore_button_label(self):
        label = _("Restore") if self._config.export_directory else _("Restore…")
        self.status_page.set_button_start_restore_label(label)
        self.button_start_export.set_label(label)

    def get_next_queued_item_idx(self) -> int | None:
        for idx, item in enumerate(self.model):
            if item.state == ExportItemState.QUEUED:
                return idx
        return None

    def continue_next_file(self):
        next_idx = self.get_next_queued_item_idx()
        if next_idx is None:
            # done, all queued items processed
            self.view_switcher.set_sensitive(True)
            self.config_sidebar.set_property("disabled", False)
            self.in_progress_idx = None
            self.update_export_buttons()
        else:
            # continue, queued items remaining
            self._start_export(self.model[next_idx].original_file, self.model[next_idx].restored_file)

    def show_video_export_started(self, save_file: Gio.File):
        self.view_switcher.set_sensitive(False)
        self.config_sidebar.set_property("disabled", True)

        idx = self.get_next_queued_item_idx()
        if idx is None:
            return

        self.in_progress_idx = idx
        self.update_export_buttons()

        model_item = self.model[idx]
        model_item.state = ExportItemState.PROCESSING

        if self.single_file:
            self.status_page.show_video_export_started(save_file)
        self.multiple_files_page.show_video_export_started(idx)

    def on_video_export_finished(self, obj):
        assert self.in_progress_idx is not None

        model_item = self.model[self.in_progress_idx]
        model_item.progress.complete()
        model_item.state = ExportItemState.FINISHED

        if self.single_file:
            self.status_page.on_video_export_finished()
        self.multiple_files_page.on_video_export_finished(self.in_progress_idx)

        self.continue_next_file()

    def on_video_export_progress(self, obj, progress: ExportItemDataProgress):
        if self.in_progress_idx is None:
            return

        model_item = self.model[self.in_progress_idx]
        model_item.progress = progress

        if self.single_file:
            self.status_page.on_video_export_progress(progress)
        self.multiple_files_page.on_video_export_progress(self.in_progress_idx, progress)

    def on_video_export_stopped(self, obj):
        assert self.in_progress_idx is not None

        model_item = self.model[self.in_progress_idx]
        model_item.state = ExportItemState.QUEUED
        model_item.progress = ExportItemDataProgress()

        if self.single_file:
            self.status_page.on_video_export_stopped()
        self.multiple_files_page.on_video_export_stopped(self.in_progress_idx)

        self.in_progress_idx = None
        self.stop_requested = False
        self.update_export_buttons()
        self.view_switcher.set_sensitive(True)
        self.config_sidebar.set_property("disabled", False)
        self.button_start_export.set_sensitive(True)
        self.button_cancel_export.set_sensitive(True)
        self.button_pause_export.set_sensitive(True)

    def on_video_export_paused(self, obj):
        assert self.in_progress_idx is not None

        model_item = self.model[self.in_progress_idx]
        model_item.state = ExportItemState.PAUSED

        if self.single_file:
            self.status_page.on_video_export_paused()
        self.multiple_files_page.on_video_export_paused(self.in_progress_idx)

        self.update_export_buttons()
        self.button_pause_export.set_sensitive(True)
        self.button_cancel_export.set_sensitive(True)
        self.pause_requested = False

    def on_video_export_resumed(self, obj):
        assert self.in_progress_idx is not None

        model_item = self.model[self.in_progress_idx]
        assert model_item.state == ExportItemState.PAUSED
        model_item.state = ExportItemState.PROCESSING

        if self.single_file:
            self.status_page.on_video_export_resumed()
        self.multiple_files_page.on_video_export_resumed(self.in_progress_idx)

        self.update_export_buttons()
        self.button_resume_export.set_sensitive(True)
        self.button_cancel_export.set_sensitive(True)

    def on_video_export_failed(self, obj, error_message):
        assert self.in_progress_idx is not None

        model_item = self.model[self.in_progress_idx]
        model_item.state = ExportItemState.FAILED
        model_item.error_details = error_message

        if self.single_file:
            self.status_page.on_video_export_failed()
        self.multiple_files_page.on_video_export_failed(self.in_progress_idx)

        export_utils.open_error_dialog(self, model_item.original_file.get_basename(), error_message)

        self.continue_next_file()

    def start_export(self, restore_directory_or_file: Gio.File):
        # Update initial guessed output restore directory/file now that the user has provided it via file/dir picker dialog
        if not self._config.export_directory:
            restored_files: list[Gio.File] = []
            if self.single_file:
                assert len(self.model) == 1
                restored_file = restore_directory_or_file
                model_item = self.model[0]
                model_item.restored_file = restored_file
                restored_files.append(restored_file)
            else:
                assert os.path.isdir(restore_directory_or_file.get_path())
                restore_directory = restore_directory_or_file
                for idx, model_item in enumerate(self.model):
                    restored_file = self.get_restored_file_path(model_item.original_file, restore_directory.get_path())
                    model_item.restored_file = restored_file
                    restored_files.append(restored_file)
            self.multiple_files_page.on_video_export_started(restored_files)

        item = self.model[self.get_next_queued_item_idx()]
        self._start_export(item.original_file, item.restored_file)

    def _start_export(self, source_file: Gio.File, restore_file: Gio.File):
        assert os.path.isfile(source_file.get_path())
        if not self.resume_info:
            self.show_video_export_started(restore_file)

        def run_export():
            frame_restorer_options = FrameRestorerOptions(self._config.mosaic_restoration_model, self._config.mosaic_detection_model, video_utils.get_video_meta_data(source_file.get_path()), self._config.device, self._config.max_clip_duration, False, False)
            video_metadata = frame_restorer_options.video_metadata
            frame_restorer_provider = FRAME_RESTORER_PROVIDER
            frame_restorer_provider.init(frame_restorer_options)
            frame_restorer = frame_restorer_provider.get()
            restore_file_path = restore_file.get_path()

            progress_update_step_size = 100
            success = True
            video_tmp_file_output_path = os.path.join(tempfile.gettempdir(),f"{os.path.basename(os.path.splitext(restore_file_path)[0])}.tmp{os.path.splitext(restore_file_path)[1]}")
            try:
                if self.resume_info:
                    start_ns = self.resume_info.get_resume_timestamp_ns()
                    start_frame_num = self.resume_info.frame_num
                    logger.info(f"Resume requested: Starting FrameRestorer at timestamp {start_ns}ns")
                else:
                    start_ns = 0
                    start_frame_num = 0
                    self.video_writer = video_utils.VideoWriter(
                        video_tmp_file_output_path, video_metadata.video_width,
                        video_metadata.video_height, video_metadata.video_fps_exact,
                        self._config.export_codec, time_base=video_metadata.time_base,
                        crf=self._config.export_crf, custom_encoder_options=self._config.custom_ffmpeg_encoder_options)
                    self.progress_calculator = export_utils.ProgressCalculator(video_metadata)

                frame_restorer.start(start_ns=start_ns)

                duration_start = time.time()
                for frame_num, elem in enumerate(frame_restorer, start=start_frame_num):
                    if self.stop_requested:
                        success = False
                        logger.warning("Stop requested: Stopping FrameRestorer")
                        break
                    if elem is None:
                        success = False
                        logger.error("Error on export: frame restorer stopped prematurely")
                        break

                    (restored_frame, restored_frame_pts) = elem
                    if self.resume_info:
                        if restored_frame_pts <= self.resume_info.frame_pts:
                            logging.debug("Received frame earlier than resume position, skipping frame...")
                            continue
                        else:
                            logger.debug("Received first frame after resume position, successful resume.")
                            self.resume_info = None
                            GLib.idle_add(lambda: self.emit('video-export-resumed'))
                    self.video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)

                    duration_end = time.time()
                    duration = duration_end - duration_start
                    duration_start = duration_end
                    self.progress_calculator.update(duration)
                    if frame_num % progress_update_step_size == 0:
                        GLib.idle_add(lambda: self.emit('video-export-progress', self.progress_calculator.get_progress()))

                    if self.pause_requested:
                        logger.info("Pause requested: Pausing FrameRestorer")
                        self.resume_info = ResumeInformation(restored_frame_pts, video_metadata.time_base, frame_num)
                        break

            except Exception as e:
                success = False
                err_msg = "".join(traceback.format_exception_only(e))
                GLib.idle_add(lambda: self.emit('video-export-failed', err_msg))
            finally:
                if not self.pause_requested:
                    self.video_writer.release()
                frame_restorer.stop()

            if self.pause_requested:
                GLib.idle_add(lambda: self.emit('video-export-paused'))
            else:
                if success:
                    audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, restore_file_path)
                    progress = self.progress_calculator.get_progress()
                    progress.complete()
                    GLib.idle_add(lambda: self.emit('video-export-progress', progress))
                    GLib.idle_add(lambda: self.emit('video-export-finished'))
                else:
                    if os.path.exists(video_tmp_file_output_path):
                        os.remove(video_tmp_file_output_path)
            if self.stop_requested:
                GLib.idle_add(lambda: self.emit('video-export-stopped'))

        exporter_thread = threading.Thread(target=run_export)
        exporter_thread.start()

    def show_export_dialog(self):
        def on_dialog_result(dialog, result):
            try:
                if self.single_file:
                    selected = dialog.save_finish(result)
                else:
                    selected = dialog.select_folder_finish(result)
                if selected is not None:
                    self.emit("video-export-requested",selected)
            except GLib.Error as error:
                if error.message == "Dismissed by user":
                    logger.debug("FileDialog cancelled: Dismissed by user")
                else:
                    logger.error(f"Error opening file: {error.message}")
                    raise error

        if self.single_file:
            file_dialog = Gtk.FileDialog()
            video_file_filter = Gtk.FileFilter()
            video_file_filter.add_mime_type("video/*")
            file_dialog.set_default_filter(video_file_filter)
            file_dialog.set_title(_("Save restored video file"))
            initial_restored_file = self.model[0].restored_file
            file_dialog.set_initial_folder(initial_restored_file.get_parent())
            file_dialog.set_initial_name(initial_restored_file.get_basename())
            file_dialog.save(callback=on_dialog_result)
        else:
            file_dialog = Gtk.FileDialog()
            file_dialog.set_title(_("Save restored video files"))
            first_original_file = self.model[0].original_file
            file_dialog.set_initial_folder(first_original_file.get_parent())
            file_dialog.select_folder(callback=on_dialog_result)

    def get_restored_file_path(self, original_file: Gio.File, output_dir: str) -> Gio.File:
        orig_file_name = os.path.splitext(original_file.get_basename())[0]
        restored_file_name = self._config.file_name_pattern.replace("{orig_file_name}", orig_file_name)
        return Gio.File.new_build_filenamev([output_dir, restored_file_name])

    def close(self):
        self.stop_requested = True
