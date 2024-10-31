import glob
import os.path
import pathlib

import torch
from gi.repository import Adw, Gtk, Gio, Gdk
from lada import MODEL_WEIGHTS_DIR
import lada.gui.video_preview

here = pathlib.Path(__file__).parent.resolve()

VIDEO_MIME_TYPES = [
    'video/mp4',
    'video/x-matroska',
]

@Gtk.Template(filename=here / 'window.ui')
class MainWindow(Adw.ApplicationWindow):
    __gtype_name__ = 'MainWindow'

    button_open_file = Gtk.Template.Child()
    button_export_video = Gtk.Template.Child()
    toggle_button_preview_video = Gtk.Template.Child()
    widget_video_preview = Gtk.Template.Child()
    spinner_video_preview = Gtk.Template.Child()
    stack = Gtk.Template.Child()
    stack_video_preview = Gtk.Template.Child()
    toggle_button_mosaic_detection = Gtk.Template.Child()
    toggle_button_mosaic_removal = Gtk.Template.Child()
    combo_row_gpu = Gtk.Template.Child()
    combo_row_mosaic_removal_models = Gtk.Template.Child()
    spin_row_export_crf = Gtk.Template.Child()
    combo_row_export_codec = Gtk.Template.Child()
    progress_bar_file_export = Gtk.Template.Child()
    status_page_export_video = Gtk.Template.Child()
    banner_no_gpu = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # todo: Added enforcing dark theme as a quick workaround as some buttons are not properly visible on a light color scheme.
        style_manager = self.get_property('application').get_property("style-manager")
        style_manager.set_color_scheme(Adw.ColorScheme.FORCE_DARK)

        # init drag-drop files
        drop_target = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        def on_connect_drop(drop_target, file: Gio.File, x, y):
            self.open_file(file)
        drop_target.connect("drop", on_connect_drop)
        self.stack.add_controller(drop_target)

        # init device
        combo_row_gpu_list = self.combo_row_gpu.get_model()
        available_gpus = self.get_available_gpus()
        for device_id, device_name in available_gpus:
            combo_row_gpu_list.append(device_name)

        no_gpu_available = len(available_gpus) == 0
        if no_gpu_available:
            self.banner_no_gpu.set_revealed(True)
            device = "cpu"
        else:
            device = f"cuda:{available_gpus[0][0]}"
            self.combo_row_gpu.set_selected(0)
        self.widget_video_preview.set_property('device', device)

        # init models
        combo_row_models_list = self.combo_row_mosaic_removal_models.get_model()
        available_models = self.get_available_models()
        for model_name in available_models:
            combo_row_models_list.append(model_name)
        idx = available_models.index("basicvsrpp-generic")
        self.combo_row_mosaic_removal_models.set_selected(idx)

    @Gtk.Template.Callback()
    def button_open_file_callback(self, button_clicked):
        self.show_open_dialog()

    @Gtk.Template.Callback()
    def button_export_video_callback(self, button_clicked):
        self.show_export_dialog()

    @Gtk.Template.Callback()
    def switch_row_mosaic_cleaning_active_callback(self, switch_row, active):
        self.widget_video_preview.set_property('mosaic_cleaning', switch_row.get_property("active"))

    @Gtk.Template.Callback()
    def toggle_button_preview_video_callback(self, button_clicked):
        preview_active = button_clicked.get_property("active")
        self.widget_video_preview.set_property('passthrough', not preview_active)

    @Gtk.Template.Callback()
    def toggle_button_mosaic_detection_callback(self, button_clicked):
        enable_mosaic_detection = button_clicked.get_property("active")
        if enable_mosaic_detection:
            self.toggle_button_mosaic_removal.set_property("active", False)
            self.widget_video_preview.set_property('mosaic-detection', True)
        else:
            self.toggle_button_mosaic_removal.set_property("active", True)
            self.widget_video_preview.set_property('mosaic-detection', False)

    @Gtk.Template.Callback()
    def toggle_button_mosaic_removal_callback(self, button_clicked):
        enable_mosaic_removal = button_clicked.get_property("active")
        if enable_mosaic_removal:
            self.toggle_button_mosaic_detection.set_property("active", False)
            self.widget_video_preview.set_property('mosaic-detection', False)
        else:
            self.toggle_button_mosaic_detection.set_property("active", True)
            self.widget_video_preview.set_property('mosaic-detection', True)

    @Gtk.Template.Callback()
    def combo_row_mosaic_removal_models_selected_callback(self, combo_row, value):
        self.widget_video_preview.set_property('mosaic-restoration-model', combo_row.get_property("selected_item").get_string())

    @Gtk.Template.Callback()
    def combo_row_gpu_selected_callback(self, combo_row, value):
        selected_gpu_name = combo_row.get_property("selected_item").get_string()
        for id, name in self.get_available_gpus():
            if name == selected_gpu_name:
                selected_gpu_id = f"cuda:{id}"
                self.widget_video_preview.set_property('device', selected_gpu_id)
                break

    @Gtk.Template.Callback()
    def spin_row_preview_buffer_duration_selected_callback(self, spin_row, value):
        self.widget_video_preview.set_property('buffer-queue-min-thresh-time', spin_row.get_property("value"))

    @Gtk.Template.Callback()
    def spin_row_clip_max_duration_selected_callback(self, spin_row, value):
        self.widget_video_preview.set_property('max-clip-length', spin_row.get_property("value"))

    def get_available_gpus(self):
        return [(i, torch.cuda.get_device_properties(i).name) for i in range(torch.cuda.device_count())]

    def get_available_models(self):
        available_models = []
        for file_path in glob.glob(os.path.join(MODEL_WEIGHTS_DIR, '**/*.pth'), recursive=True):
            file_name = os.path.basename(file_path)
            if file_name == 'lada_mosaic_restoration_model_generic.pth':
                available_models.append("basicvsrpp-generic")
            elif file_name == 'lada_mosaic_restoration_model_bj_pov.pth':
                available_models.append("basicvsrpp-bj-pov")
            elif file_name == 'clean_youknow_video.pth':
                available_models.append("deepmosaics")
        return available_models

    def show_open_dialog(self):
        file_dialog = Gtk.FileDialog()
        video_file_filter = Gtk.FileFilter()
        video_file_filter.add_mime_type("video/*")
        file_dialog.set_default_filter(video_file_filter)
        file_dialog.set_title("Select a video file")
        file_dialog.open(callback=lambda dialog, result: self.open_file(dialog.open_finish(result)))

    def show_export_dialog(self):
        file_dialog = Gtk.FileDialog()
        video_file_filter = Gtk.FileFilter()
        video_file_filter.add_mime_type("video/*")
        file_dialog.set_default_filter(video_file_filter)
        file_dialog.set_title("Save restored video file")
        file_dialog.save(callback=lambda dialog, result: self.start_export(dialog.save_finish(result)))

    def open_file(self, file: Gio.File):
        self.set_title(os.path.basename(file.get_path()))
        self.switch_to_main_view()

        if self.stack_video_preview.get_visible_child() == self.widget_video_preview:
            self.stack_video_preview.set_visible_child(self.spinner_video_preview)

        def show_video_preview(obj):
            self.stack_video_preview.set_visible_child(self.widget_video_preview)

        self.widget_video_preview.connect("video-preview-init-done", show_video_preview)
        self.widget_video_preview.open_video_file(file)

    def start_export(self, file: Gio.File):
        self.stack.set_visible_child_name("file-export")

        def show_video_export_success(obj):
            print("finished exporting")
            self.status_page_export_video.set_title("Finished video restoration!")
            self.status_page_export_video.set_icon_name("check-round-outline2-symbolic")

        def on_video_export_progress(obj, progress):
            self.progress_bar_file_export.set_fraction(progress)

        self.widget_video_preview.connect("video-export-finished", show_video_export_success)
        self.widget_video_preview.connect("video-export-progress", on_video_export_progress)
        self.widget_video_preview.export_video(file.get_path(), self.combo_row_export_codec.get_property("selected_item").get_string(), self.spin_row_export_crf.get_property("value"))

    def switch_to_main_view(self):
        self.stack.set_visible_child_name("page_main")
        self.button_export_video.set_sensitive(True)