import pathlib

from gi.repository import Gtk, GObject

from lada.gui.config import Config
from lada.gui import utils

here = pathlib.Path(__file__).parent.resolve()


@Gtk.Template(filename=here / 'config_sidebar.ui')
class ConfigSidebar(Gtk.ListBox):
    __gtype_name__ = 'ConfigSidebar'

    toggle_button_mosaic_detection = Gtk.Template.Child()
    toggle_button_mosaic_removal = Gtk.Template.Child()
    combo_row_gpu = Gtk.Template.Child()
    combo_row_mosaic_removal_models = Gtk.Template.Child()
    spin_row_export_crf = Gtk.Template.Child()
    combo_row_export_codec = Gtk.Template.Child()
    spin_row_preview_buffer_duration = Gtk.Template.Child()
    spin_row_clip_max_duration = Gtk.Template.Child()
    switch_row_mosaic_cleaning = Gtk.Template.Child()
    switch_row_mute_audio = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_done = False
        self.config = Config()
        self.config.load_config()
        self.toggle_button_mosaic_detection.set_property("active", self.config.preview_mode == 'mosaic-detection')
        self.toggle_button_mosaic_removal.set_property("active", self.config.preview_mode == 'mosaic-removal')

        # init device
        combo_row_gpu_list = self.combo_row_gpu.get_model()
        available_gpus = utils.get_available_gpus()
        # We're using GPU name in a ComboBox but libadwaita sets up the label with max-width-chars: 20 and there does not
        # seem to be a way to overwrite this. So let's try to make sure GPU names are below 20 characters to be readable
        for gpu_selection_idx, (device_id, device_name) in enumerate(available_gpus):
            if device_name.startswith("NVIDIA GeForce RTX"):
                device_name = device_name.replace("NVIDIA GeForce RTX", "RTX")
                available_gpus[gpu_selection_idx] = (device_id, device_name)
        configured_gpu_selection_idx = None
        for gpu_selection_idx, (device_id, device_name) in enumerate(available_gpus):
            combo_row_gpu_list.append(device_name)
            if self.config.device and utils.is_device_available(self.config.device) and utils.device_to_gpu_id(
                    self.config.device) == device_id:
                configured_gpu_selection_idx = gpu_selection_idx

        no_gpu_available = len(available_gpus) == 0
        if no_gpu_available:
            self.config.device = "cpu"
        elif configured_gpu_selection_idx:
            self.combo_row_gpu.set_selected(configured_gpu_selection_idx)
        else:
            self.config.device = f"cuda:{available_gpus[0][0]}"
            self.combo_row_gpu.set_selected(0)

        # init models
        combo_row_models_list = self.combo_row_mosaic_removal_models.get_model()
        available_models = utils.get_available_models()
        for model_name in available_models:
            combo_row_models_list.append(model_name)
        if self.config.mosaic_restoration_model in available_models:
            idx = available_models.index(self.config.mosaic_restoration_model)
        else:
            default_model = self.config.get_default_restoration_model()
            print(
                f"configured model {self.config.mosaic_restoration_model} is not available on the filesystem, falling back to model {default_model}")
            idx = available_models.index(default_model)
            self.config.mosaic_restoration_model = default_model
        self.combo_row_mosaic_removal_models.set_selected(idx)

        self.spin_row_export_crf.set_property('value', self.config.export_crf)

        codecs_string_list = self.combo_row_export_codec.get_model()
        for i in range(len(codecs_string_list)):
            if codecs_string_list.get_string(i) == self.config.export_codec:
                self.combo_row_export_codec.set_selected(i)

        self.spin_row_preview_buffer_duration.set_value(self.config.preview_buffer_duration)
        self.spin_row_clip_max_duration.set_value(self.config.max_clip_duration)

        self.switch_row_mosaic_cleaning.set_active(self.config.mosaic_pre_cleaning)

        self.switch_row_mute_audio.set_active(self.config.mute_audio)

        self.init_done = True

    @GObject.Property()
    def preview_mode(self):
        return self.config.preview_mode

    @preview_mode.setter
    def preview_mode(self, value):
        self.config.preview_mode = value
        self.config.save()

    @GObject.Property()
    def mosaic_restoration_model(self):
        return self.config.mosaic_restoration_model

    @mosaic_restoration_model.setter
    def mosaic_restoration_model(self, value):
        self.config.mosaic_restoration_model = value
        self.config.save()

    @GObject.Property()
    def device(self):
        return self.config.device

    @device.setter
    def device(self, value):
        self.config.device = value
        self.config.save()

    @GObject.Property()
    def preview_buffer_duration(self):
        return self.config.preview_buffer_duration

    @preview_buffer_duration.setter
    def preview_buffer_duration(self, value):
        self.config.preview_buffer_duration = value
        self.config.save()

    @GObject.Property()
    def max_clip_duration(self):
        return self.config.max_clip_duration

    @max_clip_duration.setter
    def max_clip_duration(self, value):
        self.config.max_clip_duration = value
        self.config.save()

    @GObject.Property()
    def mosaic_pre_cleaning(self):
        return self.config.mosaic_pre_cleaning

    @mosaic_pre_cleaning.setter
    def mosaic_pre_cleaning(self, value):
        self.config.mosaic_pre_cleaning = value
        self.config.save()

    @GObject.Property()
    def mute_audio(self):
        return self.config.mute_audio

    @mute_audio.setter
    def mute_audio(self, value):
        self.config.mute_audio = value
        self.config.save()

    @GObject.Property()
    def export_crf(self):
        return self.config.export_crf

    @export_crf.setter
    def export_crf(self, value):
        self.config.export_crf = value
        self.config.save()

    @GObject.Property()
    def export_codec(self):
        return self.config.export_codec

    @export_codec.setter
    def export_codec(self, value):
        self.config.export_codec = value
        self.config.save()

    @Gtk.Template.Callback()
    def toggle_button_mosaic_detection_callback(self, button_clicked):
        if not self.init_done:
            return
        enable_mosaic_detection = button_clicked.get_property("active")
        if enable_mosaic_detection:
            self.toggle_button_mosaic_removal.set_property("active", False)
            self.preview_mode = 'mosaic-detection'
        else:
            self.toggle_button_mosaic_removal.set_property("active", True)
            self.preview_mode = 'mosaic-removal'

    @Gtk.Template.Callback()
    def toggle_button_mosaic_removal_callback(self, button_clicked):
        if not self.init_done:
            return
        enable_mosaic_removal = button_clicked.get_property("active")
        if enable_mosaic_removal:
            self.toggle_button_mosaic_detection.set_property("active", False)
            self.preview_mode = 'mosaic-removal'
        else:
            self.toggle_button_mosaic_detection.set_property("active", True)
            self.preview_mode = 'mosaic-detection'

    @Gtk.Template.Callback()
    def combo_row_mosaic_removal_models_selected_callback(self, combo_row, value):
        if not self.init_done:
            return
        self.mosaic_restoration_model = combo_row.get_property("selected_item").get_string()

    @Gtk.Template.Callback()
    def combo_row_mosaic_export_codec_selected_callback(self, combo_row, value):
        if not self.init_done:
            return
        self.export_codec = combo_row.get_property("selected_item").get_string()

    @Gtk.Template.Callback()
    def spin_row_preview_export_crf_selected_callback(self, spin_row, value):
        if not self.init_done:
            return
        self.export_crf = spin_row.get_property("value")

    @Gtk.Template.Callback()
    def combo_row_gpu_selected_callback(self, combo_row, value):
        if not self.init_done:
            return
        selected_gpu_name = combo_row.get_property("selected_item").get_string()
        for id, name in utils.get_available_gpus():
            if name == selected_gpu_name:
                self.device = f"cuda:{id}"
                break

    @Gtk.Template.Callback()
    def spin_row_preview_buffer_duration_selected_callback(self, spin_row, value):
        if not self.init_done:
            return
        self.preview_buffer_duration = spin_row.get_property("value")

    @Gtk.Template.Callback()
    def spin_row_clip_max_duration_selected_callback(self, spin_row, value):
        if not self.init_done:
            return
        self.max_clip_duration = spin_row.get_property("value")

    @Gtk.Template.Callback()
    def switch_row_mosaic_cleaning_active_callback(self, switch_row, active):
        if not self.init_done:
            return
        self.mosaic_pre_cleaning = switch_row.get_property("active")

    @Gtk.Template.Callback()
    def switch_row_mute_audio_active_callback(self, switch_row, active):
        if not self.init_done:
            return
        self.mute_audio = switch_row.get_property("active")
