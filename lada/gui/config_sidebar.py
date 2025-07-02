import pathlib

from gi.repository import Gtk, GObject

from lada.gui.config import Config, ColorScheme
from lada.gui import utils
from lada.gui.utils import skip_if_uninitialized
from lada import get_available_restoration_models, get_available_detection_models

here = pathlib.Path(__file__).parent.resolve()


@Gtk.Template(filename=here / 'config_sidebar.ui')
class ConfigSidebar(Gtk.ScrolledWindow):
    __gtype_name__ = 'ConfigSidebar'

    toggle_button_mosaic_detection = Gtk.Template.Child()
    toggle_button_mosaic_removal = Gtk.Template.Child()
    combo_row_gpu = Gtk.Template.Child()
    combo_row_mosaic_removal_models = Gtk.Template.Child()
    combo_row_mosaic_detection_models = Gtk.Template.Child()
    spin_row_export_crf = Gtk.Template.Child()
    combo_row_export_codec = Gtk.Template.Child()
    spin_row_preview_buffer_duration = Gtk.Template.Child()
    spin_row_clip_max_duration = Gtk.Template.Child()
    switch_row_mute_audio = Gtk.Template.Child()
    list_box = Gtk.Template.Child()
    light_color_scheme_button = Gtk.Template.Child()
    dark_color_scheme_button = Gtk.Template.Child()
    system_color_scheme_button = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config: Config | None = None
        self.init_done = False

    def init_sidebar_from_config(self, config: Config):
        self.toggle_button_mosaic_detection.set_property("active", config.preview_mode == 'mosaic-detection')
        self.toggle_button_mosaic_removal.set_property("active", config.preview_mode == 'mosaic-removal')

        # init device
        combo_row_gpu_list = Gtk.StringList.new([])
        available_gpus = utils.get_available_gpus()
        configured_gpu_selection_idx = None
        for gpu_selection_idx, (device_id, device_name) in enumerate(available_gpus):
            combo_row_gpu_list.append(device_name)
            if config.device and utils.device_to_gpu_id(config.device) == device_id:
                configured_gpu_selection_idx = gpu_selection_idx
        self.combo_row_gpu.set_model(combo_row_gpu_list)
        if configured_gpu_selection_idx:
            self.combo_row_gpu.set_selected(configured_gpu_selection_idx)

        # init restoration model
        combo_row_models_list = Gtk.StringList.new([])
        available_models = get_available_restoration_models()
        for model_name in available_models:
            combo_row_models_list.append(model_name)
        self.combo_row_mosaic_removal_models.set_model(combo_row_models_list)
        idx = available_models.index(config.get_property("mosaic_restoration_model"))
        self.combo_row_mosaic_removal_models.set_selected(idx)

        # init detection model
        combo_row_detection_models_list = Gtk.StringList.new([])
        available_detection_models = get_available_detection_models()
        for model_name in available_detection_models:
            combo_row_detection_models_list.append(model_name)
        self.combo_row_mosaic_detection_models.set_model(combo_row_detection_models_list)
        idx = available_detection_models.index(config.mosaic_detection_model)
        self.combo_row_mosaic_detection_models.set_selected(idx)

        self.spin_row_export_crf.set_property('value', config.export_crf)

        # init codec
        codecs_string_list = self.combo_row_export_codec.get_model()
        for i in range(len(codecs_string_list)):
            if codecs_string_list.get_string(i) == config.export_codec:
                self.combo_row_export_codec.set_selected(i)

        self.spin_row_preview_buffer_duration.set_value(config.preview_buffer_duration)
        self.spin_row_clip_max_duration.set_value(config.max_clip_duration)
        self.switch_row_mute_audio.set_active(config.mute_audio)

        # init color scheme
        if config.color_scheme == ColorScheme.LIGHT: self.light_color_scheme_button.set_property("active", True)
        elif config.color_scheme == ColorScheme.DARK: self.dark_color_scheme_button.set_property("active", True)
        else: self.system_color_scheme_button.set_property("active", True)

        self.init_done = True

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value: Config):
        self._config = value
        self.init_sidebar_from_config(value)

    @GObject.Property()
    def disabled(self):
        return self.list_box.get_property("sensitive")

    @disabled.setter
    def disabled(self, value):
        self.list_box.set_property("sensitive", not value)

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def toggle_button_mosaic_detection_callback(self, button_clicked):
        enable_mosaic_detection = button_clicked.get_property("active")
        if enable_mosaic_detection:
            self.toggle_button_mosaic_removal.set_property("active", False)
            self._config.preview_mode = 'mosaic-detection'
        else:
            self.toggle_button_mosaic_removal.set_property("active", True)
            self._config.preview_mode = 'mosaic-removal'

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def toggle_button_mosaic_removal_callback(self, button_clicked):
        enable_mosaic_removal = button_clicked.get_property("active")
        if enable_mosaic_removal:
            self.toggle_button_mosaic_detection.set_property("active", False)
            self._config.preview_mode = 'mosaic-removal'
        else:
            self.toggle_button_mosaic_detection.set_property("active", True)
            self._config.preview_mode = 'mosaic-detection'

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def combo_row_mosaic_removal_models_selected_callback(self, combo_row, value):
        self._config.mosaic_restoration_model = combo_row.get_property("selected_item").get_string()

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def combo_row_mosaic_detection_models_selected_callback(self, combo_row, value):
        self._config.mosaic_detection_model = combo_row.get_property("selected_item").get_string()

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def combo_row_mosaic_export_codec_selected_callback(self, combo_row, value):
        self._config.export_codec = combo_row.get_property("selected_item").get_string()

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def spin_row_preview_export_crf_selected_callback(self, spin_row, value):
        self._config.export_crf = spin_row.get_property("value")

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def combo_row_gpu_selected_callback(self, combo_row, value):
        selected_gpu_name = combo_row.get_property("selected_item").get_string()
        for id, name in utils.get_available_gpus():
            if name == selected_gpu_name:
                self._config.device = f"cuda:{id}"
                break

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def spin_row_preview_buffer_duration_selected_callback(self, spin_row, value):
        self._config.preview_buffer_duration = spin_row.get_property("value")

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def spin_row_clip_max_duration_selected_callback(self, spin_row, value):
        self._config.max_clip_duration = spin_row.get_property("value")

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def switch_row_mute_audio_active_callback(self, switch_row, active):
        self._config.mute_audio = switch_row.get_property("active")

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def button_config_reset_callback(self, button_clicked):
        self.init_done = False
        self._config.reset_to_default_values()
        self.init_sidebar_from_config(self._config)

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def toggle_button_system_color_scheme_callback(self, button_clicked):
        self._config.color_scheme = ColorScheme.SYSTEM

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def toggle_button_light_color_scheme_callback(self, button_clicked):
        self._config.color_scheme = ColorScheme.LIGHT

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def toggle_button_dark_color_scheme_callback(self, button_clicked):
        self._config.color_scheme = ColorScheme.DARK