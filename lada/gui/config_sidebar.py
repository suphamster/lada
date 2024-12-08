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
        self.save_config = True
        self.config = Config()
        self.config.load_config()

        self.init_sidebar_from_config(self.config)

        self.init_done = True

    def init_sidebar_from_config(self, config: Config):
        self.toggle_button_mosaic_detection.set_property("active", config.preview_mode == 'mosaic-detection')
        self.toggle_button_mosaic_removal.set_property("active", config.preview_mode == 'mosaic-removal')

        # init device
        combo_row_gpu_list = Gtk.StringList.new([])
        available_gpus = utils.get_available_gpus()
        configured_gpu_selection_idx = None
        for gpu_selection_idx, (device_id, device_name) in enumerate(available_gpus):
            combo_row_gpu_list.append(device_name)
            if config.device and utils.is_device_available(config.device) and utils.device_to_gpu_id(
                    config.device) == device_id:
                configured_gpu_selection_idx = gpu_selection_idx
        self.combo_row_gpu.set_model(combo_row_gpu_list)
        no_gpu_available = len(available_gpus) == 0
        if no_gpu_available:
            self.device = "cpu"
        elif configured_gpu_selection_idx:
            self.combo_row_gpu.set_selected(configured_gpu_selection_idx)
        else:
            self.device = f"cuda:{available_gpus[0][0]}"
            self.combo_row_gpu.set_selected(0)

        # init model
        combo_row_models_list = Gtk.StringList.new([])
        available_models = utils.get_available_models()
        for model_name in available_models:
            combo_row_models_list.append(model_name)
        self.combo_row_mosaic_removal_models.set_model(combo_row_models_list)
        if config.mosaic_restoration_model in available_models:
            idx = available_models.index(config.mosaic_restoration_model)
        else:
            default_model = config.get_default_restoration_model()
            print(f"configured model {config.mosaic_restoration_model} is not available on the filesystem, falling back to model {default_model}")
            idx = available_models.index(default_model)
            self.mosaic_restoration_model = default_model
        self.combo_row_mosaic_removal_models.set_selected(idx)

        self.spin_row_export_crf.set_property('value', config.export_crf)

        # init codec
        codecs_string_list = self.combo_row_export_codec.get_model()
        for i in range(len(codecs_string_list)):
            if codecs_string_list.get_string(i) == config.export_codec:
                self.combo_row_export_codec.set_selected(i)

        self.spin_row_preview_buffer_duration.set_value(config.preview_buffer_duration)
        self.spin_row_clip_max_duration.set_value(config.max_clip_duration)
        self.switch_row_mosaic_cleaning.set_active(config.mosaic_pre_cleaning)
        self.switch_row_mute_audio.set_active(config.mute_audio)

    @GObject.Property(flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.EXPLICIT_NOTIFY)
    def preview_mode(self):
        return self.config.preview_mode

    @preview_mode.setter
    def preview_mode(self, value):
        if value == self.config.preview_mode:
            return
        self.config.preview_mode = value
        self.notify('preview-mode')
        if self.save_config:
            self.config.save()

    @GObject.Property(flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.EXPLICIT_NOTIFY)
    def mosaic_restoration_model(self):
        return self.config.mosaic_restoration_model

    @mosaic_restoration_model.setter
    def mosaic_restoration_model(self, value):
        if value == self.config.mosaic_restoration_model:
            return
        self.config.mosaic_restoration_model = value
        self.notify('mosaic-restoration-model')
        if self.save_config:
            self.config.save()

    @GObject.Property(flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.EXPLICIT_NOTIFY)
    def device(self):
        return self.config.device

    @device.setter
    def device(self, value):
        if value == self.config.device:
            return
        self.config.device = value
        self.notify('device')
        if self.save_config:
            self.config.save()

    @GObject.Property(flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.EXPLICIT_NOTIFY)
    def preview_buffer_duration(self):
        return self.config.preview_buffer_duration

    @preview_buffer_duration.setter
    def preview_buffer_duration(self, value):
        if value == self.config.preview_buffer_duration:
            return
        self.config.preview_buffer_duration = value
        self.notify('preview-buffer-duration')
        if self.save_config:
            self.config.save()

    @GObject.Property(flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.EXPLICIT_NOTIFY)
    def max_clip_duration(self):
        return self.config.max_clip_duration

    @max_clip_duration.setter
    def max_clip_duration(self, value):
        if value == self.config.max_clip_duration:
            return
        self.config.max_clip_duration = value
        self.notify('max-clip-duration')
        if self.save_config:
            self.config.save()

    @GObject.Property(flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.EXPLICIT_NOTIFY)
    def mosaic_pre_cleaning(self):
        return self.config.mosaic_pre_cleaning

    @mosaic_pre_cleaning.setter
    def mosaic_pre_cleaning(self, value):
        if value == self.config.mosaic_pre_cleaning:
            return
        self.config.mosaic_pre_cleaning = value
        self.notify('mosaic-pre-cleaning')
        if self.save_config:
            self.config.save()

    @GObject.Property(flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.EXPLICIT_NOTIFY)
    def mute_audio(self):
        return self.config.mute_audio

    @mute_audio.setter
    def mute_audio(self, value):
        if value == self.config.mute_audio:
            return
        self.config.mute_audio = value
        self.notify('mute-audio')
        if self.save_config:
            self.config.save()

    @GObject.Property(flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.EXPLICIT_NOTIFY)
    def export_crf(self):
        return self.config.export_crf

    @export_crf.setter
    def export_crf(self, value):
        if value == self.config.export_crf:
            return
        self.config.export_crf = value
        self.notify('export-crf')
        if self.save_config:
            self.config.save()

    @GObject.Property(flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.EXPLICIT_NOTIFY)
    def export_codec(self):
        return self.config.export_codec

    @export_codec.setter
    def export_codec(self, value):
        if value == self.config.export_codec:
            return
        self.config.export_codec = value
        self.notify('export-codec')
        if self.save_config:
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

    @Gtk.Template.Callback()
    def button_config_reset_callback(self, button_clicked):
        try:
            self.save_config = False
            default_config = Config()
            self.preview_mode = default_config.preview_mode
            self.mosaic_restoration_model = default_config.mosaic_restoration_model
            self.device = default_config.device
            self.preview_buffer_duration = default_config.preview_buffer_duration
            self.max_clip_duration = default_config.max_clip_duration
            self.export_crf = default_config.export_crf
            self.export_codec = default_config.export_codec
            self.mute_audio = default_config.mute_audio
            self.mosaic_pre_cleaning = default_config.mosaic_pre_cleaning
            self.init_sidebar_from_config(self.config)
            self.config.save()
        finally:
            self.save_config = True