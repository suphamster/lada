import logging

from gi.repository import GLib
from pathlib import Path
import json

from lada import LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class Config:
    _defaults = {
        'preview_mode': 'mosaic-removal',
        'mosaic_restoration_model': 'basicvsrpp-1.2',
        'mosaic_detection_model': 'v3.1-accurate',
        'export_codec': 'h264',
        'export_crf': 20,
        'preview_buffer_duration': 0,
        'max_clip_duration': 180,
        'device': 'cuda:0',
        'mute_audio': False
    }

    def __init__(self):
        self.preview_mode = self._defaults['preview_mode']
        self.mosaic_restoration_model = self._defaults['mosaic_restoration_model']
        self.mosaic_detection_model = self._defaults['mosaic_detection_model']
        self.export_codec = self._defaults['export_codec']
        self.export_crf = self._defaults['export_crf']
        self.preview_buffer_duration = self._defaults['preview_buffer_duration']
        self.max_clip_duration = self._defaults['max_clip_duration']
        self.device = self._defaults['device']
        self.mute_audio = self._defaults['mute_audio']
        self.loaded = False

    def save(self):
        config_file_path = self.get_config_file_path()
        if not config_file_path.parent.exists():
            config_file_path.parent.mkdir(parents=True)
        with open(config_file_path, 'w') as f:
            config_dict = self._as_dict()
            json.dump(config_dict, f)
            logger.info(f"Saved config file {config_file_path}: {config_dict}")

    def load_config(self):
        config_file_path = self.get_config_file_path()
        if not config_file_path.exists():
            logger.info(f"Config file doesn't exist at {config_file_path}")
            self.save()
            return

        try:
            with open(config_file_path, 'r') as f:
                config_dict = json.load(f)
                self._from_dict(config_dict)
                logger.info(f"Loaded config file {config_file_path}: {config_dict}")
                self.loaded = True
        except Exception as e:
            logger.error(f"Error loading config file {config_file_path}, falling back to defaults: {e}")

    def _as_dict(self) -> dict:
        return {
            'preview_mode': self.preview_mode,
            'mosaic_restoration_model': self.mosaic_restoration_model,
            'mosaic_detection_model': self.mosaic_detection_model,
            'export_codec': self.export_codec,
            'export_crf': self.export_crf,
            'preview_buffer_duration': self.preview_buffer_duration,
            'max_clip_duration': self.max_clip_duration,
            'device': self.device,
            'mute_audio': self.mute_audio
        }

    def get_default_value(self, key):
        return self._defaults.get(key)

    def _from_dict(self, dict):
        for key in self._defaults:
            if key in dict and dict[key] is not None:
                setattr(self, key, dict[key])

    def get_default_restoration_model(self):
        return self.get_default_value('mosaic_restoration_model')

    def get_default_detection_model(self):
        return self.get_default_value('mosaic_detection_model')

    def get_config_file_path(self) -> Path:
        base_config_dir = GLib.get_user_config_dir()
        return Path(base_config_dir).joinpath('lada').joinpath('lada.conf')

CONFIG = Config()