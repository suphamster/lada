import os
from ultralytics import settings

MODEL_WEIGHTS_DIR = '/app/model_weights' if "FLATPAK_ID" in os.environ else "model_weights"

if "FLATPAK_ID" in os.environ:
  os.environ["YOLO_CONFIG_DIR"] = "/var/config/yolo"

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

VERSION = '0.3.2-dev'

LOG_LEVEL = os.environ["LOG_LEVEL"] if "LOG_LEVEL" in os.environ else 'WARNING'

def disable_ultralytics_telemetry():
  # Disable analytics and crash reporting
  os.environ["YOLO_AUTOINSTALL"] = "false"
  settings.update({'sync': False})

disable_ultralytics_telemetry()