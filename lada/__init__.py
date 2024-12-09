import os

MODEL_WEIGHTS_DIR = '/app/model_weights' if "FLATPAK_ID" in os.environ else "model_weights"

if "FLATPAK_ID" in os.environ:
  os.environ["YOLO_CONFIG_DIR"] = "/var/config/yolo"

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

VERSION = '0.3.1-dev'