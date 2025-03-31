import os

if "LADA_MODEL_WEIGHTS_DIR" in os.environ:
  MODEL_WEIGHTS_DIR = os.environ["LADA_MODEL_WEIGHTS_DIR"]
else:
  MODEL_WEIGHTS_DIR = "model_weights"

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

VERSION = '0.7.0-dev'

LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")
