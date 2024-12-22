from ultralytics import YOLO
import argparse
from pathlib import Path
from lada.lib.ultralytics_utils import disable_ultralytics_telemetry, set_default_settings

disable_ultralytics_telemetry()
set_default_settings()

parser = argparse.ArgumentParser("Train mosaic detection model")
parser.add_argument('--config', type=Path, help="path to .yaml config file", default='lada/yolo/mosaic_detection_dataset_config.yaml')
args = parser.parse_args()

model = YOLO('yolo11m-seg.yaml')
model.train(data=args.config, epochs=150, name="train_mosaic_detection_yolo11m")
