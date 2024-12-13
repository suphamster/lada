from ultralytics import YOLO
import argparse
from pathlib import Path

from lada import disable_ultralytics_telemetry

disable_ultralytics_telemetry()

parser = argparse.ArgumentParser("Train YOLO model")
parser.add_argument('--config', type=Path, help="path to .yaml config file", default='nsfw_detection-dataset.yaml')
args = parser.parse_args()

model = YOLO('yolo11m-seg.yaml')
model.train(data=args.config, epochs=200, name="train_nsfw_detection_yolo11m")

# python train-yolo-nsfw-detection.py
