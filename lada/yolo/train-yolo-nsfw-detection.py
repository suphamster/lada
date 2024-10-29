from ultralytics import YOLO
import argparse
from pathlib import Path
from ultralytics import settings

# Disable analytics and crash reporting
settings.update({'sync': False})
print(settings)

parser = argparse.ArgumentParser("Train YOLO model")
parser.add_argument('--config', type=Path, help="path to .yaml config file", default='nsfw_detection-dataset.yaml')
args = parser.parse_args()

model = YOLO('yolo11m-seg.yaml')
#model.train(data=args.config, epochs=150, imgsz=640, optimizer='SGD')
model.train(data=args.config, epochs=200)

# python train-yolo-nsfw-detection.py
