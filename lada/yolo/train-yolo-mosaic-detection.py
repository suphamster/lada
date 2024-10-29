from ultralytics import YOLO
import argparse
from pathlib import Path
from ultralytics import settings

# Disable analytics and crash reporting
settings.update({'sync': False, 'runs_dir': 'experiments/yolo', 'datasets_dir': '../../datasets', 'tensorboard': True})
print(settings)

parser = argparse.ArgumentParser("Train YOLO model")
parser.add_argument('--config', type=Path, help="path to .yaml config file", default='mosaic-detection-dataset.yaml')
args = parser.parse_args()

model = YOLO('yolov9c-seg.yaml')
# model.train(data=args.config, epochs=200, imgsz=640, name="train_mosaic_detection_yolo9c")
model.train(data=args.config, epochs=200, imgsz=640, batch=4, name="train_mosaic_detection_yolo9c")

# python train-yolo-mosaic-detection.py
