from ultralytics import YOLO
from lada.lib.ultralytics_utils import disable_ultralytics_telemetry, set_default_settings

disable_ultralytics_telemetry()
set_default_settings()

model = YOLO('yolo11m-seg.yaml')
model.train(data='configs/yolo/mosaic_detection_dataset_config.yaml', epochs=150, imgsz=640, name="train_mosaic_detection_yolo11m")
