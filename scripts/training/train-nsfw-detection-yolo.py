from ultralytics import YOLO
from lada.lib.ultralytics_utils import disable_ultralytics_telemetry, set_default_settings

disable_ultralytics_telemetry()
set_default_settings()

model = YOLO('yolo11m-seg.pt')
model.train(data='configs/yolo/nsfw_detection_dataset_config.yaml', epochs=200, imgsz=640, name="train_nsfw_detection_yolo11m")
