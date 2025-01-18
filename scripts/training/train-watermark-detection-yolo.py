from ultralytics import YOLO
from lada.lib.ultralytics_utils import disable_ultralytics_telemetry, set_default_settings

disable_ultralytics_telemetry()
set_default_settings()

model = YOLO('yolo11s.pt')
model.train(data='configs/yolo/watermark_detection_dataset_config.yaml', epochs=100, imgsz=512, single_cls=True, name="train_watermark_detection_yolo11s")
