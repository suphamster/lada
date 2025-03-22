from lada.lib.ultralytics_utils import disable_ultralytics_telemetry, set_default_settings
disable_ultralytics_telemetry()
set_default_settings()

from ultralytics import YOLO

model = YOLO('yolo11n-seg.yaml')
model.train(data='configs/yolo/mosaic_segmentation_dataset_config.yaml', epochs=150, imgsz=640, name="train_mosaic_detection_yolo11n")

# model = YOLO('yolo11n.yaml')
# model.train(data='configs/yolo/mosaic_detection_dataset_config.yaml', epochs=150, imgsz=640, name="train_mosaic_detection_yolo11n_seg")
