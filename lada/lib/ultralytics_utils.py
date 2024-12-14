import os
from ultralytics import settings


def disable_ultralytics_telemetry():
    # Disable analytics and crash reporting
    os.environ["YOLO_AUTOINSTALL"] = "false"
    settings.update({'sync': False})
