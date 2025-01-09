from operator import itemgetter
from typing import Optional

from lada.lib.ultralytics_utils import disable_ultralytics_telemetry, convert_yolo_boxes
from lada.lib.scene_utils import box_overlap
from lada.lib import Image, Box
from ultralytics import YOLO
import random

disable_ultralytics_telemetry()

class WatermarkDetector:
    def __init__(self, model: YOLO, device):
        self.model = model
        self.device = device

    def detect(self, images:list[Image], boxes:Optional[list[Box]]=None, max_samples=12, min_confidence=0.2, min_positive_detections=3, batch_size=4) -> bool:
        if boxes:
            indices = list(range(len(images)))
            indices = random.sample(indices, min(len(images), max_samples))
            samples = itemgetter(*indices)(images)
            samples_boxes = itemgetter(*indices)(boxes)
        else:
            samples = random.sample(images, min(len(images), max_samples))
            samples_boxes = None

        batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
        positive_detections = 0
        for batch_idx, batch in enumerate(batches):
            batch_prediction_results = self.model.predict(source=batch, stream=False, verbose=False, device=self.device)
            for result_idx, results in enumerate(batch_prediction_results):
                sample_idx = batch_idx * batch_size + result_idx
                detections = results.boxes.conf.size(dim=0)
                if detections == 0:
                    continue
                detection_confidences = results.boxes.conf.tolist()
                detection_boxes = convert_yolo_boxes(results.boxes, results.orig_shape)
                single_image_watermark_detected = any(conf > min_confidence and (not samples_boxes or box_overlap(detection_boxes[i], samples_boxes[sample_idx])) for i, conf in enumerate(detection_confidences))
                if single_image_watermark_detected:
                    positive_detections += 1
        watermark_detected = positive_detections > min_positive_detections
        #print(f"watermark detector: watermark {watermark_detected}, detected {positive_detections}/{len(samples)}")
        return watermark_detected