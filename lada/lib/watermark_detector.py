from lada.lib.ultralytics_utils import disable_ultralytics_telemetry
from lada.lib import Image
from ultralytics import YOLO
import random
disable_ultralytics_telemetry()

class WatermarkDetector:
    def __init__(self, model: YOLO, device):
        self.model = model
        self.device = device

    def detect(self, images=list[Image], max_samples=12, min_confidence=0.2, min_positive_detections=3, batch_size=4) -> bool:
        samples = random.sample(images, min(len(images), max_samples))
        batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
        positive_detections = 0
        for batch in batches:
            batch_prediction_results = self.model.predict(source=batch, stream=False, verbose=False, device=self.device)
            for i, results in enumerate(batch_prediction_results):
                detections = results.boxes.conf.size(dim=0)
                if detections == 0:
                    continue
                detection_confidences = results.boxes.conf.tolist()
                single_image_watermark_detected = any(conf > min_confidence for conf in detection_confidences)
                if single_image_watermark_detected:
                    positive_detections += 1
        watermark_detected = positive_detections > min_positive_detections
        print(f"watermark detector: watermark {watermark_detected}, detected {positive_detections}/{len(samples)}")
        return watermark_detected