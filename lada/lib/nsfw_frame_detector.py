from dataclasses import dataclass

import numpy as np
import ultralytics.models
from lada.lib import Mask, Box, Image
from lada.lib import mask_utils
from lada.lib.ultralytics_utils import convert_yolo_box, convert_yolo_mask, choose_biggest_detection


@dataclass
class NsfwFrame:
    frame: Image
    box: Box | None
    mask: Mask | None

def apply_random_mask_extensions(mask: Mask) -> Mask:
    value = np.random.choice([0, 0, 1, 1, 2])
    return mask_utils.extend_mask(mask, value)

def get_nsfw_frame(yolo_results: ultralytics.engine.results.Results, random_extend_masks: bool) -> NsfwFrame | None:
    yolo_box, yolo_mask = choose_biggest_detection(yolo_results, tracking_mode=False)
    object_detected = yolo_box is not None
    if not object_detected:
        return None

    mask = convert_yolo_mask(yolo_mask, yolo_results.orig_img.shape)
    mask = mask_utils.fill_holes(mask)

    if random_extend_masks:
        mask = apply_random_mask_extensions(mask)
        box = mask_utils.get_box(mask)
    else:
        box = convert_yolo_box(yolo_box, yolo_results.orig_img.shape)

    t, l, b, r = box
    width, height = r - l + 1, b - t + 1
    if min(width, height) < 40:
        # skip tiny detections
        return None

    return NsfwFrame(yolo_results.orig_img, box, mask)

class NsfwImageDetector:
    def __init__(self, model: ultralytics.models.YOLO, device=None, random_extend_masks=False, conf=0.25):
        self.model = model
        self.device = device
        self.random_extend_masks = random_extend_masks
        self.conf = conf

    def detect(self, file_path: str) -> NsfwFrame | None:
        for results in self.model.predict(source=file_path, stream=False, verbose=False, device=self.device, conf=self.conf):
            return get_nsfw_frame(results, self.random_extend_masks)
