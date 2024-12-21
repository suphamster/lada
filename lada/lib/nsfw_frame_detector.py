import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
import ultralytics.models
from lada.lib import Mask, Box, Image, VideoMetadata
from lada.lib import mask_utils
from lada.lib.ultralytics_utils import disable_ultralytics_telemetry, convert_yolo_box, convert_yolo_mask, \
    choose_biggest_detection

disable_ultralytics_telemetry()


@dataclass
class NsfwFrame:
    frame_number: int
    frame: Image
    box: Box | None
    mask: Mask | None

def apply_random_mask_extensions(mask: Mask) -> Mask:
    value = np.random.choice([0, 0, 1, 1, 2])
    return mask_utils.extend_mask(mask, value)

def get_nsfw_frame(yolo_results: ultralytics.engine.results.Results, random_extend_masks: bool, frame_num: int) -> NsfwFrame | None:
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

    return NsfwFrame(frame_num, yolo_results.orig_img, box, mask)

class NsfwFrameGenerator:
    def __init__(self, model: ultralytics.models.YOLO, file_path, device=None, random_extend_masks=False, conf=0.25):
        self.model = model
        self.file_path = file_path
        self.device = device
        self.random_extend_masks = random_extend_masks
        self.conf = conf

    def __call__(self, *args, **kwargs) -> NsfwFrame | None:
        frame = cv2.imread(self.file_path)
        for results in self.model.predict(source=frame, stream=False, verbose=False, device=self.device, conf=self.conf):
            return get_nsfw_frame(results, self.random_extend_masks, 0)

class NsfwFramesGenerator:
    def __init__(self, model: ultralytics.models.YOLO, video_meta_data: VideoMetadata, device=None, sampling=-1, random_start=False, random_extend_masks=False, conf=0.25):
        self.sampling = sampling
        self.model = model
        self.device = device
        self.conf = conf
        self.random_extend_masks = random_extend_masks

        self.video = cv2.VideoCapture(video_meta_data.video_file)

        self.video_fps = video_meta_data.video_fps
        self.video_length = video_meta_data.frames_count

        self.frame_start = random.randint(0, sampling) if (random_start and sampling > 0) else 0
        self.frame_end = self.video_length - 1

    def __call__(self, *args, **kwargs):
        if self.frame_start > self.frame_end or self.frame_start > self.video_length or self.frame_end < 0:
            return

        frame_num = self.frame_start
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = self.video.read()
        while success and frame_num <= self.frame_end:
            for results in self.model.predict(source=frame, stream=False, verbose=False, device=self.device, conf=self.conf):
                nsfw_frame = get_nsfw_frame(results, self.random_extend_masks, frame_num)
                if nsfw_frame:
                    yield nsfw_frame

                if self.sampling != -1:
                    frame_num += math.ceil(self.sampling * self.video_fps)
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                else:
                    frame_num += 1

                success, frame = self.video.read()