import math
import random
from dataclasses import dataclass

import cv2
import ultralytics.engine.results
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
    _box: ultralytics.engine.results.Boxes
    _mask: ultralytics.engine.results.Masks
    object_detected: bool = False
    object_id: int = None
    random_extend_masks: bool = False

    @property
    def mask(self) -> Mask:
        mask = convert_yolo_mask(self._mask, self.frame.shape)
        mask = mask_utils.fill_holes(mask)
        return mask

    @property
    def box(self) -> Box:
        return convert_yolo_box(self._box, self.frame.shape)


class NsfwFrameGenerator:
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
                yolo_box, yolo_mask = choose_biggest_detection(results, tracking_mode=False)

                nsfw_frame = NsfwFrame(frame_num, results.orig_img, yolo_box, yolo_mask, yolo_box is not None, object_id=0, random_extend_masks=self.random_extend_masks)
                yield nsfw_frame

                if self.sampling != -1:
                    frame_num += math.ceil(self.sampling * self.video_fps)
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                else:
                    frame_num += 1

                success, frame = self.video.read()