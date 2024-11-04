import math
import random
from dataclasses import dataclass
from typing import Generator

import cv2
import numpy as np
import torch
import ultralytics.engine.results
import ultralytics.models
from lada.lib import Mask, Box, Image, VideoMetadata
from lada.lib import mask_utils
from ultralytics.utils.ops import scale_image


@dataclass
class CleanFrame:
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

def convert_yolo_box(yolo_box: ultralytics.engine.results.Boxes, img_shape) -> Box:
    _box = yolo_box.xyxy[0]
    l = int(torch.clip(_box[0], 0, img_shape[1]).item())
    t = int(torch.clip(_box[1], 0, img_shape[0]).item())
    r = int(torch.clip(_box[2], 0, img_shape[1]).item())
    b = int(torch.clip(_box[3], 0, img_shape[0]).item())
    return t, l, b, r


def _to_mask_img(masks, class_val=0, pixel_val=255) -> Mask:
    masks_tensor = (masks != class_val).int() * pixel_val
    mask_img = masks_tensor.cpu().numpy()[0].astype(np.uint8)
    return mask_img


def convert_yolo_mask(yolo_mask: ultralytics.engine.results.Masks, img_shape) -> Mask:
    mask_img = _to_mask_img(yolo_mask.data)
    mask_img = scale_image(mask_img, img_shape)
    if mask_img.ndim == 2:
        mask_img = np.expand_dims(mask_img, axis=-1)
    mask_img = np.where(mask_img > 127, 255, 0).astype(np.uint8)
    return mask_img


def choose_biggest_detection(result: ultralytics.engine.results.Results, tracking_mode=True) -> tuple[
    ultralytics.engine.results.Boxes | None, ultralytics.engine.results.Masks | None]:
    """
    Returns the biggest detection box and mask of a YOLO Results set
    """
    box = None
    mask = None
    yolo_box: ultralytics.engine.results.Boxes
    yolo_mask: ultralytics.engine.results.Masks
    for i, yolo_box in enumerate(result.boxes):
        if tracking_mode and yolo_box.id is None:
            continue
        yolo_mask = result.masks[i]
        if box is None:
            box = yolo_box
            mask = yolo_mask
        else:
            box_dims = box.xywh[0]
            _box_dims = yolo_box.xywh[0]
            box_size = box_dims[2] * box_dims[3]
            _box_size = _box_dims[2] * _box_dims[3]
            if _box_size > box_size:
                box = yolo_box
                mask = yolo_mask
    return box, mask

class CleanFramesGenerator:
    def __init__(self, model: ultralytics.models.YOLO, video_meta_data: VideoMetadata, device=None, random_extend_masks=False, stride_mode_activation_length=None, stride_length=None):
        self.model = model
        self.device = torch.device(device) if device is not None else device
        self.random_extend_masks = random_extend_masks
        self.video_meta_data = video_meta_data
        self.stride_mode_active = False
        self.stride_length_frames = 0
        if stride_mode_activation_length:
            if self.video_meta_data.duration > stride_mode_activation_length:
                self.stride_mode_active = True
                self.stride_length_frames = int(round(stride_length * self.video_meta_data.video_fps))
                print(f"yolo generator: stride mode activated for file {self.video_meta_data.video_file}: file duration: {int(self.video_meta_data.duration / 60)} (min), stride length: {stride_length} (s) / {self.stride_length_frames} (frames)")

    def __call__(self, *args, **kwargs) -> Generator[Image, None, None]:
        stride_window_remaining = -self.stride_length_frames
        stride_window_positive = stride_window_remaining >= 0
        for frame_num, results in enumerate(
                self.model.track(source=self.video_meta_data.video_file, stream=True, verbose=False, tracker="bytetrack.yaml", device=self.device)):
            if not self.stride_mode_active or stride_window_remaining > 0:
                yolo_box, yolo_mask = choose_biggest_detection(results, tracking_mode=True)

                clean_frame = CleanFrame(frame_num, results.orig_img, yolo_box, yolo_mask, yolo_box is not None,
                                   int(yolo_box.id.item()) if yolo_box is not None else None, self.random_extend_masks)
                yield clean_frame
                stride_window_remaining -= 1
            elif stride_window_remaining < 0:
                stride_window_remaining += 1
            else:
                if stride_window_positive:
                    # print(f"CleanFramesGenerator: finished stride at frame {frame_num:06d}, next stride will be skipped")
                    stride_window_remaining = -(self.stride_length_frames-1)
                    stride_window_positive = False
                else:
                    # print(f"CleanFramesGenerator: finished stride at frame {frame_num:06d}, next stride will be processed")
                    stride_window_remaining = self.stride_length_frames-1
                    stride_window_positive = True

class CleanFrameGeneratorSimple:
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

                clean_frame = CleanFrame(frame_num, results.orig_img, yolo_box, yolo_mask, yolo_box is not None, object_id=0, random_extend_masks=self.random_extend_masks)
                yield clean_frame

                if self.sampling != -1:
                    frame_num += math.ceil(self.sampling * self.video_fps)
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                else:
                    frame_num += 1

                success, frame = self.video.read()