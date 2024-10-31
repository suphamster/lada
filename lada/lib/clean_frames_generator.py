from dataclasses import dataclass

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


def choose_biggest_detection(result: ultralytics.engine.results.Results) -> tuple[
    ultralytics.engine.results.Boxes | None, ultralytics.engine.results.Masks | None]:
    """
    Returns the biggest detection box and mask of a YOLO Results set
    """
    box = None
    mask = None
    yolo_box: ultralytics.engine.results.Boxes
    yolo_mask: ultralytics.engine.results.Masks
    for i, yolo_box in enumerate(result.boxes):
        if yolo_box.id is None:
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

    def __call__(self, *args, **kwargs):
        stride_window_remaining = -self.stride_length_frames
        stride_window_positive = stride_window_remaining >= 0
        for frame_num, results in enumerate(
                self.model.track(source=self.video_meta_data.video_file, stream=True, verbose=False, tracker="bytetrack.yaml", device=self.device)):
            if not self.stride_mode_active or stride_window_remaining > 0:
                yolo_box, yolo_mask = choose_biggest_detection(results)

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
