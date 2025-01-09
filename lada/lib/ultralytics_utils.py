import os

import numpy as np
import torch
import ultralytics.engine
from ultralytics import settings
from ultralytics.utils.ops import scale_image

from lada.lib import Box, Mask


def disable_ultralytics_telemetry():
    # Disable analytics and crash reporting
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    settings.update({'sync': False})

def set_default_settings():
    settings.update({'runs_dir': './experiments/yolo', 'datasets_dir': './datasets', 'tensorboard': True})

def convert_yolo_box(yolo_box: ultralytics.engine.results.Boxes, img_shape) -> Box:
    _box = yolo_box.xyxy[0]
    l = int(torch.clip(_box[0], 0, img_shape[1]).item())
    t = int(torch.clip(_box[1], 0, img_shape[0]).item())
    r = int(torch.clip(_box[2], 0, img_shape[1]).item())
    b = int(torch.clip(_box[3], 0, img_shape[0]).item())
    return t, l, b, r

def convert_yolo_boxes(yolo_box: ultralytics.engine.results.Boxes, img_shape) -> list[Box]:
    _boxes = yolo_box.xyxy
    boxes = []
    for _box in _boxes:
        l = int(torch.clip(_box[0], 0, img_shape[1]).item())
        t = int(torch.clip(_box[1], 0, img_shape[0]).item())
        r = int(torch.clip(_box[2], 0, img_shape[1]).item())
        b = int(torch.clip(_box[3], 0, img_shape[0]).item())
        box = t, l, b, r
        boxes.append(box)
    return boxes

def convert_yolo_mask(yolo_mask: ultralytics.engine.results.Masks, img_shape) -> Mask:
    mask_img = _to_mask_img(yolo_mask.data)
    mask_img = scale_image(mask_img, img_shape)
    if mask_img.ndim == 2:
        mask_img = np.expand_dims(mask_img, axis=-1)
    mask_img = np.where(mask_img > 127, 255, 0).astype(np.uint8)
    return mask_img


def _to_mask_img(masks, class_val=0, pixel_val=255) -> Mask:
    masks_tensor = (masks != class_val).int() * pixel_val
    mask_img = masks_tensor.cpu().numpy()[0].astype(np.uint8)
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
