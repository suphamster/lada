# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
from mmcv.transforms import to_tensor


def can_convert_to_image(value):
    """Judge whether the input value can be converted to image tensor via
    :func:`images_to_tensor` function.

    Args:
        value (any): The input value.

    Returns:
        bool: If true, the input value can convert to image with
            :func:`images_to_tensor`, and vice versa.
    """
    if isinstance(value, (List, Tuple)):
        return all([can_convert_to_image(v) for v in value])
    elif isinstance(value, np.ndarray) and len(value.shape) > 1:
        return True
    elif isinstance(value, torch.Tensor):
        return True
    else:
        return False


def image_to_tensor(img):
    """Trans image to tensor.

    Args:
        img (np.ndarray): The original image.

    Returns:
        Tensor: The output tensor.
    """

    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    img = np.ascontiguousarray(img)
    tensor = to_tensor(img).permute(2, 0, 1).contiguous()

    return tensor


def all_to_tensor(value):
    """Trans image and sequence of frames to tensor.

    Args:
        value (np.ndarray | list[np.ndarray] | Tuple[np.ndarray]):
            The original image or list of frames.

    Returns:
        Tensor: The output tensor.
    """

    if not can_convert_to_image(value):
        return value

    if isinstance(value, (List, Tuple)):
        # sequence of frames
        if len(value) == 1:
            tensor = image_to_tensor(value[0])
        else:
            frames = [image_to_tensor(v) for v in value]
            tensor = torch.stack(frames, dim=0)
    elif isinstance(value, np.ndarray):
        tensor = image_to_tensor(value)
    else:
        # Maybe the data has been converted to Tensor.
        tensor = to_tensor(value)

    return tensor


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (np.ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        np.ndarray: Reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        if isinstance(img, np.ndarray):
            img = img.transpose(1, 2, 0)
        elif isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0)
    return img


def to_numpy(img, dtype=np.float64):
    """Convert data into numpy arrays of dtype.

    Args:
        img (Tensor | np.ndarray): Input data.
        dtype (np.dtype): Set the data type of the output. Default: np.float64

    Returns:
        img (np.ndarray): Converted numpy arrays data.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    elif not isinstance(img, np.ndarray):
        raise TypeError('Only support torch.tensor and np.ndarray, '
                        f'but got type {type(img)}')

    img = img.astype(dtype)

    return img
