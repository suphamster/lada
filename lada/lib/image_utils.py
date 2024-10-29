import math

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid

def pad_image(img, max_height, max_width, mode='zero'):
    height, width = img.shape[:2]
    if height == max_height and width == max_width:
        return img, (0, 0, 0, 0)
    pad_h = max_height - height
    pad_w = max_width - width
    pad_h_t = math.ceil(pad_h / 2)
    pad_h_b = math.floor(pad_h / 2)
    pad_w_l = math.ceil(pad_w / 2)
    pad_w_r = math.floor(pad_w / 2)
    pad = (pad_h_t, pad_h_b,pad_w_l, pad_w_r)
    padded_image =  pad_image_by_pad(img, pad, mode)
    assert padded_image.shape[:2] == (max_height, max_width)
    return padded_image, pad

def pad_image_by_pad(img, pad, mode='zero'):
    (pad_h_t, pad_h_b,pad_w_l, pad_w_r) = pad
    if img.ndim == 3:
        if mode == 'zero':
            padded_img = np.pad(img, ((pad_h_t, pad_h_b),(pad_w_l, pad_w_r),(0,0)), mode='constant', constant_values=0)
        elif mode == 'reflect':
            padded_img = np.pad(img, ((pad_h_t, pad_h_b),(pad_w_l, pad_w_r),(0,0)), mode='reflect')
        else:
            raise NotImplementedError()
    else:
        padded_img = np.pad(img, ((pad_h_t, pad_h_b),(pad_w_l, pad_w_r)), mode='constant', constant_values=0)
    return padded_img

def repad_image(imgs, pads, mode='reflect'):
    assert len(imgs) == len(pads)
    padded_imgs = []
    for img, pad in zip(imgs, pads):
        (pad_h_t, pad_h_b, pad_w_l, pad_w_r) = pad
        h, w = img.shape[:2]
        if img.ndim == 3:
            if mode == 'zero':
                padded_img = np.pad(img[pad_h_t:h-pad_h_b, pad_w_l:w-pad_w_r], ((pad_h_t, pad_h_b),(pad_w_l, pad_w_r),(0,0)), mode='constant', constant_values=0)
            elif mode == 'reflect':
                padded_img = np.pad(img[pad_h_t:h-pad_h_b, pad_w_l:w-pad_w_r], ((pad_h_t, pad_h_b),(pad_w_l, pad_w_r),(0,0)), mode='reflect')
            else:
                raise NotImplementedError()
        else:
            padded_img = np.pad(img[pad_h_t:h-pad_h_b, pad_w_l:w-pad_w_r], ((pad_h_t, pad_h_b),(pad_w_l, pad_w_r)), mode='constant', constant_values=0)
        padded_imgs.append(padded_img)
    return padded_imgs

def unpad_image(img, pad):
    (pad_h_t, pad_h_b, pad_w_l, pad_w_r) = pad
    h, w = img.shape[:2]
    unpadded_img = img[pad_h_t:h - pad_h_b, pad_w_l:w - pad_w_r]
    return unpadded_img

def img2tensor(imgs, bgr2rgb=True, float32=True, normalize_neg1_pos1 = False):
    """Numpy array to tensor. HWC to CHW

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
            if normalize_neg1_pos1:
                img = (img/ 255.0 - 0.5) / 0.5
            else:
                img = img / 255.
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor)):
        raise TypeError(f'list of tensors expected, got {type(tensor)}')

    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    return result

def resize(img, size: int|tuple[int, int], interpolation=cv2.INTER_LINEAR):
    if type(size) == int:
        h, w = img.shape[:2]
        if max(w, h) == size:
            return img
        if w >= h:
            scale_factor = size / w
            new_h = size
            new_w = math.ceil(h * scale_factor) if scale_factor < 1.0 else math.floor(h * scale_factor)
        else:
            scale_factor = size / h
            new_w = size
            new_h = math.ceil(w * scale_factor) if scale_factor < 1.0 else math.floor(w * scale_factor)
    else:
        if img.shape[:2] == size:
            return img
        new_h, new_w = size
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    assert size == max(resized_img.shape[:2]) if type(size) == int else size == resized_img.shape[:2]
    return resized_img

def resize_simple(img,size,interpolation=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    if np.min((w,h)) == size:
        return img
    if w >= h:
        res = cv2.resize(img,(int(size*w/h), size),interpolation=interpolation)
    else:
        res = cv2.resize(img,(size, int(size*h/w)),interpolation=interpolation)
    return res