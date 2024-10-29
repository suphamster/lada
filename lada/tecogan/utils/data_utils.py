import os
import os.path as osp

from scipy import signal
import numpy as np
import torch
import torch.nn.functional as F

from lada.lib.video_utils import write_frames_to_video_file


def create_kernel(sigma, ksize=None):
    if ksize is None:
        ksize = 1 + 2 * int(sigma * 3.0)

    gkern1d = signal.gaussian(ksize, std=sigma).reshape(ksize, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gaussian_kernel = gkern2d / gkern2d.sum()
    zero_kernel = np.zeros_like(gaussian_kernel)

    kernel = np.float32([
        [gaussian_kernel, zero_kernel, zero_kernel],
        [zero_kernel, gaussian_kernel, zero_kernel],
        [zero_kernel, zero_kernel, gaussian_kernel]])

    kernel = torch.from_numpy(kernel)

    return kernel


def downsample_bd(data, kernel, scale, pad_data):
    """
        Note:
            1. `data` should be torch.FloatTensor (data range 0~1) in shape [nchw]
            2. `pad_data` should be enabled in model testing
            3. This function is device agnostic, i.e., data/kernel could be on cpu or gpu
    """

    if pad_data:
        # compute padding params
        kernel_h, kernel_w = kernel.shape[-2:]
        pad_h, pad_w = kernel_h - 1, kernel_w - 1
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l

        # pad data
        data = F.pad(data, (pad_l, pad_r, pad_t, pad_b), 'reflect')

    # blur + down sample
    data = F.conv2d(data, kernel, stride=scale, bias=None, padding=0)

    return data


def rgb_to_ycbcr(img):
    """ Coefficients are taken from the  official codes of DUF-VSR
        This conversion is also the same as that in BasicSR

        Parameters:
            :param  img: rgb image in type np.uint8
            :return: ycbcr image in type np.uint8
    """

    T = np.array([
        [0.256788235294118, -0.148223529411765,  0.439215686274510],
        [0.504129411764706, -0.290992156862745, -0.367788235294118],
        [0.097905882352941,  0.439215686274510, -0.071427450980392],
    ], dtype=np.float64)

    O = np.array([16, 128, 128], dtype=np.float64)

    img = img.astype(np.float64)
    res = np.matmul(img, T) + O
    res = res.clip(0, 255).round().astype(np.uint8)

    return res


def float32_to_uint8(inputs):
    """ Convert np.float32 array to np.uint8

        Parameters:
            :param input: np.float32, (NT)CHW, [0, 1]
            :return: np.uint8, (NT)CHW, [0, 255]
    """
    return np.uint8(np.clip(np.round(inputs * 255), 0, 255))


def save_sequence(seq_dir, seq_data, name, to_bgr=False, fps=None):
    if to_bgr:
        seq_data = seq_data[..., ::-1]  # rgb2bgr
    if not fps:
        fps = 30
    os.makedirs(seq_dir, exist_ok=True)
    file_path = osp.join(seq_dir, f"{name}.mp4")
    write_frames_to_video_file(seq_data, file_path, fps)
