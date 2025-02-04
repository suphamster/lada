import io
import random

import av
import cv2
import numpy as np

from lada.lib import Image, random_utils
from lada.lib import video_utils
from lada.lib.degradations import generate_gaussian_noise, add_jpg_compression, random_mixed_kernels


def apply_video_compression(imgs: list[Image], codec, bitrate, crf=None):
    imgs = video_utils.pad_to_compatible_size_for_video_codecs(imgs)
    h, w = imgs[0].shape[:2]
    degraded_imgs = _apply_video_compression(imgs, codec, bitrate, crf)
    return [img[0:h, 0:w, :] for img in degraded_imgs]

def _apply_video_compression(imgs: list[Image], codec, bitrate, crf=None):
    # source: https://mmagic.readthedocs.io/en/latest/_modules/mmagic/datasets/transforms/random_degradations.html#RandomVideoCompression.__call__
    """This is the function to apply random compression on images.

    Args:
        imgs (list of ndarray): training images

    Returns:
        Tensor: images after randomly compressed
    """

    buf = io.BytesIO()
    with av.open(buf, 'w', 'mp4') as container:
        options = {}
        if crf: options["crf"] = str(crf)
        if codec == 'libx265': options['x265-params'] = 'log_level=error'
        if codec == 'libx264' or codec == 'libx265': options['preset'] = 'veryfast'
        stream = container.add_stream(codec, rate=1, options=options)
        stream.height = imgs[0].shape[0]
        stream.width = imgs[0].shape[1]
        stream.pix_fmt = 'yuv420p'
        if bitrate: stream.bit_rate = bitrate

        for img in imgs:
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            frame.pict_type = av.video.frame.PictureType.NONE
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

    outputs = []
    with av.open(buf, 'r', 'mp4') as container:
        if container.streams.video:
            for frame in container.decode(**{'video': 0}):
                img = frame.to_rgb().to_ndarray().astype(np.uint8)
                outputs.append(img)

    return outputs

class MosaicRandomDegradationParamsV2:
    def __init__(self, repeatable_random=False):
        rng_random, rng_numpy = random_utils.get_rngs(repeatable_random)
        codecs = {
            'libx264': (16, 28),
            'libx265': (20, 36),
            'libvpx-vp9': (6_000, 16_000),
            'mpeg2video': (18_000, 40_000),
        }
        available_codecs = ['libx264', 'libx265', 'libvpx-vp9', 'mpeg2video']
        codec_probabilities = [0.3, 0.3, 0.3, 0.1]
        self.video_codec = str(rng_numpy.choice(available_codecs, p=codec_probabilities))
        value = rng_numpy.randint(codecs[self.video_codec][0], codecs[self.video_codec][1] + 1)
        if self.video_codec in ('libvpx-vp9', 'mpeg2video'):
            self.video_crf = None
            self.video_bitrate = value
        else:
            self.video_crf = value
            self.video_bitrate = None
        self.should_add_video_compression = rng_random.random()<0.8

class MosaicRandomDegradationParams:
    def __init__(self, should_down_sample=True, should_add_noise=True, should_add_image_compression=True, should_add_video_compression=False, should_add_blur=False, repeatable_random=False):
        rng_random, rng_numpy = random_utils.get_rngs(repeatable_random)

        # down sample
        self.should_down_sample = should_down_sample and rng_random.random()<0.5
        down_sample_range = [0.5, 2.0]
        self.scale = rng_numpy.uniform(down_sample_range[0], down_sample_range[1])
        # noise
        self.should_add_noise = should_add_noise and rng_random.random()<0.3
        sigma_range = [0,2]
        self.sigma = rng_numpy.uniform(sigma_range[0], sigma_range[1])
        self.repeatable_noise = repeatable_random
        # jpeg compression
        self.should_add_jpeg_compression = should_add_image_compression and rng_random.random()<0.5
        jpeg_range = [70, 90]
        self.jpeg_quality = rng_numpy.uniform(jpeg_range[0], jpeg_range[1])
        # video compression
        codecs = {
            "libx264": (15_000, 100_000),
            "libx265": (10_000, 60_000),
            "libvpx-vp9": (10_000, 60_000),
            "mpeg4": (15_000, 100_000)
        }
        self.video_codec = rng_random.choice(list(codecs.keys()))
        self.video_bitrate = rng_numpy.randint(codecs[self.video_codec][0], codecs[self.video_codec][1] + 1)
        self.should_add_video_compression = should_add_video_compression and rng_random.random()<0.5
        # blur
        self.should_add_blur = should_add_blur and rng_random.random() < 0.5
        self.blur_kernel = random_mixed_kernels(
            kernel_list = ('iso', 'aniso'),
            kernel_prob = (0.5, 0.5),
            kernel_size = 41,
            sigma_x_range = (0., 2),
            sigma_y_range = (0., 2),
            noise_range=None,
            repeatable_random=repeatable_random)


def apply_frame_degradation(img: Image, degradation_params: MosaicRandomDegradationParams) -> Image:
    h, w = img.shape[:2]
    img_lq = img.astype(np.float32) / 255.
    # blur
    if degradation_params.should_add_blur:
        img_lq = cv2.filter2D(img_lq, -1, degradation_params.blur_kernel)
    # downsample
    if degradation_params.should_down_sample:
        img_lq = cv2.resize(img_lq, (int(w // degradation_params.scale), int(h // degradation_params.scale)), interpolation=cv2.INTER_LINEAR)
    # noise
    if degradation_params.should_add_noise:
        noise = generate_gaussian_noise(img_lq, degradation_params.sigma, False, repeatable_random=degradation_params.repeatable_noise)
        img_lq = np.clip(img_lq + noise, 0, 1)
    # jpeg compression
    if degradation_params.should_add_jpeg_compression:
        img_lq = add_jpg_compression(img_lq, degradation_params.jpeg_quality)
    # scale back up
    if degradation_params.should_down_sample:
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
    img_lq = (img_lq * 255.).astype(np.uint8)
    return img_lq


def apply_video_degradation(imgs: list[Image], degradation_params: MosaicRandomDegradationParams) -> list[Image]:
    imgs_lq = []
    for img in imgs:
        imgs_lq.append(apply_frame_degradation(img, degradation_params))
    # video_compression
    if degradation_params.should_add_video_compression:
        imgs_lq = apply_video_compression(imgs_lq, degradation_params.video_codec, degradation_params.video_bitrate)
    return imgs_lq

def apply_video_degradation_v2(imgs: list[Image], degradation_params: MosaicRandomDegradationParamsV2) -> list[Image]:
    assert max(imgs[0].shape[:2]) == 256, "video compression degradation expects width/height of 256px"
    return apply_video_compression(imgs, degradation_params.video_codec, degradation_params.video_bitrate)