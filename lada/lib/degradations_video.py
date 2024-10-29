import io

import av
import numpy as np

from lada.lib import Image
from lada.lib import video_utils


def apply_video_compression(imgs: list[Image], codec, bitrate):
    imgs = video_utils.pad_to_compatible_size_for_video_codecs(imgs)
    h, w = imgs[0].shape[:2]
    degraded_imgs = _apply_video_compression(imgs, codec, bitrate)
    return [img[0:h, 0:w, :] for img in degraded_imgs]

def _apply_video_compression(imgs: list[Image], codec, bitrate):
    # source: https://mmagic.readthedocs.io/en/latest/_modules/mmagic/datasets/transforms/random_degradations.html#RandomVideoCompression.__call__
    """This is the function to apply random compression on images.

    Args:
        imgs (list of ndarray): training images

    Returns:
        Tensor: images after randomly compressed
    """

    buf = io.BytesIO()
    with av.open(buf, 'w', 'mp4') as container:
        stream = container.add_stream(codec, rate=1)
        stream.height = imgs[0].shape[0]
        stream.width = imgs[0].shape[1]
        stream.pix_fmt = 'yuv420p'
        stream.bit_rate = bitrate

        for img in imgs:
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            frame.pict_type = 'NONE'
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
