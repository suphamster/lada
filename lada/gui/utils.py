import torch
from lada.lib import video_utils

def is_device_available(device: str) -> bool:
    device = device.lower()
    if device == 'cpu':
        return True
    elif device.startswith("cuda:"):
        return device_to_gpu_id(device) < torch.cuda.device_count()
    return False


def device_to_gpu_id(device) -> int | None:
    if device.startswith("cuda:"):
        return int(device.split(":")[-1])
    return None


def get_available_gpus():
    gpus = []
    for id in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_properties(id).name
        # We're using these GPU names in a ComboBox but libadwaita sets up the label with max-width-chars: 20 and there does not
        # seem to be a way to overwrite this. So let's try to make sure GPU names are below 20 characters to be readable
        if gpu_name.startswith("NVIDIA GeForce RTX"):
            gpu_name = gpu_name.replace("NVIDIA GeForce RTX", "RTX")
        gpus.append((id, gpu_name))
    return gpus

def skip_if_uninitialized(f):
    def noop(*args):
        return
    def wrapper(*args):
        return f(*args) if args[0].init_done else noop
    return wrapper

def get_available_video_codecs():
    filter_list = ['libx264', 'h264_nvenc', 'libx265', 'hevc_nvenc', 'libsvtav1', 'librav1e', 'libaom-av1', 'av1_nvenc']
    return [codec_short_name for codec_short_name, codec_long_name in video_utils.get_available_video_encoder_codecs() if codec_short_name in filter_list]