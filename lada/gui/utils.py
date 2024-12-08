import os

import torch

from lada.gui.config import MODEL_FILES_TO_NAMES


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
    return [(i, torch.cuda.get_device_properties(i).name) for i in range(torch.cuda.device_count())]


def get_available_models():
    available_models = []
    for file_path in MODEL_FILES_TO_NAMES:
        if os.path.exists(file_path):
            available_models.append(MODEL_FILES_TO_NAMES[file_path])
    return available_models
