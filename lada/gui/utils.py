import torch

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
