from lada.pidinet import models
import cv2
import torch
import numpy as np
import os
from lada.lib.image_utils import img2tensor, tensor2img

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_model(checkpoint_path, device="cuda", model_type="tiny"):
    args = dotdict(dict(config="carv4", dil=True, sa=True))
    if model_type == "base":
        model = models.pidinet(args)
    elif model_type == "tiny":
        model = models.pidinet_tiny(args)
    else:
        pass
    if type(device) == str:
        device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    remove_prefix = "module."
    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in
                  checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    return model

def inference(model, images, min_confidence=0.4):
    device = next(model.parameters()).device
    with torch.no_grad():
        input = torch.stack(img2tensor(images, bgr2rgb=False, float32=True), dim=0)
        results = model(input.to(device))
        results = results[-1]
        results = [torch.where(results < min_confidence, 0, 1)]
        output = tensor2img(results, rgb2bgr=False, out_type=np.uint8, min_max=(0, 1))
        return output

if __name__ == "__main__":
    CHECKPOINT_PATH = "experiments/pidinet/run1/save_models/checkpoint_019.pth"
    IMAGE_PATH = "mpv-shot0001.jpg"
    model = load_model(CHECKPOINT_PATH)

    img = cv2.imread(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images = [img]

    mosaic_edges = inference(model, images)

    os.environ["QT_QPA_PLATFORM"] = 'xcb'
    for edge in mosaic_edges:
        cv2.imshow("window", edge)
        key_pressed = cv2.waitKey()
        if key_pressed & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()