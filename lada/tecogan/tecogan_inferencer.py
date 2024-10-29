import torch
import numpy as np

from lada.lib import video_utils
from lada.lib.image_utils import img2tensor
from lada.tecogan.models import define_model
from lada.tecogan.utils.base_utils import parse_configs_inference

def load_model(opt_path, gpu_id=0):
    opt = parse_configs_inference(opt_path, str(gpu_id))
    model = define_model(opt)
    return model

def inference(img_lqs, model, device="cuda"):
    img_lqs = video_utils.resize_video_frames(img_lqs, 64)

    with torch.no_grad():
        imgs = torch.stack(img2tensor(img_lqs, bgr2rgb=False, float32=True), dim=0)
        imgs = imgs.to(device)

        # imgs -> tchw | rgb | float32 0.0-1.0
        data = dict(lr=imgs)
        model.prepare_inference_data(data)

        # hq_imgs -> thwc | rgb | uint8 0-255
        hq_imgs = model.infer()

        restored_imgs = list(hq_imgs)
        return restored_imgs

def main():
    model = load_model('TecoGAN_4xSR/test.yml')

    frame1 = np.zeros((256,256,3), dtype=np.uint8)
    frame2 = np.zeros((256,256,3), dtype=np.uint8)
    frame3 = np.zeros((256,256,3), dtype=np.uint8)
    frame4 = np.zeros((256,256,3), dtype=np.uint8)

    video = [frame1, frame2, frame3, frame4]
    inference(video, model)

if __name__ == '__main__':
    main()