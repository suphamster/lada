from lada.rvrt.models.network_rvrt import RVRT as net
from lada.lib.video_utils import img2tensor
import torch
import numpy as np

def inference(lq, model):
    # inference
    with torch.no_grad():
        imgs = [i.astype(np.float32) / 255. for i in lq]
        imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
        imgs = torch.stack(imgs, dim=0)
        imgs = torch.unsqueeze(imgs, 0)
        imgs = imgs.cuda()
        output = test_video(imgs, model)

    # conert back to image
    restored_imgs = []
    for i in range(output.shape[1]):
        img = output[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if img.ndim == 3:
            img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
        restored_imgs.append(img)

    return restored_imgs


def prepare_model_dataset(model_path="KAIR/experiments/007_train_rvrt_x_scal4/models/55000_G.pth"):
    scale = 1
    # inputconv_groups = [1, 3, 3, 3, 3, 3]
    inputconv_groups = [1, 3, 4, 6, 8, 4]
    embed_dims = [192, 192, 192]
    # scale = 4
    # inputconv_groups = [1, 1, 1, 1, 1, 1]
    # embed_dims = [144, 144, 144]
    model = net(upscale=scale, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                depths=[2, 2, 2], embed_dims=embed_dims, num_heads=[6, 6, 6],
                inputconv_groups=inputconv_groups, deformable_groups=12, attention_heads=12,
                attention_window=[3, 3], cpu_cache_length=100)

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)

    return model

def test_clip(lq, model, scale=1):
    ''' test the clip as a whole or as patches. '''

    sf = scale
    window_size = [2,8, 8]

    _, _, _, h_old, w_old = lq.size()
    h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
    w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

    lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
    lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

    output = model(lq).detach().cpu()

    output = output[:, :, :, :h_old*sf, :w_old*sf]

    return output

def test_video(lq, model):
        window_size = [2, 8, 8]
        d_old = lq.size(1)
        d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
        lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
        output = test_clip(lq, model)
        output = output[:, :d_old, :, :, :]

        return output

def get_model(device="cuda:0", model_path="KAIR/experiments/007_train_rvrt_x_scale4/models/105000_G.pth"):
    model = prepare_model_dataset(model_path)
    torch_device = torch.device(device)
    model.eval()
    model = model.to(torch_device)
    return model

def get_args():
    args = {
        "scale": 4,
        "window_size": [2,8,8],
        "nonblind_denoising": False
    }
    args = dotdict(args)
    return args

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def main():
    model = get_model()

    frame1 = np.zeros((256,256,3), dtype=np.uint8)
    frame2 = np.zeros((256,256,3), dtype=np.uint8)
    frame3 = np.zeros((256,256,3), dtype=np.uint8)
    frame4 = np.zeros((256,256,3), dtype=np.uint8)

    video = [frame1, frame2, frame3, frame4]
    inference(video, model)

if __name__ == '__main__':
    main()