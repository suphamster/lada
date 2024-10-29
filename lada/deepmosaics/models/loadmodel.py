import torch
from lada.deepmosaics.models import model_util
from lada.deepmosaics.models.BVDNet import define_G as video_G

def video(opt):
    netG = video_G(N=2,n_blocks=4,gpu_id=opt['gpu_id'])
    netG.load_state_dict(torch.load(opt['model_path']))
    netG = model_util.todevice(netG,opt['gpu_id'])
    netG.eval()
    return netG

