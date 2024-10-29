import json
import os.path
import random
import glob

import torch
import torch.utils.data as data
from lada.lib import image_utils
from lada.lib import video_utils

class MosaicVideoDataset(data.Dataset):
    def __init__(self, opt):
        super(MosaicVideoDataset, self).__init__()
        self.opt = opt
        self.lq_size = opt.get('lq_size', 256)
        self.gt_size = opt.get('gt_size', 256)
        self.gt_root, self.lq_root, self.meta_root = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_meta']
        self.max_frame_count = opt['num_frame']
        self.min_frame_count = opt['min_num_frame'] if 'min_num_frame' in opt else opt['num_frame']

        self.clip_names = []
        self.fps = []
        self.total_num_frames = []
        self.pads = []
        for meta_path in glob.glob(os.path.join(self.meta_root, '*')):
            with open(meta_path, 'r') as meta_file:
                meta_json = json.load(meta_file)
                filename = f"{os.path.splitext(os.path.basename(meta_path))[0]}.mp4"
                frame_num = meta_json["frames_count"]
                if frame_num < self.min_frame_count:
                    continue
                self.clip_names.append(filename)
                self.fps.append(meta_json["fps"])
                self.total_num_frames.append(frame_num)
                self.pads.append(meta_json["pad"])


    def __getitem__(self, index):
        clip_name = self.clip_names[index]
        total_num_frames = self.total_num_frames[index]

        if self.max_frame_count == -1:
            # select the full clip
            start_frame_idx = 0
            end_frame_idx = total_num_frames - 1
        else:
            # randomly select shorter clip of length num_frame
            start_frame_idx = random.randint(0, total_num_frames - self.max_frame_count)
            end_frame_idx = start_frame_idx + self.max_frame_count

        pads = self.pads[index][start_frame_idx:end_frame_idx]
        # get the neighboring LQ and GT frames
        vid_lq_path = os.path.join(self.lq_root, clip_name)
        vid_gt_path = os.path.join(self.gt_root, clip_name)
        img_lqs = video_utils.read_video_frames(vid_lq_path, float32=True, start_idx=start_frame_idx, end_idx=end_frame_idx)
        img_lqs = image_utils.repad_image(img_lqs, pads)
        img_lqs = video_utils.resize_video_frames(img_lqs, self.lq_size)
        img_gts = video_utils.read_video_frames(vid_gt_path, float32=True,  start_idx=start_frame_idx, end_idx=end_frame_idx)
        img_gts = image_utils.repad_image(img_gts, pads)
        img_gts = video_utils.resize_video_frames(img_gts, self.gt_size)

        img_gts = video_utils.img2tensor(img_gts, float32=True, bgr2rgb=True)
        img_lqs = video_utils.img2tensor(img_lqs, float32=True, bgr2rgb=True)

        img_gts = torch.stack(img_gts, dim=0)
        img_lqs = torch.stack(img_lqs, dim=0)

        #img_gts = torch.transpose(img_gts, 1,3)
        #img_lqs = torch.transpose(img_lqs, 1,3)

        # tchw|rgb|float32
        #print(f"selected from dataset: {clip_name}--({start_frame_idx:06d}-{end_frame_idx:06d})")
        return {
            'lr': img_lqs,
            'gt': img_gts,
            'seq_idx': f"{clip_name}--({start_frame_idx:06d}-{end_frame_idx:06d})",
            'fps': self.fps[index]
        }

    def __len__(self):
        return len(self.clip_names)