import glob
import json
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data
from lada.basicvsrpp.mmagic.data_sample import DataSample
from lada.basicvsrpp.mmagic.registry import DATASETS

import lada.lib.video_utils as video_utils
from lada.lib.mosaic_utils import addmosaic_base
from lada.lib.image_utils import unpad_image, pad_image_by_pad, repad_image
from lada.lib.degradation_utils import MosaicRandomDegradationParams, apply_video_degradation

# import cv2
# import os
# os.environ["QT_QPA_PLATFORM"] = "xcb"

@DATASETS.register_module()
class MosaicVideoDataset(data.Dataset):
    def __init__(self, **opt):
        super(MosaicVideoDataset, self).__init__()
        self.opt = opt
        self.scale = opt.get('scale', 1)
        self.lq_size = opt.get('lq_size', 256)
        self.gt_root, self.meta_root = opt['dataroot_gt'], opt['dataroot_meta']
        self.lq_root = opt.get('dataroot_lq')
        self.use_hflip = opt.get('use_hflip', False)
        self.degrade = opt.get('degrade', False)
        self.mask_root = opt.get('dataroot_mask')
        self.max_frame_count = opt['num_frame']
        self.min_frame_count = opt['min_num_frame'] if 'min_num_frame' in opt else opt['num_frame']
        self.random_mosaic_block_size = opt.get('random_mosaic_block_size', True)
        self.repad = False

        self.clip_names = []
        self.fps = []
        self.total_num_frames = []
        self.pads = []
        self.mosaic_params = []
        self.base_mosaic_block_sizes = []
        for meta_path in glob.glob(os.path.join(self.meta_root, '*')):
            with open(meta_path, 'r') as meta_file:
                meta_json = json.load(meta_file)
                clip_name = f"{os.path.splitext(os.path.basename(meta_path))[0]}"
                frame_num = meta_json.get('frame_count', meta_json.get("frames_count"))
                assert frame_num is not None
                if frame_num < self.min_frame_count:
                    continue
                self.clip_names.append(clip_name)
                self.fps.append(meta_json["fps"])
                self.total_num_frames.append(frame_num)
                self.pads.append(meta_json["pad"])
                self.mosaic_params.append(meta_json.get("mosaic"))
                self.base_mosaic_block_sizes.append(meta_json.get("base_mosaic_block_size"))

    def get_block_size(self, index):
        base_mosaic_block_size = self.base_mosaic_block_sizes[index]
        if self.random_mosaic_block_size and base_mosaic_block_size:
            mosaic_size = int(base_mosaic_block_size["mosaic_size_v1_normal"] * random.uniform(0.8, 2.2))
        else:
            mosaic_size = self.mosaic_params[index]["mosaic_size"]
        return mosaic_size

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

        vid_gt_path = os.path.join(self.gt_root, clip_name + ".mp4")
        img_gts = video_utils.read_video_frames(vid_gt_path, float32=False, start_idx=start_frame_idx, end_idx=end_frame_idx)
        if self.repad:
            img_gts = repad_image(img_gts, pads)

        if self.lq_root:
            vid_lq_path = os.path.join(self.lq_root, clip_name + ".mp4")
            img_lqs = video_utils.read_video_frames(vid_lq_path, float32=False, start_idx=start_frame_idx, end_idx=end_frame_idx)
            
            if self.repad:
                img_lqs = repad_image(img_lqs, pads)
        else:
            vid_mask_gt_path = os.path.join(self.mask_root, clip_name + ".mkv")
            mask_gts = video_utils.read_video_frames(vid_mask_gt_path, float32=False, start_idx=start_frame_idx, end_idx=end_frame_idx, binary_frames=True)
            mosaic_params = self.mosaic_params[index]
            mosaic_size = self.get_block_size(index)

            img_lqs = []
            for img_gt, mask_gt, pad in zip(img_gts, mask_gts, pads):
                img_lq, mask_lq = addmosaic_base(unpad_image(img_gt, pad),
                                                 unpad_image(mask_gt, pad),
                                                 mosaic_size,
                                                 model=mosaic_params["mod"],
                                                 rect_ratio=mosaic_params["rect_ratio"],
                                                 feather=mosaic_params["feather_size"])
                img_lqs.append(pad_image_by_pad(img_lq, pad))
            if self.degrade:
                degradation_params = MosaicRandomDegradationParams(should_down_sample=True, should_add_noise=True,
                                                                   should_add_image_compression=True,
                                                                   should_add_video_compression=True)
                img_lqs = apply_video_degradation(img_lqs, degradation_params)

        img_gts = video_utils.resize_video_frames(img_gts, self.lq_size)
        img_lqs = video_utils.resize_video_frames(img_lqs, self.lq_size)

        if self.use_hflip and random.random() < 0.5:
            img_gts = [np.fliplr(img) for img in img_gts]
            img_lqs = [np.fliplr(img) for img in img_lqs]

        # for gt, lq in zip(img_gts, img_lqs):
        #     cv2.imshow("gt", gt)
        #     cv2.imshow("lq", lq)
        #     cv2.waitKey(500)

        img_gts = video_utils.img2tensor(img_gts, float32=False, bgr2rgb=True)
        img_lqs = video_utils.img2tensor(img_lqs, float32=False, bgr2rgb=True)

        #print(f"selected from dataset: {clip_name}--({start_frame_idx:06d}-{end_frame_idx:06d})")

        data_sample = DataSample(gt_img=torch.stack(img_gts, dim=0))
        data_sample.set_predefined_data({
            'img': img_lqs,
            'img_channel_order': 'rgb',
            'img_color_type': 'color',
            'gt_img': img_gts,
            'gt_path': vid_gt_path,
            'gt_channel_order': 'rgb',
            'gt_color_type': 'color',
            'key': clip_name,
            'fps': self.fps[index]
        })
        inputs = torch.stack(img_lqs, dim=0)
        # inputs = tensor (T,C,H,W)
        return {'inputs': inputs, 'data_samples': data_sample}

    def __len__(self):
        return len(self.clip_names)