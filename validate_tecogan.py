import argparse
import glob
import os.path

import torch

from lada.lib.video_utils import read_video_frames, get_video_meta_data, write_frames_to_video_file
from lada.tecogan.tecogan_inferencer import load_model, inference

def validate(mosaic_dir, out_dir, config_path, gpu_id):
    model = load_model(config_path, gpu_id)
    with torch.no_grad():
        for video_path in glob.glob(os.path.join(mosaic_dir, '*')):
            video_metadata = get_video_meta_data(video_path)
            images = read_video_frames(video_path, float32=False)
            restored_images = inference(images, model)
            filename = os.path.basename(video_path)
            out_path = os.path.join(out_dir, filename)
            fps = video_metadata.video_fps
            write_frames_to_video_file(restored_images, out_path, fps)

def parse_args():
    parser = argparse.ArgumentParser(description='Validate a model on a validation dataset')
    parser.add_argument('--out-dir', help='the dir to save logs and models')
    parser.add_argument('--in-dir')
    parser.add_argument('--model-path')
    parser.add_argument('--config-path', help='path to test.yml')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    validate(args.in_dir, args.out_dir, args.config_path, 0)


