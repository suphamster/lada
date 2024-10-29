import glob
import os.path
import sys

from lada.rvrt import rvrt_inferencer
from lada.lib.video_utils import read_video_frames, get_video_meta_data, write_frames_to_video_file

MODEL_PATH = 'rvrt/experiments/008_train_rvrt_x_scale1_v4/models/100000_G.pth'
DEVICE = 'cuda:0'
MOSAIC_REMOVAL_VAL_MOSAIC_DIR = 'datasets/mosaic_removal_vid/val/mosaic'
OUT_DIR = 'mosaic_removal_v2_validation_runs/rvrt'
model = rvrt_inferencer.get_model(model_path=MODEL_PATH, device=DEVICE)
for video_path in glob.glob(os.path.join(MOSAIC_REMOVAL_VAL_MOSAIC_DIR, '*')):
    video_metadata = get_video_meta_data(video_path)
    images = read_video_frames(video_path, float32=False)
    restored_images = rvrt_inferencer.inference(images, model)
    filename = os.path.basename(video_path)
    out_path = os.path.join(OUT_DIR, filename)
    fps = video_metadata.video_fps
    write_frames_to_video_file(restored_images, out_path, fps, codec='x265')
