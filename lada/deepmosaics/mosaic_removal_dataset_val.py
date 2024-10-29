import glob
import os.path
from lada.deepmosaics.models import loadmodel
from lada.deepmosaics.inference import restore_video_frames
from lada.lib.video_utils import read_video_frames, get_video_meta_data, write_frames_to_video_file
MOSAIC_REMOVAL_VAL_MOSAIC_DIR = 'datasets/mosaic_removal_vid/val/mosaic'
OUT_DIR = 'mosaic_removal_v2_validation_runs/deepmosaics'
MODEL_PATH = 'deepmosaics/model_weights/clean_youknow_video.pth'
opts = {
    "gpu_id": '0',
    "model_path": MODEL_PATH
}
model = loadmodel.video(opts)
for video_path in glob.glob(os.path.join(MOSAIC_REMOVAL_VAL_MOSAIC_DIR, '*')):
    video_metadata = get_video_meta_data(video_path)
    images = read_video_frames(video_path, float32=False)
    restored_images = restore_video_frames(opts, model, images)
    filename = os.path.basename(video_path)
    out_path = os.path.join(OUT_DIR, filename)
    fps = video_metadata.video_fps
    write_frames_to_video_file(restored_images, out_path, fps, codec='x265')
