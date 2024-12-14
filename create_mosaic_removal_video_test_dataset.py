import argparse
import pathlib
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics import settings

from lada.lib import video_utils
from lada.lib.mosaic_detector import MosaicDetectorDeprecated
from lada.lib.clean_mosaic_utils import clean_cropped_mosaic
from lada.lib.video_utils import get_video_meta_data
from lada.pidinet import pidinet_inference

# Disable analytics and crash reporting
settings.update({'sync': False})

def save_vid(frame_dir: pathlib.Path, fileprefix: str, imgs: np.ndarray, fps=30, gray=False):
    frame_dir.mkdir(parents=True, exist_ok=True)
    if gray:
        video_utils.write_masks_to_video_file(imgs, str(frame_dir.joinpath(f"{fileprefix}.mkv").absolute()), fps)
    else:
        video_utils.write_frames_to_video_file(imgs, str(frame_dir.joinpath(f"{fileprefix}.mp4").absolute()), fps)

def get_dir_and_file_prefix(output_dir, name, file_name, scene_id, save_flat=False, file_suffix="-", video=False):
    if save_flat:
        file_prefix = f"{file_name}-{scene_id:06d}{file_suffix}"
        frame_dir = output_dir.joinpath(name)
    else:
        if video:
            file_prefix = f"{scene_id:06d}"
            frame_dir = output_dir.joinpath(name).joinpath(file_name)
        else:
            file_prefix = ""
            frame_dir = output_dir.joinpath(name).joinpath(file_name).joinpath(f"{scene_id:06d}")
    return frame_dir, file_prefix

def parse_args():
    parser = argparse.ArgumentParser("Create video dataset")
    parser.add_argument('--output-root', type=Path, default='video_dataset', help="set output root directory")
    parser.add_argument('--input', type=Path, help="path to video file or directory")
    parser.add_argument('--mosaic-detection-model-path', type=str,
                        default='yolo/runs/segment/train_mosaic_detection_yolov9c/weights/best.pt')
    parser.add_argument('--mosaic-cleaning-model-path', type=str,
                        default='experiments/pidinet/run2_tiny/save_models/checkpoint_019.pth')
    parser.add_argument('--mosaic-cleaning-model-type', type=str, default='tiny')
    parser.add_argument('--device', type=str, default="cuda:0", help="device to run the YOLO model on. E.g. 'cuda' or 'cuda:0'")
    parser.add_argument('--max-clip-length', type=int, default=180)
    parser.add_argument('--clip-size', type=int, default=256)
    parser.add_argument('--canny', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mosaic_detection_model = YOLO(args.mosaic_detection_model_path)
    pidinet_model = None if args.canny else pidinet_inference.load_model(args.mosaic_cleaning_model_path, model_type=args.mosaic_cleaning_model_type, device=args.device)
    pad_mode = 'zero'
    file_suffix = '-'

    output_dir = args.output_root
    if not output_dir.exists():
        output_dir.mkdir()
    input_path = args.input
    video_files = input_path.glob("*") if input_path.is_dir() else [input_path]

    for file_index, file_path in enumerate(video_files):
        print(f"{file_index}, Processing {file_path.name}")

        video_metadata = get_video_meta_data(file_path)

        mosaic_frames_generator = MosaicDetectorDeprecated(mosaic_detection_model, args.input, args.max_clip_length, args.clip_size, pad_mode=pad_mode, device=args.device, preserve_relative_scale=True, dont_preserve_relative_scale=True)
        for clip_idx, (clip_scaling_preserved, clip_scaling_not_preserved) in enumerate(mosaic_frames_generator()):

            frame_dir, file_prefix = get_dir_and_file_prefix(output_dir, "mosaic_unscaled", clip_scaling_preserved.file_path.name, clip_scaling_preserved.id, True, file_suffix, video=True)
            save_vid(frame_dir, file_prefix, clip_scaling_preserved.get_clip_images(), fps = video_metadata.video_fps, gray = False)

            frame_dir, file_prefix = get_dir_and_file_prefix(output_dir, "mosaic_scaled", clip_scaling_not_preserved.file_path.name, clip_scaling_not_preserved.id, True, file_suffix, video=True)
            save_vid(frame_dir, file_prefix, clip_scaling_not_preserved.get_clip_images(), fps = video_metadata.video_fps, gray = False)

            cleaned_images = []
            for (cropped_img, cropped_mask, cropped_box, orig_crop_shape, pad) in clip_scaling_preserved:
                cleaned_images.append(clean_cropped_mosaic(cropped_img, cropped_mask, pad, pidinet_model=pidinet_model))
            frame_dir, file_prefix = get_dir_and_file_prefix(output_dir, "mosaic_unscaled_cleaned", clip_scaling_preserved.file_path.name, clip_scaling_preserved.id, True, file_suffix, video=True)
            save_vid(frame_dir, file_prefix, cleaned_images, fps = video_metadata.video_fps, gray = False)

            cleaned_images = []
            for (cropped_img, cropped_mask, cropped_box, orig_crop_shape, pad) in clip_scaling_not_preserved:
                cleaned_images.append(clean_cropped_mosaic(cropped_img, cropped_mask, pad, pidinet_model=pidinet_model))
            frame_dir, file_prefix = get_dir_and_file_prefix(output_dir, "mosaic_scaled_cleaned", clip_scaling_not_preserved.file_path.name, clip_scaling_not_preserved.id, True, file_suffix, video=True)
            save_vid(frame_dir, file_prefix, cleaned_images, fps = video_metadata.video_fps, gray = False)


if __name__ == '__main__':
    main()
