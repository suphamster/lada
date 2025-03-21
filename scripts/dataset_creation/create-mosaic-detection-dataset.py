import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from os import path as osp
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from lada.lib import visualization_utils, mask_utils, degradation_utils, image_utils
from lada.lib.mosaic_utils import addmosaic_base, get_random_parameter
from lada.lib.nsfw_frame_detector import NsfwImageDetector
from lada.lib.threading_utils import clean_up_completed_futures
from lada.lib.ultralytics_utils import disable_ultralytics_telemetry

disable_ultralytics_telemetry()

def crop_to_box(img, box):
    t, l, b, r = box
    cropped_img = img[t:b + 1, l:r + 1]
    return cropped_img

def process_image_file(file_path, nsfw_frame_generator, output_root, show=False, window_name="mosaic"):
    nsfw_frame = nsfw_frame_generator()
    if not nsfw_frame:
        return

    img = nsfw_frame.frame
    mask = nsfw_frame.mask
    box = nsfw_frame.box

    t, l, b, r = box
    width, height = r - l + 1, b - t + 1
    if min(width, height) < 40:
        # skip tiny detections
        return

    cropped_img = crop_to_box(img, box)
    cropped_mask = crop_to_box(mask, box)

    mosaic_size, mod, rect_ratio, feather_size = get_random_parameter(mask)
    mask_dilation_iterations = np.random.choice(range(2))
    cropped_mask_mosaic = mask_utils.dilate_mask(cropped_mask, iterations=mask_dilation_iterations)

    cropped_img_mosaic, cropped_mask_mosaic = addmosaic_base(cropped_img,
                                                             cropped_mask_mosaic,
                                                             mosaic_size,
                                                             model=mod, rect_ratio=rect_ratio,
                                                             feather=feather_size)

    img_mosaic = img.copy()
    t, l, b, r = box
    img_mosaic[t:b + 1, l:r + 1, :] = cropped_img_mosaic
    mask_mosaic = np.zeros_like(mask, dtype=mask.dtype)
    mask_mosaic[t:b + 1, l:r + 1] = cropped_mask_mosaic

    degradation_params = degradation_utils.MosaicRandomDegradationParams(should_down_sample=True,
                                                                         should_add_noise=True,
                                                                         should_add_image_compression=True,
                                                                         should_add_video_compression=True,
                                                                         should_add_blur=False)

    img_mosaic = degradation_utils.apply_video_degradation([img_mosaic], degradation_params)[0]

    if show:
        show_img = visualization_utils.overlay_mask_boundary(img_mosaic, mask_mosaic, color=(0, 255, 0))
        show_img = visualization_utils.overlay_mask_boundary(show_img, mask, color=(255, 0, 0))

        cv2.imshow(window_name, show_img)

        while True:
            key_pressed = cv2.waitKey(1)
            if key_pressed & 0xFF == ord("n"):
                break
    else:
        name = osp.splitext(os.path.basename(file_path))[0]
        cv2.imwrite(f"{output_root}/img/{name}-{nsfw_frame.frame_number:06d}.jpg", img_mosaic,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(f"{output_root}/mask/{name}-{nsfw_frame.frame_number:06d}.png", mask_mosaic)

def get_files(dir, filter_func):
    file_list = []
    for r, d, f in os.walk(dir):
        for file in f:
            file_path = osp.join(r, file)
            if filter_func(file_path):
                file_list.append(Path(file_path))
    return file_list

def parse_args():
    parser = argparse.ArgumentParser("Create mosaic detection dataset")
    parser.add_argument('--output-root', type=Path, help="directory where resulting images/masks are saved")
    parser.add_argument('--input-root', type=Path, help="directory containing video files")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--model', type=str, default="model_weights/lada_nsfw_detection_model_v1.3.pt", help="path to YOLO model")
    parser.add_argument('--workers', type=int, default=4, help="number of worker threads")
    parser.add_argument('--start-index', type=int, default=0, help="Can be used to continue a previous run. Note the index number next to last processed file name")
    parser.add_argument('--show', default=False, action=argparse.BooleanOptionalAction, help="show each sample")
    parser.add_argument('--max-file-limit', type=int, default=None, help="instead of processing all files found in input-root dir it will choose files randomly up to the given limit")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model = YOLO(args.model)
    nsfw_frame_generator = NsfwImageDetector(model, args.device, random_extend_masks=True, conf=0.75)

    if not args.show:
        os.makedirs(f"{args.output_root}/mask", exist_ok=True)
        os.makedirs(f"{args.output_root}/img", exist_ok=True)
        jobs = []

    selected_files = get_files(args.input_root, image_utils.is_image_file)
    if args.max_file_limit and len(selected_files) > args.max_file_limit:
        selected_files = random.choices(selected_files, k=args.max_file_limit)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for file_idx, file_path in enumerate(selected_files):
            if file_idx < args.start_index or len(list(args.output_root.glob(f"*/{file_path.name}*"))) > 0:
                print(f"{file_idx}, Skipping {file_path.name}: Already processed")
                continue
            print(f"{file_idx}, Processing {file_path.name}")
            if args.show:
                process_image_file(file_path, args.output_root, nsfw_frame_generator, args.show)
            else:
                jobs.append(executor.submit(process_image_file, file_path, args.output_root, nsfw_frame_generator))
                clean_up_completed_futures(jobs)
    wait(jobs, return_when=ALL_COMPLETED)
    clean_up_completed_futures(jobs)

    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()