import argparse
import os
import random
from concurrent import futures
from os import path as osp
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from lada.lib.nsfw_frame_detector import NsfwFramesGenerator, NsfwFrameGenerator, NsfwFrame
from lada.lib.ultralytics_utils import disable_ultralytics_telemetry
from lada.lib.mosaic_utils import addmosaic_base, get_random_parameter
from lada.lib import visualization_utils, video_utils, mask_utils, degradation_utils, image_utils

disable_ultralytics_telemetry()

def crop_to_box(img, box):
    t, l, b, r = box
    cropped_img = img[t:b + 1, l:r + 1]
    return cropped_img

def process_frame(nsfw_frame: NsfwFrame, file_path, output_root, show=False, window_name="mosaic"):
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

def process_video_file(file_path, output_root, model, device, sampling=60, show=False, window_name="mosaic"):
    video_meta_data = video_utils.get_video_meta_data(file_path)
    nsfw_frame_generator = NsfwFramesGenerator(model, video_meta_data, device, sampling, random_extend_masks=True, random_start=True, conf=0.75)

    nsfw_frame: NsfwFrame
    for nsfw_frame in nsfw_frame_generator():
        process_frame(nsfw_frame, file_path, output_root, show=show, window_name=window_name)

def process_image_file(file_path, output_root, model, device, show=False, window_name="mosaic"):
    nsfw_frame_generator = NsfwFrameGenerator(model, file_path, device, random_extend_masks=True, conf=0.75)
    nsfw_frame = nsfw_frame_generator()
    if nsfw_frame:
        process_frame(nsfw_frame, file_path, output_root, show=show, window_name=window_name)

def get_files(dir, filter_func):
    file_list = []
    for r, d, f in os.walk(dir):
        for file in f:
            file_path = osp.join(r, file)
            if filter_func(file_path):
                file_list.append(Path(file_path))
    return file_list

def clean_up_completed_futures(completed_futures):
    for job in futures.as_completed(completed_futures):
        exception = job.exception()
        if exception:
            print(f"ERR(main): failed processing file: {type(exception).__name__}: {exception}")
            # raise exception # todo: for some images degradation_util::_apply_video_compression throws an exception. lets ignore and skip those for now...
        completed_futures.remove(job)

def parse_args():
    parser = argparse.ArgumentParser("Create mosaic detection dataset")
    parser.add_argument('--output-root', type=Path, help="directory where resulting images/masks are saved")
    parser.add_argument('--input-root', type=Path, help="directory containing video files")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--model', type=str, default="model_weights/lada_nsfw_detection_model.pt", help="path to YOLO model")
    parser.add_argument('--workers', type=int, default=4, help="number of worker threads")
    parser.add_argument('--start-index', type=int, default=0, help="Can be used to continue a previous run. Note the index number next to last processed file name")
    parser.add_argument('--sampling', type=int, default=60, help="process only 1 frame every <sampling> seconds instead of every frame in the video")
    parser.add_argument('--show', default=False, action=argparse.BooleanOptionalAction, help="show each sample")
    parser.add_argument('--videos', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--images', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--max-file-limit', type=int, default=None, help="instead of processing all files found in input-root dir it will choose files randomly up to the given limit")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model = YOLO(args.model)

    if args.show:
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        window_name = 'mosaic'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    else:
        os.makedirs(f"{args.output_root}/mask", exist_ok=True)
        os.makedirs(f"{args.output_root}/img", exist_ok=True)
        jobs = []

    selected_files = []
    if args.videos:
        video_files = get_files(args.input_root, video_utils.is_video_file)
        if args.max_file_limit and len(video_files) > args.max_file_limit:
            video_files = random.choices(video_files, k=args.max_file_limit)
        selected_files += video_files
    if args.images:
        image_files = get_files(args.input_root, image_utils.is_image_file)
        if args.max_file_limit and len(image_files) > args.max_file_limit:
            image_files = random.choices(image_files, k=args.max_file_limit)
        selected_files += image_files

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for file_idx, file_path in enumerate(selected_files):
            if file_idx < args.start_index or len(list(args.output_root.glob(f"*/{file_path.name}*"))) > 0:
                print(f"{file_idx}, Skipping {file_path.name}: Already processed")
                continue
            print(f"{file_idx}, Processing {file_path.name}")
            is_video_file = video_utils.is_video_file(file_path)
            if args.show:
                process_video_file(file_path, args.input_root, args.output_root, args.show) if is_video_file else process_image_file(file_path, args.output_root, model, args.device, args.show)
            else:
                if is_video_file:
                    jobs.append(executor.submit(process_video_file, file_path, args.output_root, model, args.device, args.sampling, args.show))
                else:
                    jobs.append(executor.submit(process_image_file, file_path, args.output_root, model, args.device, args.show))
                clean_up_completed_futures(jobs)
    wait(jobs, return_when=ALL_COMPLETED)
    clean_up_completed_futures(jobs)

    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()