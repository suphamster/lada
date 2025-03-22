import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from os import path as osp
from pathlib import Path

import cv2
from ultralytics import YOLO

from lada.lib import visualization_utils, image_utils, transforms as lada_transforms
from lada.lib.nsfw_frame_detector import NsfwImageDetector, NsfwFrame
from lada.lib.threading_utils import clean_up_completed_futures
from lada.lib.ultralytics_utils import disable_ultralytics_telemetry

disable_ultralytics_telemetry()

from torchvision.transforms import transforms as torchvision_transforms

from lada.lib.image_utils import UnsharpMaskingSharpener
from lada.lib.jpeg_utils import DiffJPEG

def _create_realesrgan_degradation_pipeline(img, scale, device):
    h, w = img.shape[:2]
    sharpener = UnsharpMaskingSharpener().to(device)
    jpeger = DiffJPEG(differentiable=False).to(device)
    kernel_range = [2 * v + 1 for v in range(3, 11)]
    return torchvision_transforms.Compose([
        lada_transforms.Sharpen(sharpener),
        lada_transforms.Blur(kernel_range=kernel_range, kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'], kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                             sinc_prob=0.1, blur_sigma=[0.2, 2], betag_range=[0.5, 4], betap_range=[1, 2], device=device),
        lada_transforms.Resize(resize_range=[0.75, 1.25], resize_prob=[0.2, 0.7, 0.1], target_base_h=h, target_base_w=w),
        lada_transforms.GaussianPoissonNoise(sigma_range=[1, 5], poisson_scale_range=[0.05, 1.4], gaussian_noise_prob=0.5, gray_noise_prob= 0.4),
        lada_transforms.JPEGCompression(jpeger, jpeg_range=[60, 95]),
        lada_transforms.Blur(kernel_range=kernel_range, kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'], kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                             sinc_prob=0.1, blur_sigma=[0.2, 1.], betag_range=[0.5, 4], betap_range=[1, 2], device=device),
        lada_transforms.Resize(resize_range=[0.85, 1.1], resize_prob= [0.3, 0.4, 0.3], target_base_h=h / scale, target_base_w=w / scale),
        lada_transforms.GaussianPoissonNoise(sigma_range=[1, 5], poisson_scale_range=[0.05, 1.], gaussian_noise_prob=0.5, gray_noise_prob=0.3),
        torchvision_transforms.RandomChoice(transforms=[
            torchvision_transforms.Compose([
                lada_transforms.Resize(resize_range=[1., 1.], resize_prob=[0, 0, 1], target_base_h=h / scale, target_base_w=w / scale),
                lada_transforms.SincFilter(kernel_range=kernel_range, sinc_prob=0.8, device=device),
                lada_transforms.JPEGCompression(jpeger, jpeg_range=[60, 95]),
            ]),
            torchvision_transforms.Compose([
                lada_transforms.JPEGCompression(jpeger, jpeg_range=[60, 95]),
                lada_transforms.Resize(resize_range=[1., 1.], resize_prob=[0, 0, 1], target_base_h=h / scale, target_base_w=w / scale),
                lada_transforms.SincFilter(kernel_range=kernel_range, sinc_prob=0.8, device=device),
            ])
        ], p=[0.5, 0.5]),
    ])

def create_degradation_pipeline(hq_img, scale=2, device='cuda'):
    return torchvision_transforms.Compose([
        lada_transforms.Image2Tensor(bgr2rgb=False, unsqueeze=True, device=device),
        _create_realesrgan_degradation_pipeline(hq_img, scale=scale, device=device),
        lada_transforms.Tensor2Image(rgb2bgr=False, squeeze=True),
        lada_transforms.VideoCompression(p=0.3, codecs=['libx264', 'libx265'], codec_probs=[0.5, 0.5],
                                         crf_ranges={'libx264': (16, 28), 'libx265': (20, 36)},
                                         bitrate_ranges={}),
    ])

def process_image_file(file_path, output_root, nsfw_frame_generator: NsfwImageDetector, device='cpu', show=False, window_name="mosaic"):
    nsfw_frame: NsfwFrame = nsfw_frame_generator.detect(file_path)
    if not nsfw_frame:
        return

    img = nsfw_frame.frame
    mask = nsfw_frame.mask

    scale = 2
    degrade = create_degradation_pipeline(img, scale=scale, device=device)

    img_mosaic, mask_mosaic = lada_transforms.Mosaic()(img, mask)
    img_mosaic = degrade(img_mosaic)
    mask_mosaic = image_utils.resize(mask_mosaic, img_mosaic.shape[:2], interpolation=cv2.INTER_NEAREST)

    if show:
        show_img = visualization_utils.overlay_mask_boundary(img_mosaic, mask_mosaic, color=(0, 255, 0))
        mask = image_utils.resize(mask, img_mosaic.shape[:2], interpolation=cv2.INTER_NEAREST)
        show_img = visualization_utils.overlay_mask_boundary(show_img, mask, color=(255, 0, 0))

        cv2.imshow(window_name, show_img)

        while True:
            key_pressed = cv2.waitKey(1)
            if key_pressed & 0xFF == ord("n"):
                break
    else:
        name = osp.splitext(os.path.basename(file_path))[0]
        cv2.imwrite(f"{output_root}/img/{name}.jpg", img_mosaic,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(f"{output_root}/mask/{name}.png", mask_mosaic)

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
    parser.add_argument('--input-root', type=Path, help="directory containing image files")
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
    nsfw_image_detector = NsfwImageDetector(model, args.device, random_extend_masks=True, conf=0.75)

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
                process_image_file(file_path, args.output_root, nsfw_image_detector, device=args.device, show=True)
            else:
                jobs.append(executor.submit(process_image_file, file_path, args.output_root, nsfw_image_detector, args.device))
                clean_up_completed_futures(jobs)
    wait(jobs, return_when=ALL_COMPLETED)
    clean_up_completed_futures(jobs)

    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()