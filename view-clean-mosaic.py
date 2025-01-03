import argparse
import os
from time import sleep

import cv2
import numpy as np
from ultralytics import YOLO

from lada.lib.clean_mosaic_utils import clean_cropped_mosaic
from lada.lib.mosaic_detector import MosaicDetectorDeprecated
from lada.lib.ultralytics_utils import disable_ultralytics_telemetry
from lada.lib.video_utils import get_video_meta_data, VideoWriter
from lada.pidinet import pidinet_inference

disable_ultralytics_telemetry()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--mosaic-detection-model-path', type=str,
                        default='yolo/runs/segment/train_mosaic_detection_yolov9c/weights/best.pt')
    parser.add_argument('--max-clip-length', type=int, default=180)
    parser.add_argument('--clip-size', type=int, default=256)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    mosaic_detection_model = YOLO(args.mosaic_detection_model_path)
    # pidinet_model = pidinet_inference.load_model("experiments/pidinet/run1/save_models/checkpoint_019.pth", model_type="base", device=args.device)
    # pidinet_model = pidinet_inference.load_model("experiments/pidinet/run2_tiny/save_models/checkpoint_019.pth", model_type="tiny", device=args.device)
    pidinet_model = None

    mosaic_generator = MosaicDetectorDeprecated(mosaic_detection_model, args.input, args.max_clip_length, args.clip_size, pad_mode='zero', preserve_relative_scale=True, dont_preserve_relative_scale=False, device=args.device)

    if args.output:
        video_metadata = get_video_meta_data(args.input)
        video_writer = VideoWriter(args.output, args.clip_size*3, args.clip_size, video_metadata.video_fps, crf=14)
    else:
        for window in ( 'orig', 'clean', 'debug'):
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            sleep(0.0001)

    quit = False
    for clip_idx, clip in enumerate(mosaic_generator()):
        images = []
        debug_images = []
        orig_images = clip.get_clip_images()
        for i, (cropped_img, cropped_mask, cropped_box, orig_crop_shape, pad) in enumerate(clip):
            cleaned_image, debug_image = clean_cropped_mosaic(cropped_img, cropped_mask, pad, draw=True, pidinet_model=pidinet_model)
            images.append(cleaned_image)
            debug_images.append(debug_image)
        for i, (image_clean, image_orig, image_debug) in enumerate(zip(images, orig_images, debug_images)):
            print(f"clip: {clip_idx:02d}, frame: {i:04d}")

            if args.output:
                frame = np.concatenate((image_orig, image_clean, image_debug), axis=1)
                video_writer.write(frame, bgr2rgb=True)
            else:
                cv2.imshow('orig', image_orig)
                cv2.imshow('clean', image_clean)
                cv2.imshow('debug', image_debug)

            if not args.output and cv2.waitKey(0) & 0xFF == ord("q"):
                quit = True
                break
        if quit:
            break
    if args.output:
        video_writer.release()
    else:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    main()
