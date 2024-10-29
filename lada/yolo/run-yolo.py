import os.path

import cv2
from pathlib import Path
from ultralytics import YOLO
import argparse

from lada.lib.video_utils import process_video_v3
from ultralytics import settings
# Disable analytics and crash reporting
settings.update({'sync': False})


def process_frame(in_frame, model):
    result = model.predict(in_frame, conf=args.mask_threshold, imgsz=640, verbose=False)
    processed_frame = result[0].plot()

    return processed_frame

def process_image(input_path, output_path, model):
    in_frame = cv2.imread(input_path)
    result = model.predict(in_frame, conf=args.mask_threshold, imgsz=640, verbose=False)
    processed_frame = result[0].plot()
    cv2.imwrite(output_path, processed_frame)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--mask-threshold', type=float, default=0.4)
    parser.add_argument('--model-path', type=str,
                        default='yolo/runs/segment/train_yolov9_c_sgd_5_more_training_data/weights/best.pt')
    args = parser.parse_args()
    return args

def process_file(input_path: str, output_dir: str):
    output_path = str(Path(output_dir).joinpath(Path(input_path).name))

    frame = cv2.imread(input_path)
    if frame is None:
        process_video_v3(input_path, output_path, lambda in_frame: process_frame(in_frame, model))
    else:
        # input is an image file
        process_image(input_path, output_path, model)

args = parse_args()
model = YOLO(args.model_path)

input_path = Path(args.input)
if input_path.is_file():
    process_file(args.input, args.output_dir)
elif input_path.is_dir():
    for file_index, dir_entry in enumerate(input_path.iterdir()):
        if dir_entry.is_file() and os.path.splitext(str(dir_entry))[1] != '.json':
            print(f"{file_index}, Processing {Path(dir_entry).name}")
            process_file(str(dir_entry), args.output_dir)


# python run-yolo.py --input testvid.mp4 --output-dir testout --model-path yolo/runs/segment/train_yolov9_c_sgd_5_more_training_data/weights/best.pt
