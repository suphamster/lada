import os.path

import cv2
from pathlib import Path
from ultralytics import YOLO
import argparse

from lada.lib.ultralytics_utils import disable_ultralytics_telemetry
from lada.lib.video_utils import process_video_v3

disable_ultralytics_telemetry()


def process_frame(in_frame, model, threshold, classes, negate=False):
    result = model.predict(in_frame, conf=threshold, imgsz=640, verbose=False, classes=classes)
    processed_frame = result[0].plot()

    return processed_frame

def process_image(input_path, output_path, model, threshold, classes, negate=False):
    in_frame = cv2.imread(input_path)
    result = model.predict(in_frame, conf=threshold, imgsz=640, verbose=False, classes=classes)
    detected = len(result[0].boxes) > 0
    if detected ^ negate:
        processed_frame = result[0].plot()
        cv2.imwrite(output_path, processed_frame)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--class-id', type=int, default=0)
    parser.add_argument('--negate', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--model-path', type=str,
                        default='yolo/runs/segment/train_yolov9_c_sgd_5_more_training_data/weights/best.pt')
    args = parser.parse_args()
    return args

def process_file(input_path: str, output_dir: str):
    output_path = str(Path(output_dir).joinpath(Path(input_path).name))

    frame = cv2.imread(input_path)
    if frame is None:
        process_video_v3(input_path, output_path, lambda in_frame: process_frame(in_frame, model, threshold=args.threshold, classes=[args.class_id], negate=args.negate))
    else:
        # input is an image file
        process_image(input_path, output_path, model, threshold=args.threshold, classes=[args.class_id], negate=args.negate)

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
