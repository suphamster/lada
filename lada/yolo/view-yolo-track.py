from ultralytics import YOLO
import cv2
import sys
import argparse

from ultralytics import settings
# Disable analytics and crash reporting
settings.update({'sync': False})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--model-path', type=str, default='yolo/runs/old_run/weights/best.pt')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = YOLO(args.model_path)
    results = model.track(source=args.input, stream=True, verbose=False, tracker="bytetrack.yaml")
    for frame_num, result in enumerate(results):
        annotated_frame = result.plot()
        cv2.imshow("yolo", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    main()

# QT_QPA_PLATFORM=xcb python view-yolo-track.py --input <some video file>