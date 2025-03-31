import cv2
import numpy as np
import os
from ultralytics import YOLO
import argparse
import hashlib


# The function to be called anytime a slider's value changes
def update():
    global videomode
    global frame
    if videomode:
        vid_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        res, _frame = vid_capture.read()
        if res: frame =_frame

    result = net.predict(frame, conf=args.mask_threshold, imgsz=640)
    output = result[0].plot()
    cv2.imshow(window_name, output)


def update_frame(f):
    global frame_num
    frame_num = f
    update()


def update_thresh(t):
    args.mask_threshold = t / 100.
    update()


def screenshot(dir):
    global frame_num
    global frame
    if dir is None:
        return
    os.makedirs(dir, exist_ok=True)

    file_path = os.path.join(dir, f"{hashlib.sha256(frame).hexdigest()}-{frame_num}.jpg")
    print("Saved screenshot:", file_path)
    cv2.imwrite(file_path, frame)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--mask-threshold', type=float, default=0.4)
    parser.add_argument('--model-path', type=str,
                        default='yolo/runs/segment/train_yolov9_c_sgd_5_more_training_data/weights/best.pt')
    parser.add_argument('--screenshot-dir', type=str, default=None)
    args = parser.parse_args()
    return args


args = parse_args()
net = YOLO(args.model_path)

videomode = True
frame_num = 0
window_name = 'img'

try:
    frame = cv2.imread(args.input)
    if frame is None:
        # input is a video file
        vid_capture = cv2.VideoCapture(args.input)
        if not vid_capture.isOpened():
            raise Exception("Unable to read from input file")
        res, frame = vid_capture.read()
    else:
        # input is an image file
        videomode = False

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    update()
    if videomode:
        frame_count = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.createTrackbar('frame', window_name, 0, frame_count, update_frame)
    cv2.createTrackbar('thresh', window_name, int(args.mask_threshold * 100), 100, update_thresh)

    while True:
        key_pressed = cv2.waitKey(1)
        if key_pressed & 0xFF == ord("q"):
            print("q")
            break
        elif key_pressed & 0xFF == ord("s"):
            print("s %s" % frame_num)
            screenshot(args.screenshot_dir)

except KeyboardInterrupt:
    print("Closed with Ctrl+C")

except Exception as err:
    raise err

finally:
    # Clean up resources
    cv2.destroyAllWindows()
    if videomode:
        vid_capture.release()

# QT_QPA_PLATFORM=xcb python view-yolo.py --input <a video file> --model-path yolo/runs/segment/train/weights/best.pt
