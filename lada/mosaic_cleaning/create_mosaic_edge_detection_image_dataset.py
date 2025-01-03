###
# todo: currently broken / outdated
###

import argparse
import glob
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from pathlib import Path

import cv2
import numpy as np

from lada.lib import visualization_utils, degradation_utils
from lada.lib.image_utils import resize, pad_image
from lada.lib.mosaic_utils import addmosaic_base
from lada.lib.ultralytics_utils import disable_ultralytics_telemetry

disable_ultralytics_telemetry()

def read_video_frames(path: str, indices: list[int], mask: bool) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise Exception(f"Unable to open video file: {path}")
    frames = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if i in indices:
                if mask:
                    frame = frame[:, :, 0]  # throw away color channel
                frames.append(frame)
                if len(frames) == len(indices):
                    break
            i += 1
        else:
            break
    cap.release()
    return frames

def mask_to_box(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    box = None
    if len(contours) > 0:
        for contour in contours:
            _box = cv2.boundingRect(contour)
            if box is None:
                box = _box
            else:
                _x, _y, _w, _h = _box
                x, y, w, h = box
                if _w * _h > w * h:
                    box = _box
    # convert to t, l, b, r
    x, y, w, h = box
    t, l, b, r = y, x, y + h, x + w
    return t, l, b, r

def crop_to_box(img, box):
    t, l, b, r = box
    cropped_img = img[t:b + 1, l:r + 1]
    return cropped_img

def draw_grid(mosaic_block_corner_points, canvas_shape, block_size, color=255):
    h, w = canvas_shape[:2]
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    pad = block_size
    canvas_padded = np.pad(canvas,((0,pad),(0,pad),(0,0)), mode='constant', constant_values=0)

    for ((x_start,y_start),(x_end,y_end)) in mosaic_block_corner_points:
        edges = np.expand_dims(np.pad(np.zeros((y_end-y_start-1, x_end-x_start-1), dtype=canvas.dtype), pad_width=1, mode='constant',constant_values=color), axis=-1)
        canvas_padded[y_start:y_end+1, x_start:x_end+1,:] = edges

    canvas = canvas_padded[0:h, 0:w]
    return canvas

def process_clip(clip_name, input_root, output_root, show, frame_count_per_clip=3, window_name="mosaic", out_size = 256):
    img_vid_path = f"{input_root}/crop_unscaled_img/{clip_name}.mp4"
    mask_vid_path = f"{input_root}/crop_unscaled_mask/{clip_name}.mkv"
    meta_path = f"{input_root}/crop_unscaled_meta/{clip_name}.json"
    assert os.path.exists(img_vid_path) and os.path.exists(mask_vid_path) and os.path.exists(meta_path)

    with open(meta_path, 'r') as meta_file:
        meta_json = json.load(meta_file)

    frame_indices = list(sorted(random.sample(list(range(meta_json["frames_count"])), k=frame_count_per_clip)))
    img_frames = read_video_frames(img_vid_path, frame_indices, False)
    mask_frames = read_video_frames(mask_vid_path, frame_indices, True)

    for img_idx in range(frame_count_per_clip):
        img = img_frames[img_idx]
        mask = mask_frames[img_idx]

        box = mask_to_box(mask)
        cropped_img = crop_to_box(img, box)
        cropped_mask = crop_to_box(mask, box)

        cropped_img_mosaic, cropped_mask_mosaic, mosaic_block_corner_points = addmosaic_base(cropped_img,
                                                                 cropped_mask,
                                                                 meta_json["mosaic"]["mosaic_size"],
                                                                 model=meta_json["mosaic"]["mod"],
                                                                 rect_ratio=meta_json["mosaic"]["rect_ratio"],
                                                                 feather=meta_json["mosaic"]["feather_size"],
                                                                 return_mosaic_edges=True)

        degradation_params = degradation_utils.MosaicRandomDegradationParams(should_down_sample=True,
                                                                             should_add_noise=True,
                                                                             should_add_image_compression=True,
                                                                             should_add_video_compression=True,
                                                                             should_add_blur=False)

        img_mosaic = degradation_utils.apply_video_degradation([cropped_img], degradation_params)[0]


        y_unscaled, x_unscaled = img_mosaic.shape[:2]
        img_mosaic = resize(img_mosaic, out_size, interpolation=cv2.INTER_LINEAR)
        y_scaled, x_scaled = img_mosaic.shape[:2]
        y_scale_factor = y_scaled / y_unscaled
        x_scale_factor = x_scaled / x_unscaled
        scaled_mosaic_block_corner_points = [((int(l*x_scale_factor),int(t*y_scale_factor)), (int(r*x_scale_factor),int(b*y_scale_factor))) for (l,t),(r,b) in mosaic_block_corner_points]
        cropped_mosaic_edges = draw_grid(scaled_mosaic_block_corner_points, [y_scaled, x_scaled, 1], int(round(meta_json["mosaic"]["mosaic_size"])))
        cropped_mosaic_edges, _ = pad_image(cropped_mosaic_edges, out_size, out_size)
        img_mosaic, final_mosaic_image_pad = pad_image(img_mosaic, out_size, out_size)

        if show:
            show_img = visualization_utils.overlay_edges(img_mosaic, cropped_mosaic_edges, color=(0, 255, 0))
            cv2.imshow(window_name, show_img)
            while True:
                key_pressed = cv2.waitKey(1)
                if key_pressed & 0xFF == ord("n"):
                    break
        else:
            cv2.imwrite(f"{output_root}/mosaic/{clip_name}-{frame_indices[img_idx]}.jpg", img_mosaic,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite(f"{output_root}/mosaic_edge/{clip_name}-{frame_indices[img_idx]}.png", cropped_mosaic_edges)


def parse_args():
    parser = argparse.ArgumentParser("Create mosaic dataset")
    parser.add_argument('--output-root', type=Path, help="set output root directory")
    parser.add_argument('--input-root', type=Path, help="path to mosaic_removal_vid dataset")
    parser.add_argument('--workers', type=int, default=12, help="number of worker threads")
    parser.add_argument('--show', default=False, action=argparse.BooleanOptionalAction, help="show each sample")
    args = parser.parse_args()
    return args

def main():
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    args = parse_args()

    if args.show:
        window_name = 'mosaic'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    else:
        os.makedirs(f"{args.output_root}/mosaic_edge")
        os.makedirs(f"{args.output_root}/mosaic")
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = []
    clip_names = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{args.input_root}/crop_unscaled_img/*")]


    for file_idx, clip_name in enumerate(clip_names):
        if args.show:
            process_clip(clip_name, args.input_root, args.output_root, args.show)
        else:
            futures.append(executor.submit(process_clip, clip_name, args.input_root, args.output_root, args.show))

    if args.show:
        cv2.destroyAllWindows()
    else:
        wait(futures, return_when=ALL_COMPLETED)

if __name__ == '__main__':
    main()