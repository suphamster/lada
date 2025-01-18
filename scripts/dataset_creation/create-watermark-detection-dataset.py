"""
source: https://github.com/tgenlis83/dnn-watermark
"""
import argparse

import glob
import math
import os
import random
from logging import getLogger
from typing import List
from tqdm import tqdm

from PIL import Image

from lada.lib.watermark_creation_utils import add_logo_watermark, add_text_watermark, load_fonts, convert_to_yolo

logger = getLogger(__name__)

def create_dataset(
    image_directory: str, logo_directory: str, yolo_labels_path: str, yolo_images_path: str, dataset_min_size: int
) -> None:
    font_paths = load_fonts()
    logo_pathes: List[str] = os.listdir(logo_directory)
    images_paths: List[str] = glob.glob(os.path.join(image_directory, '*'))
    image_multiplier = math.ceil(dataset_min_size / len(images_paths)) if len(images_paths) < dataset_min_size else 1
    images_paths = (images_paths * image_multiplier)[:dataset_min_size]
    random.shuffle(logo_pathes)

    for idx, image_path in enumerate(tqdm(images_paths, desc="Generating watermarks")):
        font = font_paths[idx % len(font_paths)]
        logo_path: str = logo_directory + "/" + logo_pathes[idx % len(logo_pathes)]

        try:
            if idx % 2 == 0:
                image_pil: Image = Image.open(image_path)
                logo_pil: Image = Image.open(logo_path)
                watermarked_image, bbox, category = add_logo_watermark(image_pil, logo_pil)
            else:
                image_pil: Image = Image.open(image_path)
                watermarked_image, bbox, category = add_text_watermark(image_pil, font)
        except Exception as e:
            logger.warning(e)
            continue

        if bbox is None:
            # logger.warning("Empty bbox in image %s", image_properties["file_name"])
            continue

        convert_to_yolo(
            file_name=os.path.basename(image_path),
            bbox=bbox,
            category_id=category,
            yolo_labels_path=yolo_labels_path,
            yolo_images_path=yolo_images_path,
            watermarked_image=watermarked_image,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-images-dir', type=str)
    parser.add_argument('--val-images-dir', type=str)
    parser.add_argument('--logos-dir', type=str)
    parser.add_argument('--yolo-dir', type=str, default='datasets/watermark_detection')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    create_dataset(
        image_directory=args.train_images_dir,
        logo_directory=args.logos_dir,
        yolo_labels_path=os.path.join(args.yolo_dir, "train", "labels"),
        yolo_images_path=os.path.join(args.yolo_dir, "train", "images"),
        dataset_min_size=16_000
    )

    create_dataset(
        image_directory=args.val_images_dir,
        logo_directory=args.logos_dir,
        yolo_labels_path=os.path.join(args.yolo_dir, "val", "labels"),
        yolo_images_path=os.path.join(args.yolo_dir, "val", "images"),
        dataset_min_size=800
    )

if __name__ == "__main__":
    main()

