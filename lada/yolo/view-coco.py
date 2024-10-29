import os
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--images-dir', type=str, help='directory of images')
    parser.add_argument('--coco-file', type=str, help='path to COCO JSON file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
 
    coco = COCO(args.coco_file)
    catIds = coco.getCatIds(catNms=['0'])  # category ids, e.g., catNms=['0', '1', ...] which is in accordance with the CATEGORIES defined in main.py
    imgIds = coco.getImgIds(catIds=catIds )
    for i in range(len(imgIds)):
        img = coco.loadImgs(imgIds[i])[0]
        I = cv2.imread(os.path.join(args.images_dir, img['file_name']))
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(I)
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        #if len(anns) < 2:
        #    continue
        coco.showAnns(anns, draw_bbox=True)
        for i, ann in enumerate(anns):
            plt.text(anns[i]['bbox'][0], anns[i]['bbox'][1], anns[i]['category_id'], style='italic',
            bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 5})
        plt.show() 


if __name__ == "__main__":
    main()
