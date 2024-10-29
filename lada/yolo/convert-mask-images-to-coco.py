import numpy as np
import os
import glob
from tqdm import tqdm
import cv2
import json
import argparse
import datetime
 
INFO = {
    "description": "nsfw Dataset",
}
 
LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]
 
CATEGORIES = [
    {
        'id': 1,
        'name': 'nsfw',
        'supercategory': 'nsfw',
        'color': [255, 255, 255]  # the color used to mask the object
    }
]

def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[1],
            "height": image_size[0],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info


def create_annotation_infos(annotation_id, image_id, category_info, binary_mask):
    
    is_crowd = category_info['is_crowd']
    annotation_infos = []
    
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = (np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0) * 255).astype('uint8')
    contours, _ = cv2.findContours(padded_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    #print(contours)
    #contours = np.subtract(contours, 1)
    
    for i, contour in enumerate(contours):
        contour = np.subtract(contour, 1)
        if len(contour) < 3:  # filter unenclosed objects
            continue
    
        x, y, w, h = cv2.boundingRect(contour)   
        bounding_box = [x, y, w, h]
        seg_area = cv2.contourArea(contour)
        bbox_area = w * h
        
        #if bbox_area < int(binary_mask.shape[0] * binary_mask.shape[1] * 0.001):  # filter small objects
        if bbox_area < 4:  # filter small objects
            continue

        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]

        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_info["id"],
            "iscrowd": is_crowd,
            "area": seg_area,  # it's float
            "bbox": bounding_box,
            "segmentation": [segmentation],
            "width": binary_mask.shape[1],
            "height": binary_mask.shape[0],
        } 
        
        annotation_id += 1
        
        annotation_infos.append(annotation_info)

    return annotation_infos, annotation_id



def parse_args():
    
    parser = argparse.ArgumentParser(description='args')

    parser.add_argument('--single-mask-dir',
                        help='choose to generate a segmentation mask for each category in an image',
                        type=str,
                        default=None
                        )
    parser.add_argument('--images-dir', type=str, help='directory of images')
    parser.add_argument('--masks-dir', type=str, help='directory of image masks')
    parser.add_argument('--output-file', type=str, help='json result path')
    args = parser.parse_args()
    return args
 
def main():
    
    args = parse_args()
 
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    
    # initial ids
    image_id = 1
    segmentation_id = 1
   
    # find image and mask paths
    image_files = glob.glob(args.images_dir + '/*.jpg') + glob.glob(args.images_dir + '/*.png')
    mask_files = glob.glob(args.masks_dir + '/*.png')
 
    # go through each image
    for image_filename in tqdm(image_files):
        image = cv2.imread(image_filename)

        # skip the image without label file
        base_name = os.path.basename(image_filename)
        mask_name = os.path.join(args.masks_dir, os.path.splitext(base_name)[0] + '.png')
        if mask_name not in mask_files:
            print(f"skipping image because mask file is missing {mask_name}")
            continue
        
        mask = cv2.imread(mask_name)  # bgr mask 
        
        image_info = create_image_info(
            image_id, os.path.basename(image_filename), image.shape)
        coco_output["images"].append(image_info)        
        
        # go through each existing category
        for category_dict in CATEGORIES:
            color = category_dict['color']
            class_id = category_dict['id']
            class_name = category_dict['name']
            category_info = {'id': class_id, 'is_crowd': 0}  # do not support the crowded type
            
            binary_mask = np.all(mask == color, axis=-1).astype('uint8')  # quick search
            
            if args.single_mask_dir is not None:
                cv2.imwrite(os.path.join(args.single_mask_dir, os.path.splitext(base_name)[0] + '_' + class_name + '.png'), binary_mask * 255)            
            
            annotation_info, annotation_id = create_annotation_infos(
                segmentation_id, image_id, category_info, binary_mask)

            coco_output["annotations"].extend(annotation_info)

            segmentation_id = annotation_id

        image_id = image_id + 1
 
    with open(args.output_file, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
 
 
if __name__ == "__main__":
    main()

# python yolo/convert-mask-images-to-coco.py --images-dir datasets/nsfw_detection/train/img --masks-dir  datasets/nsfw_detection/train/img_mask --output-file datasets/nsfw_detection/train/coco.json
# python yolo/convert-mask-images-to-coco.py --images-dir datasets/nsfw_detection/val/img --masks-dir  datasets/nsfw_detection/val/img_mask --output-file datasets/nsfw_detection/val/coco.json
