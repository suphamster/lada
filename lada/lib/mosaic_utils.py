import math
import random

import cv2
import numpy as np

from lada.lib import image_utils
from lada.lib import visualization


def get_mask_area(mask):
    mask = cv2.threshold(mask,127,255,0)[1]
    contours= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    try:
        area = cv2.contourArea(contours[0])
    except:
        area = 0
    return area

def _mosaic_get_block_data_fun(model):
    if model=='squa_avg':
        return lambda img, i, j, n_h, n_w, h_start, w_start: img[i*n_h+h_start:(i+1)*n_h+h_start,j*n_w+w_start:(j+1)*n_w+w_start,:].mean(axis=(0,1))
    elif model=='squa_mid':
        return lambda img, i, j, n_h, n_w, h_start, w_start: img[i*n_h+n_h//2+h_start,j*n_w+n_w//2+w_start,:]
    elif model == 'squa_random':
        return lambda img, i, j, n_h, n_w, h_start, w_start: img[h_start+int(i*n_h-n_h/2+n_h*random.random()),w_start+int(j*n_w-n_w/2+n_w*random.random()),:]
    elif model =='rect_avg':
        return lambda img, i, j, n_h, n_w, h_start, w_start: img[i*n_h+h_start:(i+1)*n_h+h_start,j*n_w+w_start:(j+1)*n_w+w_start,:].mean(axis=(0,1))
    raise Exception()


def addmosaic_base(img, mask, n, model='squa_avg', rect_ratio=1.6, feather=0, return_mosaic_edges=False):
    '''
    img: input image
    mask: input mask
    n: mosaic size
    model : squa_avg squa_mid squa_random squa_avg_circle_edge rect_avg
    rect_ratio: if model==rect_avg , mosaic w/h=rect_ratio
    feather : feather size, -1->no 0->auto
    '''
    n = int(n)
    rect_ratio = 1.0 if model != 'rect_avg' else rect_ratio
    n_h = n
    n_w = int(n * rect_ratio)
    h_start = 0
    w_start = 0
    pix_mid_h = n_h // 2 + h_start
    pix_mid_w = n_w // 2 + w_start
    h, w = img.shape[:2]
    h_step = math.ceil((h - h_start) / n_h)
    w_step = math.ceil((w - w_start) / n_w)
    assert img.shape[:2] == mask.shape[:2]
    get_block_data = _mosaic_get_block_data_fun(model)
    pad = n
    img_padded = np.pad(img,((0,pad),(0,pad),(0,0)), mode='reflect')
    mask_padded = np.pad(mask,((0,pad),(0,pad),(0,0)), mode='constant', constant_values=0)
    img_mosaic = img.copy()
    mask_mosaic = np.zeros_like(mask, dtype=mask.dtype)
    mask_val = 255

    block_corner_points = []
    for i in range(h_step):
        for j in range(w_step):
            if mask_padded[i * n_h + pix_mid_h, j * n_w + pix_mid_w]:
                y_start = i * n_h + h_start
                y_end = (i + 1) * n_h + h_start
                x_start = j * n_w + w_start
                x_end = (j + 1) * n_w + w_start
                img_mosaic[y_start:y_end, x_start:x_end,:] = get_block_data(img_padded, i, j, n_h, n_w, h_start, w_start)
                mask_mosaic[y_start:y_end, x_start:x_end,:] = mask_val
                block_corner_points.append(((x_start,y_start),(x_end,y_end)))

    if feather != -1:
        if feather == 0:
            blurred_mask = (cv2.blur(mask, (n, n)))
        else:
            blurred_mask = (cv2.blur(mask, (feather, feather)))
        blurred_mask = blurred_mask / 255.0
        for i in range(3):
            img_mosaic[:, :, i] = (img[:, :, i] * (1 - blurred_mask) + img_mosaic[:, :, i] * blurred_mask)
        img_mosaic = img_mosaic.astype(np.uint8)

    return (img_mosaic, mask_mosaic, block_corner_points) if return_mosaic_edges else (img_mosaic, mask_mosaic)

def get_mosaic_block_size(mask_img,area_type = 'normal'):
    h,w = mask_img.shape[:2]
    size = np.min([h,w])
    mask = image_utils.resize_simple(mask_img,size)
    alpha = size/512
    try:
        if area_type == 'normal':
            area = get_mask_area(mask)
        elif area_type == 'bounding':
            w,h = cv2.boundingRect(mask)[2:]
            area = w*h
    except:
        area = 0
    area = area/(alpha*alpha)
    if area>50000:
        size = alpha*((area-50000)/50000+12)
    elif 20000<area<=50000:
        size = alpha*((area-20000)/30000+8)
    elif 5000<area<=20000:
        size = alpha*((area-5000)/20000+7)
    elif 1000<area<=5000:
        size = alpha*((area-1000)/5000+6)
    elif 0<=area<=1000:
        size = alpha*((area-0)/1000+5)
    else:
        pass
    return size

def get_mosaic_block_size_v2(mask):
    # assumption: if mosaic area covers a huge area its a close-up and we need bigger mosaic block size.
    # If the area is small in comparison its far away and lesser block size is necessary
    # there is a minimal block size and also a maximum
    h,w = mask.shape[:2]
    #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    _, _, box_w, box_h = cv2.boundingRect(contours[0])
    #mosaic_area = get_mask_area(mask)
    #full_area = h * w
    mosaic_area = max(box_w, box_h)
    full_area = max(h,w)
    ratio_mosaic_area_covered = mosaic_area / full_area
    block_sizes = np.linspace(0.01, 0.018, num=20)
    area_cover_ratios = np.linspace(0, 1, num=20)
    idx = (np.abs(area_cover_ratios - ratio_mosaic_area_covered)).argmin()
    block_size_normalized = block_sizes[idx]
    block_size_pixel = block_size_normalized * max(w, h)
    return block_size_pixel

def get_random_parameter(mask, randomize_size=True, v2=False):
    # mosaic size
    p = np.array([0.5,0.5])
    mod = np.random.choice(['normal','bounding'], p = p.ravel())
    if v2:
        mosaic_size = get_mosaic_block_size_v2(mask)
    else:
        mosaic_size = get_mosaic_block_size(mask,area_type = mod)
    if randomize_size:
        mosaic_size = int(mosaic_size*random.uniform(0.9,2.2))

    # mosaic mod
    p = np.array([0.25, 0.3, 0.45])
    mod = np.random.choice(['squa_mid','squa_avg','rect_avg'], p = p.ravel())

    # rect_rat for rect_avg
    rect_rat = random.uniform(1.1,1.6)
    
    # feather size
    feather = -1
    if random.random()<0.7:
        feather = int(mosaic_size*random.uniform(0,1.5))

    return mosaic_size,mod,rect_rat,feather


if __name__ == '__main__':
    window_name = 'mosaic'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    rng = np.random.default_rng()
    window_size_h = 700
    window_size_w = 1000
    img = rng.integers(0, 255, (window_size_h,window_size_w,3), dtype=np.uint8)
    global mask_img
    mask_img = np.zeros((window_size_h,window_size_w,1), dtype=np.uint8)

    global box_w
    global box_h
    box_l = int(window_size_w * 0.1)
    box_t = int(window_size_h * 0.1)
    box_w = int(window_size_w * 0.2)
    box_h = int(window_size_h * 0.2)

    def update_box_h(new_box_h):
        global mask_img
        global box_h
        if new_box_h < 1:
            new_box_h = 1
        box_h = new_box_h
        mask_img = np.zeros_like(mask_img, dtype=mask_img.dtype)
        mask_img[box_t:box_t+box_h, box_l:box_l+box_w] = 255
        cv2.imshow(window_name, create_mosaic_img())

    def update_box_w(new_box_w):
        global mask_img
        global box_w
        if new_box_w < 1:
            new_box_w = 1
        box_w = new_box_w
        mask_img = np.zeros_like(mask_img, dtype=mask_img.dtype)
        mask_img[box_t:box_t+box_h, box_l:box_l+box_w] = 255
        cv2.imshow(window_name, create_mosaic_img())

    def create_mosaic_img():
        mosaic_size, mod, rect_ratio, feather_size = get_random_parameter(mask_img, randomize_size=False, v2=False)
        mosaic_size, mod, rect_ratio, feather_size = mosaic_size, 'squa_mid', 1.5, int(mosaic_size*1.5)
        mosaic_img, mosaic_mask_img = addmosaic_base(img, mask_img, mosaic_size, model=mod, rect_ratio=rect_ratio,
                                                     feather=feather_size)
        mosaic_img = visualization.overlay_mask(mosaic_img, mask_img)
        return mosaic_img

    cv2.createTrackbar('box w', window_name, box_w, img.shape[1], update_box_w)
    cv2.createTrackbar('box h', window_name, box_h, img.shape[0], update_box_h)

    mask_img[box_t:box_t+box_h, box_l:box_l+box_w] = 255

    cv2.imshow(window_name, create_mosaic_img())

    while True:
        key_pressed = cv2.waitKey(1)
        if key_pressed & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        elif key_pressed & 0xFF == ord("r"):
            cv2.imshow(window_name, create_mosaic_img())