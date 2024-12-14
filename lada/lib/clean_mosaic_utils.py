import itertools
import math
import os
import time
from time import sleep

import cv2 as cv
import numpy as np

from lada.lib import visualization
from lada.pidinet import pidinet_inference


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    if amount == 0:
        return image
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def sharp(imgage):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv.filter2D(imgage, -1, kernel)

def get_grid_via_houghp(edges, img=None, draw=False, hough_min_length=25):
    grid_x = []
    grid_y = []
    if draw:
        houghp=np.copy(img)
    lines = cv.HoughLinesP(image=edges,rho=1,theta=np.pi/90, threshold=50,lines=np.array([]), minLineLength=hough_min_length,maxLineGap=5)
    if lines is None or len(lines) == 0:
        print('no lines detected via houghlinesp!')
        return (houghp, grid_x, grid_y) if draw else (grid_x, grid_y)
    delta_pix=2
    for line in lines:
        x1,y1,x2,y2 = line[0]
        if not (abs(x1-x2) < delta_pix or abs(y1-y2) < delta_pix):
            continue
        if abs(x1-x2) < delta_pix:
            grid_x.append(int(round(x1)))
        else:
            grid_y.append(int(round(y1)))
        if draw:
            cv.line(houghp,(x1,y1),(x2,y2),(0,255,0),1)
    return (houghp, grid_x, grid_y) if draw else (grid_x, grid_y)

def get_grid_via_hough(edges, img=None, draw=False, hough_min_length=25):
    grid_x = []
    grid_y = []
    if draw:
        hough = np.copy(img)
        h, w = img.shape[:2]
    lines = cv.HoughLines(edges, 1, np.pi / 90, hough_min_length)
    if lines is None or len(lines) == 0:
        print('no lines detected via houghlines!')
        return (hough, grid_x, grid_y) if draw else (grid_x, grid_y)
    for line in lines:
        rho, theta = line[0]
        deg = (theta * 180) / np.pi
        if not (89.5 <= deg <= 90.5 or -0.5 <= deg <= 0.5):
            continue
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        if x0 > 0:
            grid_x.append(int(round(x0)))
        else:
            grid_y.append(int(round(y0)))
        if draw:
            if x0 > 0:
                x0 = int(round(x0))
                x1, x2 = x0, x0
                y1, y2 = 0, h-1
            else:
                y0 = int(round(y0))
                x1, x2 = 0, w-1
                y1, y2 = y0, y0
            cv.line(hough, (x1, y1), (x2, y2), (0, 0, 255), 1)
    if len(grid_x) == 0:
        print('no vertical lines detected via houghlines!')
    if len(grid_y) == 0:
        print('no horizontal lines detected via houghlines!')
    return (hough, grid_x, grid_y) if draw else (grid_x, grid_y)

def get_horizontal(edges, horizontal_size = 9): # 11
    edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    edges = cv.erode(edges, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    horizontal = np.copy(edges)
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    return horizontal

def get_vertical(edges, vertical_size = 9): # 11
    edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    edges = cv.erode(edges, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    vertical = np.copy(edges)
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)
    return vertical


def get_cleaned_mosaic_raw(img, img_unpad, mask_unpad, pad, grid_x_clean, grid_y_clean):
    (pad_h_t, pad_h_b, pad_w_l, pad_w_r) = pad
    img_clean = np.copy(img)
    for x1, x2 in itertools.pairwise(grid_x_clean):
        diff_x = x2 - x1
        x_mid = x1 + diff_x // 2
        for y1, y2 in itertools.pairwise(grid_y_clean):
            diff_y = y2 - y1
            y_mid = y1 + diff_y // 2
            if mask_unpad[y_mid, x_mid] > 0:
                b_max = 3 # as we're probably not perfectly aligned lets not consider border pixels of a block when calculating the mean
                b_x = min(2*b_max, diff_x-1)//2
                b_y = min(2*b_max, diff_y-1)//2
                mosaic_block = np.mean(img_unpad[y1+b_y:y2-b_y, x1+b_x:x2-b_x], axis=(0,1))
                overlap = 0
                img_clean[y1+pad_h_t-overlap:y2+pad_h_t+overlap, x1+pad_w_l-overlap:x2+pad_w_l+overlap, :] = mosaic_block
    return img_clean

def draw_grid(h, w, stepsize_x, stepsize_y, offset_x=0, offset_y=0,color=255, thickness=1):
    edges = np.zeros((h, w), dtype=np.uint8)

    cols = math.ceil((w-offset_x) / stepsize_x)
    rows = math.ceil((h-offset_y) / stepsize_y)

    for col in range(cols):
        x = offset_x + (col * stepsize_x)
        cv.line(edges, (x, 0), (x, h), color=color, thickness=thickness)

    for row in range(rows):
        y = offset_y + (row * stepsize_y)
        cv.line(edges, (0, y), (w, y), color=color, thickness=thickness)

    return edges


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def get_corners(edges, draw=False):
    harris = cv.cornerHarris(np.float32(edges), 2, 3, 0.04)

    corner_coords_filter = harris > 0.01 * harris.max()
    corner_coords_y, corner_coords_x = corner_coords_filter.nonzero()


    # _harris = cv.dilate(_harris, None)
    corners = np.zeros_like(edges, dtype=np.uint8)
    corners[corner_coords_filter] = 255

    # # find centroids
    # ret, labels, stats, centroids = cv.connectedComponentsWithStats(corners)
    #
    # # define the criteria to stop and refine the corners
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # corner_coords_subpix = cv.cornerSubPix(edges, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # corner_coords_y, corner_coords_x = tuple(np.swapaxes(corner_coords_subpix.round(), 0, 1).astype(int).tolist())

    corner_coords_list = list(zip(corner_coords_x, corner_coords_y))

    if draw:
        return corners, corner_coords_list
    else:
        return corner_coords_list


def gaussian_weights(window_ext, sigma=1):
    # source: https://scikit-image.org/docs/stable/auto_examples/transform/plot_matching.html
    y, x = np.mgrid[-window_ext : window_ext + 1, -window_ext : window_ext + 1]
    g = np.zeros(y.shape, dtype=np.double)
    g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
    g /= 2 * np.pi * sigma * sigma
    return g

def get_mse_grid(grid, grid_clean):
    squared_errors=[]
    for grid_val in grid:
        closest_match = min(grid_clean, key=lambda grid_clean_val: abs(grid_clean_val-grid_val))
        squared_errors.append((grid_val-closest_match)**2)
    mean_squared_error = sum(squared_errors) / len(grid)
    return mean_squared_error

def get_clean_grid_v2(grid_x, grid_y, img, mask, draw=False):
    min_step_size = 4
    max_step_size = 50

    if len(grid_x) == 0 or len(grid_y) == 0:
        return (img, [], []) if draw else ([], [])

    grid_x=sorted(grid_x)
    grid_y=sorted(grid_y)

    h, w = img.shape[:2]
    best_grid_params = dict()

    start = time.perf_counter()
    lowest_err = None
    for x_stepsize in range(min_step_size,max_step_size):
        for x_offset in range(x_stepsize):
            cols = math.ceil((w - x_offset) / x_stepsize)
            grid_x_clean = []
            for col in range(cols):
                x = x_offset + (col * x_stepsize)
                grid_x_clean.append(x)

            err = get_mse_grid(grid_x, grid_x_clean) + cols
            if lowest_err is None or err < lowest_err:
                #print(f"err: {err}, grid_x: {grid_x}, grid_x_clean: {grid_x_clean}, cols: {cols}, lowest_err: {lowest_err}, x_stepsize: {x_stepsize}, x_offset: {x_offset}")
                lowest_err = err
                best_grid_params["x_stepsize"] = x_stepsize
                best_grid_params["x_offset"] = x_offset

    lowest_err = None
    for y_stepsize in range(min_step_size, max_step_size):
        for y_offset in range(y_stepsize):
            rows = math.ceil((w - y_offset) / y_stepsize)
            grid_y_clean = []
            for row in range(rows):
                y = y_offset + (row * y_stepsize)
                grid_y_clean.append(y)

            err = get_mse_grid(grid_y, grid_y_clean) + rows
            if lowest_err is None or err < lowest_err:
                lowest_err = err
                best_grid_params["y_stepsize"] = y_stepsize
                best_grid_params["y_offset"] = y_offset

    end = time.perf_counter()
    took = end-start
    #print("get_clean_grid_v2", "took", took, "best_grid_params", best_grid_params)

    grid_y_clean = []
    grid_x_clean = []
    x_stepsize = best_grid_params["x_stepsize"]
    x_offset = best_grid_params["x_offset"]
    y_stepsize = best_grid_params["y_stepsize"]
    y_offset = best_grid_params["y_offset"]
    cols = math.ceil((w-x_offset) / x_stepsize)
    rows = math.ceil((h-y_offset) / y_stepsize)
    for col in range(cols):
        x = x_offset + (col * x_stepsize)
        grid_x_clean.append(x)
    for row in range(rows):
        y = y_offset + (row * y_stepsize)
        grid_y_clean.append(y)


    borders_t, borders_l, borders_b, borders_r = mosaic_borders_image(img, mask, mosaic_min_size=20)
    if borders_t:
        grid_y_clean.insert(0, 0)
    if borders_l:
        grid_x_clean.insert(0, 0)
    if borders_b:
        grid_y_clean.append(h - 1)
    if borders_r:
        grid_x_clean.append(w - 1)

    #print(f"grid_x_clean: {grid_x_clean}, grid_y_clean: {grid_y_clean}")

    if draw:
        out = img.copy()
        color = [255, 0, 0]
        thickness = 1
        for x in grid_x_clean:
            cv.line(out, (x, 0), (x, h), color=color, thickness=thickness)
        for y in grid_y_clean:
            cv.line(out, (0, y), (w, y), color=color, thickness=thickness)
        out = visualization.overlay_mask_boundary(out, mask)
        return out, grid_x_clean, grid_y_clean
    else:
        return grid_x_clean, grid_y_clean


def get_grid_via_harris(edges, img=None, draw=False):
    h, w = edges.shape[:2]
    corners, corner_coords = get_corners(edges, draw=True)

    lowest_err = None
    best_grid_params = None
    start = time.perf_counter()
    for stepsize_y in range(4, 35):
        for stepsize_x in range(4,35):
            grid = draw_grid(h, w, stepsize_x, stepsize_y, color=255, thickness=1)
            for offset_x in range(stepsize_x):
                for offset_y in range(stepsize_y):
                    shifted_grid = np.pad(grid[offset_y:, offset_x:], ((0,offset_y),(0,offset_x)), constant_values=0, mode='constant')

                    err = mse(shifted_grid, corners)
                    if lowest_err is None or err < lowest_err:
                        lowest_err = err
                        best_grid_params = dict(stepsize_x=stepsize_x, stepsize_y=stepsize_y, offset_x=offset_x,
                                                offset_y=offset_y)
    end = time.perf_counter()
    took = end-start
    print("get_grid_via_harris", "took", took, "best_grid_params", best_grid_params)

    grid_x = []
    grid_y = []
    cols = math.ceil(w / best_grid_params["stepsize_x"])
    rows = math.ceil(h / best_grid_params["stepsize_y"])
    for x in np.linspace(start=best_grid_params["stepsize_x"]+best_grid_params["offset_x"], stop=w-best_grid_params["stepsize_x"]-best_grid_params["offset_x"], num=cols-1):
        x = int(round(x))
        grid_x.append(x)
    for y in np.linspace(start=best_grid_params["stepsize_y"]+best_grid_params["offset_y"], stop=h-best_grid_params["stepsize_y"]-best_grid_params["offset_y"], num=rows-1):
        y = int(round(y))
        grid_y.append(y)

    if draw:
        grid = draw_grid(h, w, best_grid_params["stepsize_x"], best_grid_params["stepsize_y"], best_grid_params["offset_x"], best_grid_params["offset_y"], color=255, thickness=1)
        out = np.copy(img)
        out[grid > 0] = [0, 0, 255]
        out[corners > 0] = [255, 0, 0]
        return out, grid_x, grid_y
    else:
        return grid_x, grid_y

def pixelize(img, block_size_x, block_size_y, offset_x, offset_y):
    n_h = block_size_y
    n_w = block_size_x
    h_start = n_h - 1 if offset_y == 0 else offset_y
    w_start = n_w - 1 if offset_x == 0 else offset_x
    h, w = img.shape[:2]
    h_step = math.ceil((h + offset_y) / n_h)
    w_step = math.ceil((w + offset_x) / n_w)
    pad_h = n_h
    pad_w = n_w
    img_padded = np.pad(img,((pad_h,pad_h),(pad_w,pad_w),(0,0)), mode='reflect')
    img_mosaic = img_padded.copy()

    for i in range(h_step):
        for j in range(w_step):
            img_mosaic[i * n_h + h_start:(i + 1) * n_h + h_start, j * n_w + w_start:(j + 1) * n_w + w_start,:] = img_padded[i*n_h+h_start:(i+1)*n_h+h_start,j*n_w+w_start:(j+1)*n_w+w_start,:].mean(axis=(0,1))

    img_mosaic_unpad = img_mosaic[pad_h:h+pad_h, pad_w:w+pad_w]
    assert img_mosaic_unpad.shape == img.shape
    return img_mosaic_unpad

def get_grid_via_mosaic(img, mask, draw=False):
    image = img.copy()
    image[mask != 255] = 0
    h, w = edges.shape[:2]
    lowest_err = None
    lowest_err_img_mosaic = None
    best_grid_params = None
    start=time.perf_counter()
    for stepsize_y in range(4, 35):
        for stepsize_x in range(4,35):
            for offset_x in range(stepsize_x):
                for offset_y in range(stepsize_y):
                    img_mosaic = pixelize(image, block_size_x=stepsize_x, block_size_y=stepsize_y, offset_x=offset_x, offset_y=offset_y)
                    err = mse(image, img_mosaic)
                    if lowest_err is None or err < lowest_err:
                        lowest_err = err
                        lowest_err_img_mosaic = img_mosaic
                        best_grid_params = dict(stepsize_x=stepsize_x, stepsize_y=stepsize_y, offset_x=offset_x,
                                                offset_y=offset_y)
    end = time.perf_counter()
    took = end-start
    print("get_grid_via_mosaic", "took", took, "best_grid_params", best_grid_params)

    grid_x = []
    grid_y = []
    cols = math.ceil(w / best_grid_params["stepsize_x"])
    rows = math.ceil(h / best_grid_params["stepsize_y"])
    for x in np.linspace(start=best_grid_params["stepsize_x"]+best_grid_params["offset_x"], stop=w-best_grid_params["stepsize_x"]-best_grid_params["offset_x"], num=cols-1):
        x = int(round(x))
        grid_x.append(x)
    for y in np.linspace(start=best_grid_params["stepsize_y"]+best_grid_params["offset_y"], stop=h-best_grid_params["stepsize_y"]-best_grid_params["offset_y"], num=rows-1):
        y = int(round(y))
        grid_y.append(y)
    return (lowest_err_img_mosaic, grid_x, grid_y) if draw else (grid_x, grid_y)

def mosaic_borders_image(img_unpad, mask_unpad, mosaic_min_size):
    # if mosaic is bordering image it will not be detected as edges so let's add those lines manually
    h_unpad, w_unpad = img_unpad.shape[:2]
    borders_t = np.count_nonzero(mask_unpad[0, :]) > mosaic_min_size
    borders_b = np.count_nonzero(mask_unpad[h_unpad - 1, :]) > mosaic_min_size
    borders_l = np.count_nonzero(mask_unpad[:, 0]) > mosaic_min_size
    borders_r = np.count_nonzero(mask_unpad[:, w_unpad - 1]) > mosaic_min_size
    return borders_t, borders_l, borders_b, borders_r


def clean_cropped_mosaic(mosaic, mask, pad=(0,0,0,0), draw=False, pidinet_model=None):
    sharp_amount = 0.0
    canny_min=8 # 10
    canny_max=16 # 20
    hough_min_length=20

    h, w = mosaic.shape[:2]
    (pad_h_t, pad_h_b, pad_w_l, pad_w_r) = pad
    mask = mask.squeeze()
    mosaic_unpad = mosaic[pad_h_t:h - pad_h_b, pad_w_l:w - pad_w_r]
    mask_unpad = mask[pad_h_t:h - pad_h_b, pad_w_l:w - pad_w_r]

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask_unpad = cv.dilate(mask_unpad, kernel)

    if pidinet_model:
        edges = pidinet_inference.inference(pidinet_model, [mosaic_unpad])[0]
    else:
        gray = cv.cvtColor(mosaic_unpad, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        gray = unsharp_mask(gray, kernel_size=(5, 5), sigma=1.0, amount=sharp_amount, threshold=0)
        edges = cv.Canny(gray,canny_min,canny_max)
        edges[mask_unpad != 255] = 0

    horizontal = get_horizontal(edges)
    vertical = get_vertical(edges)
    grid = horizontal + vertical

    grid_x, grid_y = get_grid_via_hough(grid, img=mosaic_unpad, draw=False, hough_min_length=hough_min_length)
    if len(grid_x) == 0 or len(grid_y) == 0:
        return (mosaic, mosaic) if draw else mosaic
    if draw:
        draw_out, grid_x_clean, grid_y_clean = get_clean_grid_v2(grid_x, grid_y, mosaic_unpad, mask_unpad, draw=True)
        draw_out = np.pad(draw_out, ((pad_h_t, pad_h_b), (pad_w_l, pad_w_r), (0, 0)), mode='constant', constant_values=0)
    else:
        grid_x_clean, grid_y_clean = get_clean_grid_v2(grid_x, grid_y, mosaic_unpad, mask_unpad, draw=False)
    result = get_cleaned_mosaic_raw(mosaic, mosaic_unpad, mask_unpad, pad, grid_x_clean, grid_y_clean)

    return (result, draw_out) if draw else result

if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    from ultralytics import YOLO
    from lada.lib.mosaic_detector import MosaicDetectorDeprecated

    input = 'sample_vid.mp4'

    mosaic_detection_model = YOLO('yolo/runs/segment/train_mosaic_detection_yolov9c/weights/best.pt')
    pidinet_model = pidinet_inference.load_model("experiments/pidinet/run1/save_models/checkpoint_019.pth")
    mosaic_generator = MosaicDetectorDeprecated(mosaic_detection_model, input, 30, 256, pad_mode='zero')
    quit = False
    for clip_id, clip in enumerate(mosaic_generator()):
        images = []
        orig_images = clip.get_clip_images()
        for frame_id, (img, mask, _, _, pad) in enumerate(clip):
            print(f"clip: {clip_id:02d}, frame: {frame_id:04d}")
            mask = mask.squeeze()

            h, w = img.shape[:2]
            (pad_h_t, pad_h_b, pad_w_l, pad_w_r) = pad
            img_unpad = img[pad_h_t:h - pad_h_b, pad_w_l:w - pad_w_r]
            mask_unpad = mask[pad_h_t:h - pad_h_b, pad_w_l:w - pad_w_r]

            sharpen_amount = 0.0

            gray = cv.cvtColor(img_unpad, cv.COLOR_BGR2GRAY)
            gray = cv.GaussianBlur(gray, (3, 3), 0)
            gray = unsharp_mask(gray, kernel_size=(5, 5), sigma=1.0, amount=sharpen_amount, threshold=0)
            gray_orig = np.copy(gray)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            mask_unpad = cv.dilate(mask_unpad, kernel)

            # canny_min = 10
            # canny_max = 60
            # sharpen_amount = 0.5

            # canny_min = 20
            # canny_max = 30
            # sharpen_amount = 1.0

            canny_min = 8
            canny_max = 16

            edges = cv.Canny(gray, canny_min, canny_max)
            edges[mask_unpad != 255] = 0

            #edges = pidinet_inference.inference(pidinet_model, [img_unpad])[0]

            for window in ('gray', 'edges', 'horizontal', 'vertical', 'grid', 'img','clean', 'grid_clean', 'hough', 'houghp', 'mask'):
                cv.namedWindow(window, cv.WINDOW_NORMAL)
                sleep(0.0001)

            cv.imshow('edges', edges)
            cv.imshow('gray', gray)
            cv.imshow('mask', mask_unpad)
            #cv.imshow('grid', draw_grid(h, w, 12, 12, thickness=1))

            def clean_it():
                horizontal = get_horizontal(edges)
                vertical = get_vertical(edges)
                grid = horizontal + vertical

                hough_min_length = 20

                hough, grid_x_h, grid_y_h = get_grid_via_hough(grid, img=img_unpad, draw=True, hough_min_length=hough_min_length)
                grid_clean_h, grid_x_clean_h, grid_y_clean_h = get_clean_grid_v2(grid_x_h, grid_y_h, img=img_unpad, mask=mask_unpad, draw=True)

                houghp, grid_x_hp, grid_y_hp = get_grid_via_houghp(grid, img=img_unpad, draw=True, hough_min_length=hough_min_length)
                grid_clean_hp, grid_x_clean_hp, grid_y_clean_hp = get_clean_grid_v2(grid_x_hp, grid_y_hp, img=img_unpad, mask=mask_unpad, draw=True)

                #harris, grid_x_clean_harr, grid_y_clean_harr = get_grid_via_harris(grid, img=img_unpad, draw=True)
                #mosaic, grid_x_clean_m, grid_y_clean_m = get_grid_via_mosaic(img_unpad, mask_unpad, draw=True)

                mosaic_clean = get_cleaned_mosaic_raw(img, img_unpad, mask_unpad, pad, grid_x_clean_h, grid_y_clean_h)

                cv.imshow('horizontal', horizontal)
                cv.imshow('vertical', vertical)
                cv.imshow('grid', grid)
                cv.imshow('img', img)
                cv.imshow('hough', hough)
                cv.imshow('houghp', houghp)
                cv.imshow('clean', mosaic_clean)
                cv.imshow('grid_clean', grid_clean_h)
                #cv.imshow('mosaic', mosaic)
                #cv.imshow('harris', harris)

            def update_canny_min(new_canny_min):
                global canny_min
                global edges
                canny_min = new_canny_min
                edges = cv.Canny(gray, canny_min, canny_max)
                edges[mask_unpad != 255] = 0
                cv.imshow('edges', edges)
                clean_it()

            def update_canny_max(new_canny_max):
                print("canny_max", new_canny_max)
                global canny_max
                global edges
                canny_max = new_canny_max
                edges = cv.Canny(gray, canny_min, canny_max)
                edges[mask_unpad != 255] = 0
                cv.imshow('edges', edges)
                clean_it()

            def update_sharpen(new_sharpen_amount_int):
                global sharpen_amount
                global gray
                global edges
                sharpen_amount = new_sharpen_amount_int / 10.
                gray = unsharp_mask(gray_orig, kernel_size=(5, 5), sigma=1.0, amount=sharpen_amount, threshold=0)
                cv.imshow('gray', gray)
                edges = cv.Canny(gray, canny_min, canny_max)
                edges[mask_unpad != 255] = 0
                cv.imshow('edges', edges)
                clean_it()


            cv.createTrackbar('sharpen', 'edges', int(sharpen_amount * 10), 100, update_sharpen)
            cv.createTrackbar('canny min', 'edges', canny_min, 200, update_canny_min)
            cv.createTrackbar('canny max', 'edges', canny_max, 200, update_canny_max)

            clean_it()

            while True:
                key_pressed = cv.waitKey(0)
                if key_pressed & 0xFF == ord("q"):
                    quit = True
                    break
                elif key_pressed & 0xFF == ord("n"):
                    break
            if quit:
                break
        if quit:
            break
    cv.destroyAllWindows()
