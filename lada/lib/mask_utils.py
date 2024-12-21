import time

import cv2
import numpy as np

from lada.lib import Box, Mask
from lada.lib import image_utils
from lada.lib import visualization_utils


def get_box(mask: Mask) -> Box:
    points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(points)
    t, l, b, r = y, x, y+h, x+w
    box = t, l, b, r
    return box

def morph(mask: Mask, iterations=1) -> Mask:
    if get_mask_area(mask) < 0.01:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=iterations)

def dilate_mask(mask: Mask, dilatation_size=11, iterations=2):
    if iterations == 0:
        return mask
    element = np.ones((dilatation_size, dilatation_size), np.uint8)
    mask_img = cv2.dilate(mask, element, iterations=iterations).reshape(mask.shape)
    return mask_img

def extend_mask(mask: Mask, value) -> Mask:
    # value between 0 and 3 -> higher values mean more extension of mask area. 0 does not change mask at all
    if value == 0:
        return mask
    target_size = 256
    # Dilations are slow when using huge kernels (which we would need for high-res masks). therefore we downscale mask to perform morph operations on much smaller pixel space with smaller kernels
    extended_mask = clean_up_boundaries(image_utils.resize(morph(image_utils.resize(mask, target_size, interpolation=cv2.INTER_NEAREST), iterations=value), mask.shape[:2], interpolation=cv2.INTER_NEAREST)).reshape(mask.shape)
    assert mask.shape == extended_mask.shape
    return extended_mask

def clean_up_boundaries(mask: Mask, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) -> Mask:
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def fill_holes(mask: Mask) -> Mask:
    edited_mask = np.zeros_like(mask, dtype=mask.dtype)
    contour, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(edited_mask, [cnt], 0, 255, -1)

    return edited_mask

def get_mask_area(mask: Mask) -> float:
    pixels = cv2.countNonZero(mask)
    return pixels / (mask.shape[0] * mask.shape[1])


if __name__ == "__main__":
    mask = cv2.imread("mask3.png")
    mask_orig = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    output_mask_orig = mask_orig.copy()
    visualization_utils.draw_box(output_mask_orig, get_box(mask_orig))
    cv2.imshow("orig", output_mask_orig)

    mask_extended = extend_mask(mask_orig, 1)
    output_mask_extended = mask_extended.copy()
    visualization_utils.draw_box(output_mask_extended, get_box(mask_extended))
    cv2.imshow("extended", output_mask_extended)

    while True:
        key_pressed = cv2.waitKey(1)
        if key_pressed & 0xFF == ord("q"):
            print("q")
            break
        elif key_pressed & 0xFF == ord("r"):
            print("r")
            value = np.random.choice([0, 0, 1, 2, 3])
            start = time.time()
            mask_extended = extend_mask(mask_orig, value)
            print(f"took: {time.time()-start}")
            output_mask_extended = mask_extended.copy()
            visualization_utils.draw_box(output_mask_extended, get_box(mask_extended))
            cv2.imshow("extended", output_mask_extended)
            print(get_mask_area(mask_extended))

    cv2.destroyAllWindows()
