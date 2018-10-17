#!/usr/bin/python
# -*- coding: utf-8 -*-

from skimage import data
from skimage.measure import label, regionprops
import imageio
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import glob
from window_evaluation import ccl_window_evaluation
from candidate_generation_pixel import candidate_generation_pixel

def candidate_generation_window_example1(im, pixel_candidates):
    window_candidates = [[17.0, 12.0, 49.0, 44.0], [60.0,90.0,100.0,130.0]]

    return window_candidates

def candidate_generation_window_example2(im, pixel_candidates):
    window_candidates = [[21.0, 14.0, 54.0, 47.0], [63.0,92.0,103.0,132.0],[200.0,200.0,250.0,250.0]]

    return window_candidates

# Create your own candidate_generation_window_xxx functions for other methods
# Add them to the switcher dictionary in the switch_method() function
# These functions should take an image, a pixel_candidates mask (and perhaps other parameters) as input and output the window_candidates list.

def candidate_generation_window_ccl(im, pixel_candidates):

    label_image = label(pixel_candidates)
    regions = regionprops(label_image)
    window_candidates = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        if ccl_window_evaluation(pixel_candidates, region.bbox):
            window_candidates.append([minr, minc, maxr, maxc])
    return window_candidates

def switch_method(im, pixel_candidates, method):
    switcher = {
        'example1': candidate_generation_window_example1,
        'ccl': candidate_generation_window_ccl
    }
    # Get the function from switcher dictionary
    func = switcher.get(method, lambda: "Invalid method")

    # Execute the function
    window_candidates = func(im, pixel_candidates)

    return window_candidates


def visualize_boxes(pixel_candidates, window_candidates):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(pixel_candidates * 255)
    for candidate in window_candidates:
        minr, minc, maxr, maxc = candidate
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    plt.show()



def candidate_generation_window(im, pixel_candidates, method):

    window_candidates = switch_method(im, pixel_candidates, method)

    return window_candidates

    
if __name__ == '__main__':
    for img_file in sorted(glob.glob('data/train/*.jpg')):
        img_file = imageio.imread(img_file)
        pixel_candidates = candidate_generation_pixel(img_file, "ihsl_1")
        window_candidates = candidate_generation_window(img_file, pixel_candidates, 'ccl')
        visualize_boxes(pixel_candidates, window_candidates)


    
