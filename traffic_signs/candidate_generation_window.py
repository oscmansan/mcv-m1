#!/usr/bin/env python3
import multiprocessing as mp

import numpy as np
import imageio
from skimage.measure import label, regionprops
from skimage import transform

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import cv2

from evaluation.bbox_iou import bbox_iou
from window_evaluation import ccl_window_evaluation, template_matching_evaluation


def candidate_generation_window_example1(im, pixel_candidates):
    window_candidates = [[17.0, 12.0, 49.0, 44.0], [60.0,90.0,100.0,130.0]]

    return window_candidates


def candidate_generation_window_example2(im, pixel_candidates):
    window_candidates = [[21.0, 14.0, 54.0, 47.0], [63.0,92.0,103.0,132.0],[200.0,200.0,250.0,250.0]]

    return window_candidates

def window_evaluation(pixel_candidates, bbox):
    return bool(np.random.binomial(1, 0.0001))  # returns True with probability 0.0001


def candidate_generation_window_ccl(im, pixel_candidates):
    label_image = label(pixel_candidates)
    regions = regionprops(label_image)
    window_candidates = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        if ccl_window_evaluation(pixel_candidates, region.bbox):
            window_candidates.append([minr, minc, maxr, maxc])
    return window_candidates


def nms(bboxes, threshold=.5):
    # based on https://github.com/opencv/opencv/blob/master/modules/dnn/src/nms.inl.hpp
    indices = []
    for idx in range(len(bboxes)):
        keep = True
        for kept_idx in indices:
            overlap = bbox_iou(bboxes[idx], bboxes[kept_idx])
            if overlap > threshold:
                keep = False
                break
        if keep:
            indices.append(idx)
    return indices


def _worker_template(x):
    im, pixel_candidates, step, box_h, box_w, template = x
    h, w = im.shape[:2]
    template = transform.resize(template, (box_h, box_w), preserve_range=True)
    template = np.round(template).astype(np.uint8)

    window_candidates = []
    for i in range(0, h-box_h, step):
        print(i)
        for j in range(0, w-box_w, step):
            bbox = [i, j, i+box_h, j+box_w]
            if template_matching_evaluation(im, template, bbox):
                window_candidates.append(bbox)

    return window_candidates


def template_matching(im, pixel_candidates, template, step=5, nms_threshold=.4):
    #scales = [(h, w) for h in range(50, 250, 50) for w in range(50, 250, 50)]
    scales = [(150, 150)]

    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = np.round(im).astype(np.uint8)

    with mp.Pool(processes=4) as p:
        window_candidates = p.map(_worker_template, [(im, pixel_candidates, step, box_h, box_w, template) for box_h, box_w in scales])
    window_candidates = [bbox for sublist in window_candidates for bbox in sublist]

    indices = nms(window_candidates, threshold=nms_threshold)
    window_candidates = [window_candidates[idx] for idx in indices]

    return window_candidates


def _worker(x):
    im, pixel_candidates, step, box_h, box_w = x
    h, w = im.shape[:2]

    window_candidates = []
    for i in range(0, h-box_h, step):
        for j in range(0, w-box_w, step):
            bbox = [i, j, i+box_h, j+box_w]
            if window_evaluation(pixel_candidates, bbox):
                window_candidates.append(bbox)

    return window_candidates


def sliding_window_par(im, pixel_candidates, step=5, nms_threshold=.4):
    scales = [(h, w) for h in range(50, 250, 50) for w in range(50, 250, 50)]

    with mp.Pool(processes=12) as p:
        window_candidates = p.map(_worker, [(im, pixel_candidates, step, box_h, box_w) for box_h, box_w in scales])
    window_candidates = [bbox for sublist in window_candidates for bbox in sublist]

    indices = nms(window_candidates, threshold=nms_threshold)
    window_candidates = [window_candidates[idx] for idx in indices]

    return window_candidates


def sliding_window_seq(im, pixel_candidates, step=5, nms_threshold=.4):
    h, w = im.shape[:2]
    scales = [(h, w) for h in range(50, 200, 50) for w in range(50, 200, 50)]  # TODO: adjust scales based on granulometry

    # generate windows
    window_candidates = []
    for box_h, box_w in scales:
        for i in range(0, h-box_h, step):
            for j in range(0, w-box_w, step):
                bbox = [i, j, i+box_h, j+box_w]
                # evaluate window (possible detection)
                if window_evaluation(pixel_candidates, bbox):
                    window_candidates.append(bbox)

    # remove overlapping detections
    indices = nms(window_candidates, threshold=nms_threshold)
    window_candidates = [window_candidates[idx] for idx in indices]

    return window_candidates


def visualize_boxes(pixel_candidates, window_candidates):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(pixel_candidates * 255)
    for candidate in window_candidates:
        minr, minc, maxr, maxc = candidate
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    plt.show()


def switch_method(im, pixel_candidates, method):
    switcher = {
        'example1': candidate_generation_window_example1,
        'example2': candidate_generation_window_example2,
        'ccl': candidate_generation_window_ccl
    }

    # Get the function from switcher dictionary
    func = switcher.get(method, lambda: "Invalid method")

    # Execute the function
    window_candidates = func(im, pixel_candidates)

    return window_candidates


def candidate_generation_window(im, pixel_candidates, method):
    window_candidates = switch_method(im, pixel_candidates, method)
    return window_candidates


if __name__ == '__main__':
    #window_candidates = candidate_generation_window(im, pixel_candidates, 'ccl')
    #visualize_boxes(pixel_candidates, window_candidates)

    import glob, os, time
    imfile = np.random.choice(glob.glob('data/train/01.000936.jpg'))
    name = os.path.splitext(os.path.split(imfile)[1])[0]
    im = imageio.imread('data/train/{}.jpg'.format(name))
    mask = imageio.imread('data/train/mask/mask.{}.png'.format(name))
    template = imageio.imread('shape_templates/circle.png')

    start = time.time()
    window_candidates = template_matching(im, mask, template)
    end = time.time()
    print(end-start)

    visualize_boxes(im, window_candidates)