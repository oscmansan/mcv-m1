#!/usr/bin/env python3
import multiprocessing as mp

import numpy as np
import imageio
from skimage.measure import label, regionprops
from skimage.transform import resize
import cv2

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from evaluation.bbox_iou import bbox_iou
import window_evaluation as we


def candidate_generation_window_example1(im, pixel_candidates):
    window_candidates = [[17.0, 12.0, 49.0, 44.0], [60.0,90.0,100.0,130.0]]

    return window_candidates


def candidate_generation_window_example2(im, pixel_candidates):
    window_candidates = [[21.0, 14.0, 54.0, 47.0], [63.0,92.0,103.0,132.0],[200.0,200.0,250.0,250.0]]

    return window_candidates


# Create your own candidate_generation_window_xxx functions for other methods
# Add them to the switcher dictionary in the switch_method() function
# These functions should take an image, a pixel_candidates mask (and perhaps other parameters) as input and output the window_candidates list.


def candidate_generation_window_ccl(im, pixel_candidates, eval_method):
    label_image = label(pixel_candidates)
    regions = regionprops(label_image)
    masked_im = im*pixel_candidates

    window_candidates = []
    for region in regions:
        bbox = list(region.bbox)
        box_h, box_w = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if eval_method == 'features':
            if we.window_evaluation_features(pixel_candidates, bbox):
                window_candidates.append(bbox)
        elif eval_method == 'template':
            for template in we.TEMPLATES:
                template = resize(template, (box_h, box_w), preserve_range=True).astype(np.uint8)
                if we.window_evaluation_template(masked_im, bbox, template, corr_thresh=0.7):
                    window_candidates.append(bbox)
        else:
            raise ValueError('Invalid method {}'.format(eval_method))
    return window_candidates


def is_contained(container, content):
    return container[0] <= content[0] and \
           container[1] <= content[1] and \
           container[2] >= content[2] and \
           container[3] >= content[3]


def nms(bboxes, threshold=.4):
    # based on https://github.com/opencv/opencv/blob/master/modules/dnn/src/nms.inl.hpp

    # sort bboxes by area descending
    bboxes.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)

    indices = []
    for idx in range(len(bboxes)):
        keep = True
        for kept_idx in indices:
            overlap = bbox_iou(bboxes[idx], bboxes[kept_idx])
            contained = is_contained(bboxes[idx], bboxes[kept_idx]) or is_contained(bboxes[kept_idx], bboxes[idx])
            if contained or overlap > threshold:
                keep = False
                break
        if keep:
            indices.append(idx)
    return indices


def _worker_template(x):
    im, pixel_candidates, step, box_h, box_w = x
    h, w = im.shape[:2]
    masked_im = im*pixel_candidates

    window_candidates = []
    for template in we.TEMPLATES:
        template = resize(template, (box_h, box_w), preserve_range=True).astype(np.uint8)

        #for i in range(0, h-box_h, step):
        #    for j in range(0, w-box_w, step):
        #        bbox = [i, j, i+box_h, j+box_w]
        #        if we.window_evaluation_template(masked_im, bbox, template, corr_thresh=0.7):
        #            window_candidates.append(bbox)

        # OpenCV implementation of template matching is a lot faster
        res = cv2.matchTemplate(masked_im, template, method=cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        tlx, tly = max_loc
        score = max_val
        if score > 0.8:
            bbox = [tly, tlx, tly + box_h, tlx + box_w]
            window_candidates.append(bbox)

    return window_candidates


def _worker_features(x):
    im, pixel_candidates, step, box_h, box_w = x
    h, w = im.shape[:2]

    window_candidates = []
    for i in range(0, h-box_h, step):
        for j in range(0, w-box_w, step):
            bbox = [i, j, i+box_h, j+box_w]
            if we.window_evaluation_features(pixel_candidates, bbox):
                window_candidates.append(bbox)
    return window_candidates


def sliding_window_par(im, pixel_candidates, eval_method, step=5, nms_threshold=.1):
    # scales = [(h, w) for h in range(180, 30-1, -50) for w in range(180, 30-1, -50)]
    scales = [(217, 235), (181, 186), (140, 143), (164, 113), (116, 117), (100, 99), (119, 78), (82, 80), (66, 60), (45, 43)]

    if eval_method == 'features':
        worker = _worker_features
    elif eval_method == 'template':
        worker = _worker_template
    else:
        raise ValueError('Invalid method {}'.format(eval_method))

    with mp.Pool(processes=12) as p:
        window_candidates = p.map(worker, [(im, pixel_candidates, step, box_h, box_w) for box_h, box_w in scales])
    window_candidates = [bbox for sublist in window_candidates for bbox in sublist]

    indices = nms(window_candidates, threshold=nms_threshold)
    window_candidates = [window_candidates[idx] for idx in indices]

    return window_candidates


def sliding_window_seq(im, pixel_candidates, eval_method, step=5, nms_threshold=.1):
    h, w = im.shape[:2]
    # scales = [(h, w) for h in range(180, 30-1, -50) for w in range(180, 30-1, -50)]
    scales = [(217, 235), (181, 186), (140, 143), (164, 113), (116, 117), (100, 99), (119, 78), (82, 80), (66, 60), (45, 43)]

    # generate windows
    window_candidates = []
    for box_h, box_w in scales:
        if eval_method == 'features':
            for i in range(0, h-box_h, step):
                for j in range(0, w-box_w, step):
                    bbox = [i, j, i+box_h, j+box_w]
                    # evaluate window (possible detection)
                    if we.window_evaluation_features(pixel_candidates, bbox):
                        window_candidates.append(bbox)
        elif eval_method == 'template':
            for template in we.TEMPLATES:
                # resize template to match box size
                template = resize(template, (box_h, box_w), preserve_range=True).astype(np.uint8)
                for i in range(0, h - box_h, step):
                    for j in range(0, w - box_w, step):
                        bbox = [i, j, i + box_h, j + box_w]
                        # evaluate window (possible detection)
                        if we.window_evaluation_template(im*pixel_candidates, bbox, template):
                            window_candidates.append(bbox)
        else:
            raise ValueError('Invalid method {}'.format(eval_method))

    # remove overlapping detections
    indices = nms(window_candidates, threshold=nms_threshold)
    window_candidates = [window_candidates[idx] for idx in indices]

    return window_candidates


def visualize_boxes(pixel_candidates, window_candidates):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(pixel_candidates)
    for candidate in window_candidates:
        minr, minc, maxr, maxc = candidate
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    plt.show()


def switch_method(im, pixel_candidates, method):
    switcher = {
        'ccl_features': (candidate_generation_window_ccl, 'features'),
        'ccl_template': (candidate_generation_window_ccl, 'template'),
        'sw_features': (sliding_window_par, 'features'),
        'sw_template': (sliding_window_par, 'template')
    }

    # Get the function from switcher dictionary
    func, eval_method = switcher.get(method, lambda: "Invalid method")

    # Execute the function
    window_candidates = func(im, pixel_candidates, eval_method)

    return window_candidates


def candidate_generation_window(im, pixel_candidates, method):
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    window_candidates = switch_method(im_gray, pixel_candidates, method)
    return window_candidates


if __name__ == '__main__':
    import glob, os, time
    img_file = np.random.choice(glob.glob('data/train/*.jpg'))
    name = os.path.splitext(os.path.split(img_file)[1])[0]
    im = imageio.imread(img_file)
    mask = imageio.imread('output/hsv_ranges/{}.png'.format(name))

    start = time.time()
    window_candidates = candidate_generation_window(im, mask, 'sw_template')
    end = time.time()
    print('time: {}'.format(end-start))

    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    visualize_boxes(im_gray*mask, window_candidates)
    visualize_boxes(im, window_candidates)
