#!/usr/bin/env python3
import os
import glob
from collections import defaultdict

import imageio
import numpy as np

from evaluation.bbox_iou import bbox_iou


def size(mask, bbox):
    tly, tlx, bry, brx = bbox
    return np.count_nonzero(mask[tly:bry, tlx:brx])


def form_factor(bbox):
    tly, tlx, bry, brx = bbox
    width = brx - tlx
    height = bry - tly
    return width / height


def filling_ratio(mask, bbox):
    tly, tlx, bry, brx = bbox
    width = brx - tlx
    height = bry - tly
    bbox_area = width * height
    mask_area = size(mask, bbox)
    return mask_area / bbox_area


def compute_num_overlap(gts, img):
    if len(gts) > 1:
        for i in range(len(gts)):
            bboxA = list(map(float, gts[i][:4]))
            for j in range(i + 1, len(gts)):
                bboxB = list(map(float, gts[j][:4]))
                print(img, bbox_iou(bboxA, bboxB))


def main():
    class_frequency = defaultdict(int)
    size_per_class = defaultdict(list)
    form_factor_per_class = defaultdict(list)
    filling_ratio_per_class = defaultdict(list)
    for img_file in sorted(glob.glob('data/train/*.jpg')):
        name = os.path.splitext(os.path.split(img_file)[1])[0]
        mask_file = 'data/train/mask/mask.{}.png'.format(name)
        gt_file = 'data/train/gt/gt.{}.txt'.format(name)
        img = imageio.imread(img_file)
        mask = imageio.imread(mask_file)
        gts = [line.split(' ') for line in open(gt_file, 'r').read().splitlines()]
        for gt in gts:
            bbox = np.round(list(map(int, map(float, gt[:4]))))
            label = gt[4]

            class_frequency[label] += 1
            size_per_class[label].append(size(mask, bbox))
            form_factor_per_class[label].append(form_factor(bbox))
            filling_ratio_per_class[label].append(filling_ratio(mask, bbox))

    for clase in form_factor_per_class.keys():
        print("CLASE ", clase)
        print("FORM FACTOR", "max", max(form_factor_per_class[clase]), "min", min(form_factor_per_class[clase]))
        print("FILLING RATIO", "max", max(filling_ratio_per_class[clase]), "min", min(filling_ratio_per_class[clase]))
        print("SIZE", "max", max(size_per_class[clase]), "min", min(size_per_class[clase]))


if __name__ == '__main__':
    main()
