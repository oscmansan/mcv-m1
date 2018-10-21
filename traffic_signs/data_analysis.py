#!/usr/bin/env python3
import os
import glob
from collections import defaultdict

import imageio
import numpy as np
from sklearn.cluster import KMeans

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

def filling_ratio_integral(mask_area, bbox):
    tly, tlx, bry, brx = bbox
    width = brx - tlx
    height = bry - tly
    bbox_area = width * height
    return mask_area / bbox_area

def compute_num_overlap(gts, img):
    if len(gts) > 1:
        for i in range(len(gts)):
            bboxA = list(map(float, gts[i][:4]))
            for j in range(i + 1, len(gts)):
                bboxB = list(map(float, gts[j][:4]))
                print(img, bbox_iou(bboxA, bboxB))

def height(bbox):
    tly, tlx, bry, brx = bbox
    return bry - tly


def width(bbox):
    tly, tlx, bry, brx = bbox
    return brx - tlx


def compute_metrics():
    class_frequency = defaultdict(int)
    size_per_class = defaultdict(list)
    form_factor_per_class = defaultdict(list)
    filling_ratio_per_class = defaultdict(list)
    height_bbox = []
    width_bbox = []
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
            height_bbox.append(height(bbox))
            width_bbox.append(width(bbox))

    for clase in form_factor_per_class.keys():
        print("CLASE ", clase)
        print("FORM FACTOR", "max", max(form_factor_per_class[clase]), "min", min(form_factor_per_class[clase]))
        print("FILLING RATIO", "max", max(filling_ratio_per_class[clase]), "min", min(filling_ratio_per_class[clase]))
        print("SIZE", "max", max(size_per_class[clase]), "min", min(size_per_class[clase]))
    print("WIDTH", 'max', max(width_bbox), 'min', min(width_bbox), 'mean', np.mean(width_bbox), 'std',
          np.std(width_bbox))
    print("HEIGHT", 'max', max(height_bbox), 'min', min(height_bbox), 'mean', np.mean(height_bbox), 'std',
          np.std(height_bbox))


def compute_box_sizes(k):
    sizes = []
    for gt_file in glob.glob('data/train/gt/gt.*.txt'):
        gts = [line.split(' ') for line in open(gt_file, 'r').read().splitlines()]
        for gt in gts:
            bbox = np.round(list(map(int, map(float, gt[:4]))))
            height = bbox[2] - bbox[0]
            width = bbox[3] - bbox[1]
            sizes.append([height, width])
    sizes = np.array(sizes)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(sizes)
    centroids = kmeans.cluster_centers_
    return centroids


def main():
    sizes = compute_box_sizes(k=10)
    print(sorted([(h, w) for h, w in sizes.round().astype(np.uint32)], key=lambda x: x[0] * x[1], reverse=True))


if __name__ == '__main__':
    main()
