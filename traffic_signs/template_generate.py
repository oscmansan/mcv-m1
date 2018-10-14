#!/usr/bin/env python3
import os
import glob
import argparse
import multiprocessing as mp
from skimage import color

import numpy as np
import imageio
from collections import defaultdict, Counter
import colorsys

from candidate_generation_pixel import candidate_generation_pixel


def worker(x):
    image_file, output_dir, pixel_method = x

    name = os.path.splitext(os.path.split(image_file)[1])[0]

    im = imageio.imread(image_file)
    print(image_file)

    pixel_candidates = candidate_generation_pixel(im, pixel_method)

    fd = os.path.join(output_dir, pixel_method)
    if not os.path.exists(fd):
        os.makedirs(fd)

    out_mask_name = os.path.join(fd, name + '.png')
    imageio.imwrite(out_mask_name, np.uint8(np.round(pixel_candidates)))


def traffic_signal(mask, bbox):
    tly, tlx, bry, brx = bbox
    return mask[tly:bry,tlx:brx]

def generate_masks(images_dir, output_dir, pixel_method):
    images = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))

    with mp.Pool(processes=8) as p:
        p.map(worker, [(image_file, output_dir, pixel_method) for image_file in images])


if __name__ == '__main__':
    images_label = defaultdict(list)
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for img_file in sorted(glob.glob('train_val/train/*.jpg')):
        name = os.path.splitext(os.path.split(img_file)[1])[0]
        mask_file = 'train_val/train/mask/mask.{}.png'.format(name)
        gt_file = 'train_val/train/gt/gt.{}.txt'.format(name)
        img = imageio.imread(img_file)
        mask = imageio.imread(mask_file)
        gts = [line.split(' ') for line in open(gt_file, 'r').read().splitlines()]
        for gt in gts:
            bbox = np.round(list(map(int, map(float, gt[:4]))))
            label = gt[4]
            images_label[label].append(color.rgb2gray(traffic_signal(mask, bbox)))

    for label in labels:
        i = 0
        mean_gray = 0
        for image in images_label[label]:
            i += 1
            mean_gray += image
