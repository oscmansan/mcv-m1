#!/usr/bin/env python3
from __future__ import division

import os
import glob
import argparse
import multiprocessing as mp
from skimage import color

import numpy as np
import imageio
from collections import defaultdict, Counter
import colorsys
from matplotlib import pyplot as plt
import cv2


def traffic_signal(img, mask, bbox):
    tly, tlx, bry, brx = bbox
    mask_r = mask[tly:bry, tlx:brx]
    mask3 = [mask[tly:bry, tlx:brx], mask[tly:bry, tlx:brx], mask[tly:bry, tlx:brx]]
    img_result = img[tly:bry, tlx:brx, :]
    return img_result

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


if __name__ == '__main__':
    images_label = defaultdict(list)
    mean_gray = defaultdict(int)
    mean_form_factor = defaultdict(int)
    mean_filling_ratio = defaultdict(int)
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for img_file in sorted(glob.glob('train_val/train/00.000948.jpg')):
        name = os.path.splitext(os.path.split(img_file)[1])[0]
        mask_file = 'train_val/train/mask/mask.{}.png'.format(name)
        gt_file = 'train_val/train/gt/gt.{}.txt'.format(name)
        img = imageio.imread(img_file)
        mask = imageio.imread(mask_file)
        gts = [line.split(' ') for line in open(gt_file, 'r').read().splitlines()]
        for gt in gts:
            bbox = np.round(list(map(int, map(float, gt[:4]))))
            label = gt[4]
            images_label[label].append(color.rgb2gray(traffic_signal(img, mask, bbox)))
            mean_form_factor[label] += form_factor(bbox)
            mean_filling_ratio[label] += filling_ratio(mask, bbox)
            #plt.imshow(traffic_signal(img, mask, bbox))
            #plt.show()
            #plt.imshow(color.rgb2gray(traffic_signal(img, mask, bbox)))
            #plt.show()

    for label in labels:
        print(label)
        if len(images_label[label]) > 0:
            i = 0
            for image in images_label[label]:
                i += 1
                [r, c] = image.shape
                mean_gray[label] += sum(sum(image))/(r*c)
                print('MEAN')
                print(mean_gray[label])
                #plt.imshow(image/mean_gray[label])
                #plt.show()
            mean_gray[label] = mean_gray[label]/i
            mean_form_factor[label] = mean_form_factor[label]/i
            mean_filling_ratio[label] = mean_filling_ratio[label]/i
            print('MEAN')
            print(mean_gray)
            print('FORM')
            print(mean_form_factor)
            print('FILL')
            print(mean_filling_ratio)



