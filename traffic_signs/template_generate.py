#!/usr/bin/env python3
from __future__ import division

import os
import glob
import argparse
import multiprocessing as mp
from skimage import color
from skimage import transform,io
from PIL import Image

import numpy as np
import imageio
from collections import defaultdict, Counter
import colorsys
from matplotlib import pyplot as plt
import cv2


def traffic_signal(img, bbox):
    tly, tlx, bry, brx = bbox
    img_result = img[tly:bry, tlx:brx]
    return img_result


def size(mask, bbox):
    tly, tlx, bry, brx = bbox
    return np.count_nonzero(mask[tly:bry,tlx:brx])


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
    template_gray = dict()
    template_mask = dict()
    quantity_images = defaultdict(int)
    mean_gray = defaultdict(int)
    mean_form_factor = defaultdict(int)
    mean_size = defaultdict(int)
    height = defaultdict(int)
    width = defaultdict(int)

    #Analysis of data per shape (find average sizes of the masks)
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

            if label == 'A':
                shape = 'triangle'
            elif label == 'B':
                shape = 'triangle_inv'
            elif label == 'F':
                shape = 'square'
            else:
                shape = 'circle'


            mean_form_factor[shape] += form_factor(bbox)
            mean_size[shape] += size(mask, bbox)
            quantity_images[shape] += 1

    for label in quantity_images.keys():
        print(label)

        mean_form_factor[label] = mean_form_factor[label]/quantity_images[label]
        mean_size[label] = mean_size[label]/quantity_images[label]
        width[label] = round(np.sqrt(mean_size[label]*mean_form_factor[label]))
        height[label] = round(np.sqrt(mean_size[label]/mean_form_factor[label]))

        print('MEAN')
        print(mean_gray)
        print('FORM')
        print(mean_form_factor)
        print('SIZE')
        print(mean_size)
        print('Height')
        print(height)
        print('Width')
        print(width)

    #Computing the mean grayscale value of all signals per shape and the average of all masks per each shape
    for img_file in sorted(glob.glob('train_val/train/*.jpg')):
        name = os.path.splitext(os.path.split(img_file)[1])[0]
        mask_file = 'train_val/train/mask/mask.{}.png'.format(name)
        gt_file = 'train_val/train/gt/gt.{}.txt'.format(name)
        img = imageio.imread(img_file, as_gray = True)
        mask = imageio.imread(mask_file)
        gts = [line.split(' ') for line in open(gt_file, 'r').read().splitlines()]
        for gt in gts:
            bbox = np.round(list(map(int, map(float, gt[:4]))))
            label = gt[4]

            img_part = traffic_signal(img, bbox)
            mask_part = traffic_signal(mask, bbox)

            if label == 'A':
                shape = 'triangle'
            elif label == 'B':
                shape = 'triangle_inv'
            elif label == 'F':
                shape = 'square'
            else:
                shape = 'circle'

            resized_img = transform.resize(img_part, (int(width[shape]), int(height[shape])), preserve_range=True)
            resized_mask = transform.resize(mask_part, (int(width[shape]), int(height[shape])), preserve_range=True)

            if shape in template_gray.keys():
                template_gray[shape] += resized_img
                template_mask[shape] += resized_mask
            else:
                template_gray[shape] = resized_img
                template_mask[shape] = resized_mask


    #Visualize results and save templates in folder
    fd = os.path.join('shape_templates')
    if not os.path.exists(fd):
        os.makedirs(fd)

    for key in template_gray.keys():
        template_gray[key] = template_gray[key]/quantity_images[key]
        template_mask[key] = template_mask[key]/quantity_images[key]

        plt.imshow(template_gray[key])
        plt.show()

        plt.imshow(template_mask[key])
        plt.show()

        template = template_gray[key]*template_mask[key]
        plt.imshow(template)
        plt.show()

        out_mask_name = os.path.join(fd, key + '.png')
        imageio.imwrite(out_mask_name, np.uint8(np.round(template)))
