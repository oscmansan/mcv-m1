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


    images_label = defaultdict(list)
    mean_gray = defaultdict(int)
    mean_form_factor = defaultdict(int)
    mean_filling_ratio = defaultdict(int)
    mean_size = defaultdict(int)
    height = defaultdict(int)
    width = defaultdict(int)
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

            if label == 'A':
                shape = 'triangle'
            elif label == 'B':
                shape = 'triangle_inv'
            elif label == 'F':
                shape = 'square'
            else:
                shape = 'circle'


            images_label[shape].append(color.rgb2gray(traffic_signal(img, bbox)))
            mean_form_factor[shape] += form_factor(bbox)
            mean_filling_ratio[shape] += filling_ratio(mask, bbox)
            mean_size[shape] += size(mask, bbox)
            #plt.imshow(traffic_signal(img, mask, bbox))
            #plt.show()
            #plt.imshow(color.rgb2gray(traffic_signal(img, mask, bbox)))
            #plt.show()

    for label in images_label.keys():
        print(label)
        if len(images_label[label]) > 0:
            i = 0
            for image in images_label[label]:
                i += 1
                [r, c] = image.shape
                mean_gray[label] += sum(sum(image))/(r*c)
                #plt.imshow(image/mean_gray[label])
                #plt.show()

            mean_gray[label] = mean_gray[label]/i
            mean_form_factor[label] = mean_form_factor[label]/i
            mean_filling_ratio[label] = mean_filling_ratio[label]/i
            mean_size[label] = mean_size[label]/i
            width[label] = round(np.sqrt(mean_size[label]*mean_form_factor[label]))
            height[label] = round(np.sqrt(mean_size[label]/mean_form_factor[label]))
            print('MEAN')
            print(mean_gray)
            print('FORM')
            print(mean_form_factor)
            print('FILL')
            print(mean_filling_ratio)
            print('SIZE')
            print(mean_size)
            print('Height')
            print(height)
            print('Width')
            print(width)

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


            #img = Image.open(img_file)
            #mask = Image.open(mask_file)
            #resized_img = img.resize((width[label], height[label]))
            #resized_mask = mask.resize((width[label], height[label]))
            img_part = traffic_signal(img, bbox)
            mask_part = traffic_signal(mask, bbox)


            #resized_img = Image.fromarray(img_part)
            #resized_img = resized_img.resize((int(width[label]), int(height[label])))
            #resized_img = resized_img.resize((80, 80))



            #images_label[label].append(color.rgb2gray(resized_img))
            #plt.imshow(traffic_signal(img, mask, bbox))
            #plt.show()
            #plt.imshow(color.rgb2gray(traffic_signal(img, mask, bbox)))
            #plt.show()

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

            quantity_images[shape] += 1

    for key in template_gray.keys():
        template_gray[key] = template_gray[key]/quantity_images[key]
        template_mask[key] = template_mask[key]/quantity_images[key]

        plt.imshow(template_gray[key])
        plt.show()

        plt.imshow(template_mask[key])
        plt.show()
