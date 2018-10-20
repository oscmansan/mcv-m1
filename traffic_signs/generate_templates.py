#!/usr/bin/env python3
from __future__ import division

import os
import glob
import argparse
from skimage import transform

import numpy as np
import imageio
from collections import defaultdict
from matplotlib import pyplot as plt
from data_analysis import form_factor, size


def traffic_signal(img, bbox):
    tly, tlx, bry, brx = bbox
    img_result = img[tly:bry, tlx:brx]
    return img_result


def data_shape_analysis(images_dir):
    # Analysis of data per shape (find average sizes of the masks)

    quantity_images = defaultdict(int)
    mean_form_factor = defaultdict(int)
    mean_size = defaultdict(int)
    height = defaultdict(int)
    width = defaultdict(int)

    for img_file in sorted(glob.glob(os.path.join(images_dir, '*.jpg'))):
        name = os.path.splitext(os.path.split(img_file)[1])[0]
        mask_file = '{}/mask/mask.{}.png'.format(images_dir, name)
        gt_file = '{}/gt/gt.{}.txt'.format(images_dir, name)
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

        mean_form_factor[label] = mean_form_factor[label] / quantity_images[label]
        mean_size[label] = mean_size[label] / quantity_images[label]
        width[label] = round(np.sqrt(mean_size[label] * mean_form_factor[label]))
        height[label] = round(np.sqrt(mean_size[label] / mean_form_factor[label]))

    return height, width, quantity_images


def create_templates(images_dir, output_dir, height, width, quantity_images):
    # Computing the mean grayscale value of all signals per shape and the average of all masks per each shape
    template_gray = dict()
    template_mask = dict()

    for img_file in sorted(glob.glob(os.path.join(images_dir, '*.jpg'))):
        name = os.path.splitext(os.path.split(img_file)[1])[0]
        mask_file = '{}/mask/mask.{}.png'.format(images_dir, name)
        gt_file = '{}/gt/gt.{}.txt'.format(images_dir, name)
        img = imageio.imread(img_file, as_gray=True)
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
    fd = os.path.join(output_dir)
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


def template_generation(images_dir, output_dir):
    height, width, quantity_images = data_shape_analysis(images_dir)
    create_templates(images_dir, output_dir, height, width, quantity_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    template_generation(args.images_dir, args.output_dir)
    #We run it as template_generation('train_val/train/', 'shape_templates')







