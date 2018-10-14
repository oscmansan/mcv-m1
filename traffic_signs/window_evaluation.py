import os,glob
import data_analysis as da
import numpy as np
import imageio
import cv2
from matplotlib import pyplot as plt

SIZE_MAX = 30000
SIZE_MIN = 460
FILLING_RATIOS = [0.49, 0.74, 1]


def simple_features_evaluator(mask, bbox):
    window_size = da.size(mask, bbox)
    window_filling_ratio = da.filling_ratio(mask, bbox)
    threshold = 0.1
    dist = min(abs(FILLING_RATIOS - window_filling_ratio))
    if dist < threshold and SIZE_MIN<window_size<SIZE_MAX:
        return True
    return False

def haar_cascade_classifier(window):
    return False

def template_matching_comparer(mask, template, bbox, method):
    window = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    threshold = 0.8
    matched = cv2.matchTemplate(window, template, method)  # returns float32
    if matched > threshold:
        return True
    return False

def main():
    for img_file in sorted(glob.glob('data/train/*.jpg')):
        name = os.path.splitext(os.path.split(img_file)[1])[0]
        mask_file = 'data/train/mask/mask.{}.png'.format(name)
        gt_file = 'data/train/gt/gt.{}.txt'.format(name)
        img = imageio.imread(img_file)
        mask = imageio.imread(mask_file)
        gts = [line.split(' ') for line in open(gt_file, 'r').read().splitlines()]
        for gt in gts:
            bbox = np.round(list(map(int, map(float, gt[:4]))))
            print(simple_features_evaluator(mask, bbox))

if __name__ == '__main__':
    main()