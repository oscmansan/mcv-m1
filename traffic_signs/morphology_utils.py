#!/usr/bin/env python3
import glob
import collections

import imageio
import numpy as np
import cv2
from matplotlib import pyplot as plt

def fill_holes(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 7)
    return mask

def filter_noise(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 7)
    return mask

def granulometry(mask, steps, dict_kernels):
    g_curve = np.zeros(steps)
    g_curve[0] = 0
    for i in range(steps-1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i+1, i+1))
        remain = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        g_curve[i+1] = np.sum(np.abs(remain))
    pecstrum = np.gradient(g_curve)
    pecstrum = np.array(pecstrum)
    for i in range(3):
        peak = np.where(pecstrum==max(pecstrum))[0][0]
        if peak in dict_kernels.keys():
            dict_kernels[peak]+=1
        else:
            dict_kernels[peak]=1
        pecstrum[peak] = min(pecstrum)
    return dict_kernels

def main():
    max_size = 30
    dict_kernels = collections.defaultdict()
    for mask_file in sorted(glob.glob('results/ihsl_1/00.002254.png')):
        print(mask_file)
        mask = imageio.imread(mask_file)
        plt.imshow(filter_noise(fill_holes(mask))*255)
        dict_kernels = granulometry(mask, max_size, dict_kernels)
    print(list(dict_kernels.values()))
    plt.hist(list(dict_kernels.values()))
    #print(plt.keys())
    plt.show()

if __name__ == '__main__':
    main()