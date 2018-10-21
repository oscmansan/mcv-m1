#!/usr/bin/env python3
"""
Usage:
  traffic_sign_detection.py <dirName> <outPath>
  traffic_sign_detection.py -h | --help
Options:
  --windowMethod=<wm>        Window method       [default: None]
"""


import fnmatch
import os
import sys
import pickle
import time

import numpy as np
import imageio
from docopt import docopt

from candidate_generation_pixel import candidate_generation_pixel
from candidate_generation_window import candidate_generation_window
from evaluation.load_annotations import load_annotations
import evaluation.evaluation_funcs as evalf

def traffic_sign_detection(directory, output_dir, pixel_method, window_method):

    pixelTP  = 0
    pixelFN  = 0
    pixelFP  = 0
    pixelTN  = 0

    windowTP = 0
    windowFN = 0
    windowFP = 0

    window_precision = 0
    window_accuracy  = 0

    # Load image names in the given directory
    file_names = sorted(fnmatch.filter(os.listdir(directory), '*.jpg'))

    pixel_time = 0
    for name in file_names:
        base, extension = os.path.splitext(name)

        # Read file
        im = imageio.imread('{}/{}'.format(directory,name))
        print ('{}/{}'.format(directory,name))

        # Candidate Generation (pixel) ######################################
        start = time.time()
        pixel_candidates = candidate_generation_pixel(im, pixel_method)
        end = time.time()
        pixel_time += (end - start)

        fd = '{}/{}_{}'.format(output_dir, pixel_method, window_method)
        if not os.path.exists(fd):
            os.makedirs(fd)

        out_mask_name = '{}/{}.png'.format(fd, base)
        imageio.imwrite (out_mask_name, np.uint8(np.round(pixel_candidates)))

        if window_method != 'None':
            window_candidates = candidate_generation_window(im, pixel_candidates, window_method)

            out_list_name = '{}/{}.pkl'.format(fd, base)

            with open(out_list_name, "wb") as fp:   #Pickling
                pickle.dump(window_candidates, fp)

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    images_dir = args['<dirName>']  # Directory with input images and annotations. For instance, '../../DataSetDelivered/test'
    output_dir = args['<outPath>']  # Directory where to store output masks, etc. For instance '~/m1-results/week1/test'
    pixel_method = 'hsv_ranges'
    window_method = 'None'
    methods_pixels = ['normrgb','hsv','ihsl_1','ihsl_2','hsv_euclidean','rgb','hsv_ranges']
    methods_windows = ['ccl_features','ccl_template','sw_features','sw_template','sw_feature_integral']

    for method in methods_pixels:
        print(method)
        traffic_sign_detection(images_dir, output_dir, method, window_method);
    for method in methods_windows:
        print(method)
        traffic_sign_detection(images_dir, output_dir, pixel_method, method);