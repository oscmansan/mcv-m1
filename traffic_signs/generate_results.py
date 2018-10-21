#!/usr/bin/env python3
import os
import glob
import argparse
import pickle
import multiprocessing.dummy as mp

import numpy as np
import imageio

from candidate_generation_pixel import candidate_generation_pixel
from candidate_generation_window import candidate_generation_window


def worker(x):
    image_file, output_dir, pixel_method, window_method = x
    
    name = os.path.splitext(os.path.split(image_file)[1])[0]

    im = imageio.imread(image_file)
    print(image_file)
    
    pixel_candidates = candidate_generation_pixel(im, pixel_method)
    window_candidates = candidate_generation_window(im, pixel_candidates, window_method)
    
    fd = os.path.join(output_dir, '{}_{}'.format(pixel_method, window_method))
    if not os.path.exists(fd):
        os.makedirs(fd)
        
    out_mask_name = '{}/{}.png'.format(fd, name)
    imageio.imwrite(out_mask_name, pixel_candidates.astype(np.uint8))

    out_list_name = '{}/{}.pkl'.format(fd, name)
    with open(out_list_name, "wb") as fp:
        pickle.dump(window_candidates, fp)


def generate_masks(images_dir, output_dir, pixel_method, window_method):
    images = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
    
    with mp.Pool(processes=2) as p:
        p.map(worker, [(image_file, output_dir, pixel_method, window_method) for image_file in images])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir')
    parser.add_argument('output_dir')
    parser.add_argument('pixel_method')
    parser.add_argument('window_method')
    args = parser.parse_args()

    generate_masks(args.images_dir, args.output_dir, args.pixel_method, args.window_method)
