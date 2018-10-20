#!/usr/bin/env python3
import numpy as np
import imageio

import data_analysis as da
from integral_image_utils import sum_region

OBJECT_SIZE_MAX = 30000
OBJECT_SIZE_MIN = 460
OBJECT_FORMFACTOR_MAX = 1.4
OBJECT_FORMFACTOR_MIN = 0.44
FILLING_RATIOS = [0.49, 0.74, 1.0]

# load templates
CIRCLE_TEMPLATE = imageio.imread('data/templates/circle.png')
SQUARE_TEMPLATE = imageio.imread('data/templates/square.png')
TRIANGLE_TEMPLATE = imageio.imread('data/templates/triangle.png')
TRIANGLE_INV_TEMPLATE = imageio.imread('data/templates/triangle_inv.png')
TEMPLATES = [CIRCLE_TEMPLATE, SQUARE_TEMPLATE, TRIANGLE_TEMPLATE, TRIANGLE_INV_TEMPLATE]


def window_evaluation_features(pixel_candidates, bbox, fr_thresh=.1, sr_thresh=.01):
    # discard by size
    object_size = da.size(pixel_candidates, bbox)
    if (object_size < OBJECT_SIZE_MIN) or (object_size > OBJECT_SIZE_MAX):
        return False

    # discard by form factor
    object_ff = da.form_factor(bbox)
    if (object_ff < OBJECT_FORMFACTOR_MIN) or (object_ff > OBJECT_FORMFACTOR_MAX):
        return False

    # discard by filling ratio
    fr = da.filling_ratio(pixel_candidates, bbox)
    dist = min(abs(np.array(FILLING_RATIOS) - fr))
    if dist > fr_thresh:
        return False

    #discard by being part of a larger object detected
    tly, tlx, bry, brx = bbox
    tly2 =tly-1 if tly>0 else tly
    tlx2 =tlx-1 if tly>0 else tlx
    bry2 =bry+1 if tly>0 else bry
    brx2 =brx+1 if tly>0 else brx
    surround_bbox = [tly2, tlx2, bry2, brx2]
    object_surround_size = da.size(pixel_candidates, surround_bbox)
    if (object_surround_size-object_size)/object_size > sr_thresh:
        return False
    return True


def window_evaluation_integral_image(integral_image, bbox, fr_thresh=.1, sr_thresh=.01):

    # discard by size
    tly, tlx, bry, brx = bbox
    object_size = sum_region(integral_image, [tly, tlx], [bry,brx])
    if (object_size < OBJECT_SIZE_MIN) or (object_size > OBJECT_SIZE_MAX):
        return False

    # discard by filling ratio

    width = brx - tlx
    height = bry - tly
    bbox_area = width * height
    fr = object_size / bbox_area
    dist = min(abs(np.array(FILLING_RATIOS) - fr))
    if dist > fr_thresh:
        return False

    #discard by being part of a larger object detected
    tly, tlx, bry, brx = bbox
    tly2 =tly-1 if tly>0 else tly
    tlx2 =tlx-1 if tly>0 else tlx
    bry2 =bry+1 if tly>0 else bry
    brx2 =brx+1 if tly>0 else brx
    object_surround_size = sum_region(integral_image, [tly2, tlx2], [bry2,brx2])
    if (object_surround_size-object_size)/object_size > sr_thresh:
        return False
    return True


def correlation_coefficient(patch1, patch2):
    corr = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    norm_corr = corr / stds if stds != 0 else 0
    return norm_corr


def window_evaluation_template(im, bbox, template, corr_thresh=.6):
    tly, tlx, bry, brx = bbox
    window = im[tly:bry, tlx:brx]
    corr = correlation_coefficient(window, template)
    if abs(corr) > corr_thresh:
        return True
    return False


def window_evaluation_rand(pixel_candidates, bbox):
    return bool(np.random.binomial(1, 0.0001))  # returns True with probability 0.0001


def main():
    import os, glob
    from skimage.transform import resize
    from matplotlib import pyplot as plt

    img_file = np.random.choice(glob.glob('data/train_val/val/*.jpg'))
    name = os.path.splitext(os.path.split(img_file)[1])[0]
    mask_file = 'data/train/mask/mask.{}.png'.format(name)
    gt_file = 'data/train/gt/gt.{}.txt'.format(name)
    img = imageio.imread(img_file, as_gray=True).astype(np.uint8)
    mask = imageio.imread(mask_file)
    plt.figure(); plt.imshow(img * mask)

    gts = [line.split(' ') for line in open(gt_file, 'r').read().splitlines()]
    for gt in gts:
        bbox = np.round(list(map(int, map(float, gt[:4]))))
        box_h, box_w = bbox[2]-bbox[0], bbox[3]-bbox[1]
        #print(window_evaluation_features(mask, bbox))
        for template in TEMPLATES:
            template = resize(template, (box_h, box_w), preserve_range=True).astype(np.uint8)
            plt.figure(); plt.imshow(template)
            print(window_evaluation_template(img, bbox, template))
    plt.show()


if __name__ == '__main__':
    main()
