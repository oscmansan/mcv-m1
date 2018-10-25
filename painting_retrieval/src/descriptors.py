from __future__ import division
import numpy as np


def descriptor(image):
    """
    Extract descriptors of an image.

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        ndarray: 1D array of type np.float32 containing image descriptors.

    """

    pass


def rgb_histogram(image):
    h, w, c = image.shape

    descriptors = []
    for i in range(c):
        hist = np.histogram(image[:, :, i], bins=256, range=(0, 255))[0]
        hist = hist / (h * w)  # normalize
        descriptors.append(hist)
    descriptors = np.concatenate(descriptors).astype(np.float32)

    return descriptors


def block_descriptor(image, descriptor_fn, num_blocks):
    h, w = image.shape[:2]
    block_h = int(np.ceil(h / num_blocks))
    block_w = int(np.ceil(w / num_blocks))

    descriptors = []
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            block = image[i:i+block_h, j:j+block_w]
            descriptors.append(descriptor_fn(block))
    descriptors = np.concatenate(descriptors).astype(np.float32)

    return descriptors


def pyramid_descriptor(image, descriptor_fn, max_level):
    descriptors = []
    for level in range(max_level+1):
        num_blocks = 2**level
        descriptors.append(block_descriptor(image, descriptor_fn, num_blocks))
    descriptors = np.concatenate(descriptors).astype(np.float32)

    return descriptors
