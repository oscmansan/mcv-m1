from __future__ import division

import numpy as np
from scipy.fftpack import dct
import cv2


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


def descriptor_hsv(image):
    """
    Extract descriptors of an image in HSV color space.

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        ndarray: 1D array of type np.float32 containing image descriptors.

    """

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue_hist = cv2.calcHist([img_hsv], [0], None, [256], [0, 256])
    saturation_hist = cv2.calcHist([img_hsv], [1], None, [256], [0, 256])
    value_hist = cv2.calcHist([img_hsv], [2], None, [256], [0, 256])

    acum_hist = (hue_hist + saturation_hist + value_hist) / 3

    return np.array(acum_hist, dtype=np.float32).ravel()


def descriptor_rgb(image):
    """
    Extract descriptors of an image in RGB color space.

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        ndarray: 1D array of type np.float32 containing image descriptors.

    """

    r_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    b_hist = cv2.calcHist([image], [2], None, [256], [0, 256])

    acum_hist = (r_hist + g_hist + b_hist) / 3

    return np.array(acum_hist, dtype=np.float32).ravel()


def descriptor_Lab(image):
    """
    Extract descriptors of an image using the Lab color space.

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        histogram (ndarray): 1D array of type np.float32 containing the concatenated histograms for
                            L, a, b channels in this order.

    """

    # Convert image from RGB color space to Lab
    image_Lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # Select the number of bins
    bins = 256
    features = []
    # Compute histogram for every channel
    for i in range(3):
        hist = cv2.calcHist([image_Lab], [i], None, [bins], [0, 255])
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        features.extend(hist)
    # Retrieve the concatenation of channel histograms
    return np.array(features, dtype=np.float32)


def descriptor_CLD(image):
    """
    Extract descriptors of an image using the Color Layout Descriptor (CLD).
    The method is explained in 'Color Based Image Classification and Description' (Sergi Laencina - MS Thesis).

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        DCT main coefficients (ndarray): 1D array of type np.float32 containing the concatenated histograms for
                            L, a, b channels in this order.

    """
    # Convert image from RGB color space to YCbCr
    # Save size of original image
    N = image.shape[0]
    M = image.shape[1]
    # Define the window size
    windowsize_r = int(N / 8)
    windowsize_c = int(M / 8)

    # Create tiny image from 64 blocks grid in original image, and average color for block
    tiny_img = np.zeros(shape=(8, 8, 3))
    for channel in range(3):
        tr = 0
        for r in range(0, image.shape[0] - windowsize_r, windowsize_r):
            tc = 0
            for c in range(0, image.shape[1] - windowsize_c, windowsize_c):
                block = image[r:r + windowsize_r, c:c + windowsize_c][channel]
                block_color = np.average(block)
                tiny_img[tr][tc][channel] = block_color
                tc += 1
            tr += 1

    # Change tiny image to YCBCR color space
    Y = 0.299 * tiny_img[:, :, 0] + 0.587 * tiny_img[:, :, 1] + 0.114 * tiny_img[:, :, 2] - 128
    Cb = 0.169 * tiny_img[:, :, 0] - 0.331 * tiny_img[:, :, 1] + 0.500 * tiny_img[:, :, 2]
    Cr = 0.500 * tiny_img[:, :, 0] - 0.419 * tiny_img[:, :, 1] + 0.081 * tiny_img[:, :, 2]

    # Compute DCT coefficients for YCbCr channels
    DCT_Y = dct(dct(Y).T).T
    DCT_CB = dct(dct(Cb).T).T
    DCT_CR = dct(dct(Cr).T).T

    # Store DCT weighted coefficients as features
    features = [0, DCT_Y[0, 1], DCT_Y[1, 0], DCT_Y[2, 0], DCT_Y[1, 1], DCT_Y[0, 2], DCT_Y[0, 3], DCT_Y[1, 2],
                DCT_Y[2, 1], DCT_Y[0, 3], DCT_Y[0, 4], DCT_Y[1, 3], DCT_Y[2, 2], DCT_Y[3, 1], DCT_Y[4, 0],
                DCT_CB[0, 0], DCT_CB[0, 1], DCT_CB[1, 0], DCT_CB[2, 0], DCT_CB[1, 1], DCT_CB[0, 2], DCT_CB[0, 3],
                DCT_CB[1, 2], DCT_CB[2, 1], DCT_CB[0, 3], DCT_CB[0, 4], DCT_CB[1, 3], DCT_CB[2, 2], DCT_CB[3, 1],
                DCT_CB[4, 0],
                DCT_CR[0, 0], DCT_CR[0, 1], DCT_CR[1, 0], DCT_CR[2, 0], DCT_CR[1, 1], DCT_CR[0, 2], DCT_CR[0, 3],
                DCT_CR[1, 2], DCT_CR[2, 1], DCT_CR[0, 3], DCT_CR[0, 4], DCT_CR[1, 3], DCT_CR[2, 2], DCT_CR[3, 1],
                DCT_CR[4, 0]]

    return np.array(features, dtype=np.float32)
