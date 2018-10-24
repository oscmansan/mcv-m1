import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def descriptor(image):
    """
    Extract descriptors of an image.

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        ndarray: 1D array of type np.float32 containing image descriptors.

    """

    pass

def descriptor_hsv(image):
    """
    Extract descriptors of an image in HSV color space.

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        ndarray: 1D array of type np.float32 containing image descriptors.

    """

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue_hist = cv2.calcHist([img_hsv],[0],None,[256],[0,256])
    saturation_hist = cv2.calcHist([img_hsv],[1],None,[256],[0,256])
    value_hist = cv2.calcHist([img_hsv],[2],None,[256],[0,256])

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

    r_hist = cv2.calcHist([image],[0],None,[256],[0,256])
    g_hist = cv2.calcHist([image],[1],None,[256],[0,256])
    b_hist = cv2.calcHist([image],[2],None,[256],[0,256])

    acum_hist = (r_hist + g_hist + b_hist) / 3

    return np.array(acum_hist, dtype=np.float32).ravel()
