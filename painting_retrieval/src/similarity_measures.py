from __future__ import division

import numpy as np
import cv2


def similarity(u, v):
    """
    Compare descriptor vectors based on a similarity measure.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0 and 1.
    """

    pass


def euclidean_distance(u, v):
    return np.sum((u - v) ** 2) ** 0.5


def similarity_correlation(u, v):
    """
    Compare histograms based on a similarity correlation.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0 and 1.
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_CORREL)


def similarity_chi_square(u, v):
    """
    Compare histograms based on a similarity chi-square.

    Args:
        u (ndarray): Expected 1D array of type np.float32 containing image descriptors.
        v (ndarray): Observed 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0 (equal) and >0 (high values mean more difference).
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_CHISQR)


def similarity_hellinguer(u, v):
    """
    Compare histograms based on a similarity Hellinguer.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0 and 1.
    """

    return 1 - cv2.compareHist(u, v, cv2.HISTCMP_HELLINGER)


def similarity_bhattacharya(u, v):
    """
    Compare histograms based on Bhattacharyya distance.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0 (equal) and >0 (high values mean more difference).
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_BHATTACHARYYA)


def similarity_intersect(u, v):
    """
    Compare histograms based on intersection between histograms.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0 and 1.
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_INTERSECT)
