from __future__ import division
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
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
    Compare descriptor vectors based on a similarity correlation.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0 and 1.
    """
    return cv2.compareHist(u, v, cv2.HISTCMP_CORREL)


def similarity_chi_square(u, v):
    """
    Compare descriptor vectors based on a similarity chi-square.

    Args:
        u (ndarray): Expected 1D array of type np.float32 containing image descriptors.
        v (ndarray): Observed 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0(equal) and >0(high values means more difference).
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_CHISQR)


def similarity_hellinguer(u, v):
    """
    Compare descriptor vectors based on a similarity hellinguer.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0 and 1.
    """

    return 1 - cv2.compareHist(u, v, cv2.HISTCMP_HELLINGER)


def similarity_Bhattacharya(u, v):
    """
    Compare descriptor vectors based on Bhattacharyya distance.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0(equal) and >0(high values means more difference).
    """
    return cv2.compareHist(u, v, cv2.HISTCMP_BHATTACHARYYA)


def similarity_Intersect(u, v):
    """
    Compare descriptor vectors based on intersection between histograms.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0 and 1.
    """
    return cv2.compareHist(u, v, cv2.HISTCMP_INTERSECT)


def similarity_dominant_color(image_expected, image_observed, k=4, image_processing_size=None):
    """
    Compare descriptor vectors based on a similarity hellinguer.

    Args:
        image_expected (ndarray): (H x W x C) 3D array of type np.uint8 containing an expected image.
        image_observed (ndarray): (H x W x C) 3D array of type np.uint8 containing an observed image.
        k (int): Number of clusters.
        image_processing_size (ndarray): Size to rescale images.

    Returns:
        float: similarity score between 0 and 1.
    """

    if image_processing_size is not None:
        image_expected = cv2.resize(image_expected, image_processing_size, interpolation=cv2.INTER_AREA)
        image_observed = cv2.resize(image_observed, image_processing_size, interpolation=cv2.INTER_AREA)

        image_expected = image_expected.reshape((image_expected.shape[0] * image_expected.shape[1], 3))
        image_observed = image_observed.reshape((image_observed.shape[0] * image_observed.shape[1], 3))

    clt_expected = KMeans(n_clusters=k)
    clt2_observed = KMeans(n_clusters=k)

    labels_expected = clt_expected.fit_predict(image_expected)
    labels_observed = clt2_observed.fit_predict(image_observed)

    label_counts_expected = Counter(labels_expected)
    label_counts_observed = Counter(labels_observed)

    dominant_color_expected = clt_expected.cluster_centers_[label_counts_expected.most_common(1)[0][0]]
    dominant_color_observed = clt2_observed.cluster_centers_[label_counts_observed.most_common(1)[0][0]]

    dominant_color_expected = cv2.cvtColor(np.uint8([[dominant_color_expected]]), cv2.COLOR_RGB2Lab)
    dominant_color_observed = cv2.cvtColor(np.uint8([[dominant_color_observed]]), cv2.COLOR_RGB2Lab)
    dominant_color_expected = np.float32(dominant_color_expected)
    dominant_color_observed = np.float32(dominant_color_observed)
    dominant_color_expected *= 1 / 255;
    dominant_color_observed *= 1 / 255;

    return 1 - np.linalg.norm(dominant_color_expected - dominant_color_observed)
