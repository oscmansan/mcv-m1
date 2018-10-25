from __future__ import division

import numpy as np
from scipy.spatial import distance
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

def euclidean_similarity(u, v):
    """
    Compare descriptor vectors based on the euclidian distance between two vectors.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0 (less similar) and 1 (more similar).
    """

    sim = 1 - distance.euclidean(u/np.linalg.norm(u), v/np.linalg.norm(v))

    return sim


def L1_distance(u, v):
    """
    Compare descriptor vectors based on L1 distance.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance score >= 0. If 0, u and v are the same. The higher the result, the more different are the
        two vectors.
    """

    return sum(abs(a - b) for a, b in zip(u, v))


def square_rooted(x):
    """
    Auxiliar function for calculating the cosine similarity between two vectors
    :param x: 1D array of type np.float32 containing image descriptors
    :return: square root of x
    """
    return round(np.sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(u, v):
    """
    Compare descriptor vectors based on the cosine similarity between two vectors.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0 (less similar) and 1 (more similar).
    """

    numerator = np.dot(u, v)
    denominator = square_rooted(u) * square_rooted(v)
    return round(numerator / float(denominator), 3)


if __name__ == '__main__':

    u = [1, 2, 3, 4]
    v = [4, 3, 2, 1]
    print(euclidean_similarity(u,v))
    print(L1_distance(u, v))
    print(cosine_similarity(u, v))