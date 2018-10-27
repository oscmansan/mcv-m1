from __future__ import division

import numpy as np
from scipy.spatial import distance
import cv2


def _distance(u, v):
    """
    Compare descriptor vectors based on a distance measure.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance between descriptor vectors.
    """

    pass


def correlation(u, v):
    """
    Compare histograms based on correlation.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance between histograms.
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_CORREL)


def chi_square(u, v):
    """
    Compare histograms based on the Chi-Square test.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance between histograms.
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_CHISQR)


def hellinguer_distance(u, v):
    """
    Compare histograms based on the Hellinguer distance.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance between histograms.
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_HELLINGER)


def bhattacharya_distance(u, v):
    """
    Compare histograms based on the Bhattacharyya distance.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance between histograms.
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_BHATTACHARYYA)


def intersection(u, v):
    """
    Compare histograms based on their intersection.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance between histograms.
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_INTERSECT)


def kl_divergence(u, v):
    """
    Compare histograms based on the Kullback-Leibler divergence.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance between histograms.
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_KL_DIV)


def euclidean_distance(u, v):
    """
    Compare descriptor vectors based on euclidian distance.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance between descriptor vectors.
    """

    return distance.euclidean(u/np.linalg.norm(u), v/np.linalg.norm(v))


def l1_distance(u, v):
    """
    Compare descriptor vectors based on L1 distance.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance between descriptor vectors.
    """

    return sum(abs(a - b) for a, b in zip(u, v))


def cosine_similarity(u, v):
    """
    Compare descriptor vectors based on the cosine similarity between two vectors.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: similarity score between 0 (less similar) and 1 (more similar).
    """

    def square_rooted(x):
        return round(np.sqrt(sum([a * a for a in x])), 3)

    numerator = np.dot(u, v)
    denominator = square_rooted(u) * square_rooted(v)
    return round(numerator / float(denominator), 3)


def cosine_distance(u, v):
    return distance.cosine(u, v)


def compute_distance(u, v, metric):
    func = {
        'euclidean_distance': euclidean_distance,
        'l1_distance': l1_distance,
        'cosine_distance': cosine_distance,
        'correlation': correlation,
        'chi_square': chi_square,
        'intersection': intersection,
        'hellinguer_distance': hellinguer_distance,
        'bhattacharya_distance': bhattacharya_distance,
        'kl_divergence': kl_divergence
    }
    return func[metric](u, v)


if __name__ == '__main__':
    u = [1, 2, 3, 4]
    v = [4, 3, 2, 1]
    print(euclidean_distance(u, v))
    print(l1_distance(u, v))
    print(cosine_distance(u, v))
