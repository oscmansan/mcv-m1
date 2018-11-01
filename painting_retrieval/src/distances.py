from __future__ import division

import cv2
import numpy as np
from scipy.spatial import distance


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

    return 1 - cv2.compareHist(u, v, cv2.HISTCMP_CORREL)


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
    return 1 - cv2.compareHist(u, v, cv2.HISTCMP_INTERSECT)


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

    return distance.euclidean(u, v)


def l1_distance(u, v):
    """
    Compare descriptor vectors based on L1 distance.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance between descriptor vectors.
    """

    #return float(np.sum(abs(a - b) for a, b in zip(u, v)))
    return distance.minkowski(u, v, 1)


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
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
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


def bf_match(query_des, image_des, distance_metric):
    norm_type = {
        'l1': cv2.NORM_L1,
        'l2': cv2.NORM_L2,
        'hamming': cv2.NORM_HAMMING,
        'hamming2': cv2.NORM_HAMMING2
    }[distance_metric]

    bf = cv2.BFMatcher(normType=norm_type)
    # For each image descriptor, find best k matches among query descriptors
    matches = bf.knnMatch(image_des, query_des, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    distances = [match.distance for match in good]
    return np.mean(distances)


def match_descriptors(query_des, image_des, method, distance_metric):
    func = {
        'brute_force': bf_match
    }
    return func[method](query_des, image_des, distance_metric)
