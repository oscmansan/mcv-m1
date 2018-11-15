from __future__ import division

import cv2
import numpy as np
from scipy.spatial import distance
import nmslib


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


def _filter_matches(matches, ratio=0.5):
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def _compute_similarity_score(matches, matches_thresh=10, dist_thresh=920):
    m = len(matches)
    d = np.mean([match.distance for match in matches]) if m > 0 else np.inf
    if m < matches_thresh or d > dist_thresh:
        return 0
    else:
        return m / d


def bf_match(query_des, image_des, distance_metric):
    norm_type = {
        'l1': cv2.NORM_L1,
        'l2': cv2.NORM_L2,
        'hamming': cv2.NORM_HAMMING,
        'hamming2': cv2.NORM_HAMMING2
    }
    bf = cv2.BFMatcher(normType=norm_type[distance_metric])

    # For each image descriptor, find best k matches among query descriptors
    matches = bf.knnMatch(image_des, query_des, k=2)
    good = _filter_matches(matches)
    score = _compute_similarity_score(good)

    return score


def flann_match(query_des, image_des, distance_metric):
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # For each image descriptor, find best k matches among query descriptors
    matches = flann.knnMatch(image_des, query_des, k=2)
    good = _filter_matches(matches)
    score = _compute_similarity_score(good)

    return score


def nmslib_match(query_des, image_des, distance_metric):
    index = nmslib.init(method='hnsw', space='l2sqr_sift', data_type=nmslib.DataType.DENSE_UINT8_VECTOR, dtype=nmslib.DistType.INT)
    index.addDataPointBatch(query_des.astype(np.uint8))
    index.createIndex({'M': 16, 'efConstruction': 100})
    index.setQueryTimeParams({'efSearch': 100})

    # For each image descriptor, find best k matches among query descriptors
    matches = index.knnQueryBatch(image_des.astype(np.uint8), k=2)
    matches = [[cv2.DMatch(query_idx, train_idx, distance) for train_idx, distance in zip(*match)] for query_idx, match in enumerate(matches)]
    good = _filter_matches(matches)
    score = _compute_similarity_score(good)

    return score


def match_descriptors(query_des, image_des, method, distance_metric):
    func = {
        'brute_force': bf_match,
        'flann': flann_match,
        'nmslib': nmslib_match
    }
    return func[method](query_des, image_des, distance_metric)
