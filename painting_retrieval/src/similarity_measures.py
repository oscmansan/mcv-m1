from __future__ import division
import numpy as np


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
