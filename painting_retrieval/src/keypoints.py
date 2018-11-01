import cv2
from skimage import feature
import numpy as np


def _keypoints(image):
    """
    Extract keypoints of an image.

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        list: list of keypoints.

    """

    pass


def sift(image):
    """
    Extract keypoints of an image using Difference of Gaussians method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.

    Returns:
        list (list of object keypoint): list of keypoints.

    """

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(image)
    return keypoints


def laplacian_of_gaussian(image):
    """
    Extract keypoints of an image using Laplacian of Gaussians method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.

    Returns:
        list (list of object keypoint): list of keypoints.

    """

    blobs_log = feature.blob_log(image, max_sigma=30, num_sigma=10, threshold=.1)
    points2f = np.array(blobs_log[:, [0, 1]], dtype=np.float32)
    sizes = np.array(list(blobs_log[:, 2] * np.sqrt(2) * 2), dtype=np.float32)
    keypoints = []
    for i in range(len(points2f)):
        keypoints.append(cv2.KeyPoint_convert([points2f[i]], sizes[i])[0])
    return keypoints


def difference_of_gaussian(image):
    """
    Extract keypoints of an image using Difference of Gaussians method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.

    Returns:
        list (list of object keypoint): list of keypoints.

    """

    blobs_dog = feature.blob_dog(image, max_sigma=30, threshold=.1)
    points2f = np.array(blobs_dog[:, [0, 1]], dtype=np.float32)
    sizes = np.array(list(blobs_dog[:, 2] * np.sqrt(2) * 2), dtype=np.float32)
    keypoints = []
    for i in range(len(points2f)):
        keypoints.append(cv2.KeyPoint_convert([points2f[i]], sizes[i])[0])
    return keypoints


def determinant_of_hessian(image):
    """
    Extract keypoints of an image using Determinant of Hessian method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.

    Returns:
        list (list of object keypoint): list of keypoints.

    """

    blobs_doh = feature.blob_doh(image, max_sigma=30, threshold=.001)
    points2f = np.array(blobs_doh[:, [0, 1]], dtype=np.float32)
    sizes = np.array(list(blobs_doh[:, 2] * np.sqrt(2) * 2), dtype=np.float32)
    keypoints = []
    for i in range(len(points2f)):
        keypoints.append(cv2.KeyPoint_convert([points2f[i]], sizes[i])[0])
    return keypoints


def detect_keypoints(image, method):
    func = {
        'dog': difference_of_gaussian,
        'log': laplacian_of_gaussian,
        'doh': determinant_of_hessian,
        'sift': sift
    }
    return func[method](image)
