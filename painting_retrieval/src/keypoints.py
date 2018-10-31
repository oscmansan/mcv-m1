import cv2
from skimage import feature
import math
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


def sift_keypoints(image):
    """
        Extract keypoints of an image using Difference of Gaussians method.

        Args:
            image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

        Returns:
            list (list of object keypoint): list of keypoints.

        """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(gray, None)
    return keypoints


def laplacian_of_gaussian(image):
    """
            Extract keypoints of an image using Laplacian of Gaussians method.

            Args:
                image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

            Returns:
                (ndarray): list of keypoints coordinates.
                (ndarray): list of sizes -diameter- of keypoints
            """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blobs_log = feature.blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
    keypoints = np.array(blobs_log[:, [0, 1]], dtype=np.float32)
    sizes = np.array(list(blobs_log[:, 2] * np.sqrt(2) * 2), dtype=np.float32)
    return keypoints, sizes


def difference_of_gaussian(image):
    """
            Extract keypoints of an image using Difference of Gaussians method.

            Args:
                image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

            Returns:
                (ndarray): list of keypoints coordinates.
                (ndarray): list of sizes -diameter- of keypoints
            """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blobs_dog = feature.blob_dog(image_gray, max_sigma=30, threshold=.1)
    keypoints = np.array(blobs_dog[:, [0, 1]], dtype=np.float32)
    sizes = np.array(list(blobs_dog[:, 2] * np.sqrt(2) * 2), dtype=np.float32)
    return keypoints, sizes


def determinant_of_hessian(image):
    """
            Extract keypoints of an image using Determinant of Hessian method.

            Args:
                image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

            Returns:
                (ndarray): list of keypoints coordinates.
                (ndarray): list of sizes -diameter- of keypoints
            """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blobs_doh = feature.blob_doh(image_gray, max_sigma=30, threshold=.001)
    keypoints = np.array(blobs_doh[:, [0, 1]], dtype=np.float32)
    sizes = np.array(list(blobs_doh[:, 2] * np.sqrt(2) * 2), dtype=np.float32)
    return keypoints, sizes


def detect_keypoints(image, method):
    func = {
        'dog': difference_of_gaussian,
        'log': laplacian_of_gaussian,
        'doh': determinant_of_hessian,
        'sift': sift_keypoints
    }
    return func[method](image)
