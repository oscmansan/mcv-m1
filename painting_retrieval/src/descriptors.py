from __future__ import division

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.fftpack import dct
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import gabor_kernel
from sklearn.cluster import KMeans


def _descriptors(image):
    """
    Extract descriptors of an image.

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        list: list of 1D arrays of type np.float32 containing image descriptors.

    """

    pass


def block_descriptor(image, descriptor_fn, num_blocks):
    h, w = image.shape[:2]
    block_h = int(np.ceil(h / num_blocks))
    block_w = int(np.ceil(w / num_blocks))

    descriptors = []
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            block = image[i:i + block_h, j:j + block_w]
            descriptors.extend(descriptor_fn(block))

    return descriptors


def pyramid_descriptor(image, descriptor_fn, max_level):
    descriptors = []
    for level in range(max_level + 1):
        num_blocks = 2 ** level
        descriptors.extend(block_descriptor(image, descriptor_fn, num_blocks))

    return descriptors


def rgb_histogram(image):
    h, w, c = image.shape

    descriptors = []
    for i in range(c):
        hist = np.histogram(image[:, :, i], bins=256, range=(0, 255))[0]
        hist = hist / (h * w)  # normalize
        descriptors.append(np.array(hist, dtype=np.float32))

    return descriptors


def hsv_histogram(image):
    h, w, c = image.shape
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    sizes = [180, 256, 256]
    ranges = [(0, 180), (0, 256), (0, 256)]
    descriptors = []
    for i in range(c):
        hist = cv2.calcHist([hsv], [i], None, [sizes[i]], ranges[i]).ravel()
        hist = hist / (h * w)  # normalize
        descriptors.append(np.array(hist, dtype=np.float32))

    return descriptors


def lab_histogram(image):
    """
    Extract descriptors of an image using the Lab color space.

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        list: list of 1D arrays of type np.float32 containing the histograms for
            L, a, b channels in this order.

    """

    # Convert image from RGB color space to Lab
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # Select the number of bins
    bins = 256
    descriptors = []
    # Compute histogram for every channel
    for i in range(3):
        hist = cv2.calcHist([image_lab], [i], None, [bins], [0, 256]).ravel()
        cv2.normalize(hist, hist)
        descriptors.append(np.array(hist, dtype=np.float32))

    # Retrieve the concatenation of channel histograms
    return descriptors


def ycrcb_histogram(image):
    """
    Extract descriptors of an image using its histogram in the YCbCr space.

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        list: list of 1D arrays of type np.float32 containing image descriptors,
            which correspond to the three channel histograms (Y, Cr and Cb).

    """

    bins = 256
    descriptors = []
    imageYCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    for i in range(3):
        hist = cv2.calcHist([imageYCrCb], [i], None, [bins], [0, 256]).ravel()
        cv2.normalize(hist, hist)
        descriptors.append(np.array(hist, dtype=np.float32))

    return descriptors


def cld(image):
    """
    Extract descriptors of an image using the Color Layout Descriptor (CLD).
    The method is explained in 'Color Based Image Classification and Description'
    (Sergi Laencina - MS Thesis).

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        ndarray: list with a 1D array of type np.float32 containing the DCT coefficients.

    """

    # Convert image from RGB color space to YCbCr
    # Save size of original image
    N = image.shape[0]
    M = image.shape[1]
    # Define the window size
    windowsize_r = int(N / 8)
    windowsize_c = int(M / 8)

    # Create tiny image from 64 blocks grid in original image, and average color for block
    tiny_img = np.zeros(shape=(8, 8, 3))
    for channel in range(3):
        tr = 0
        for r in range(0, image.shape[0] - windowsize_r, windowsize_r):
            tc = 0
            for c in range(0, image.shape[1] - windowsize_c, windowsize_c):
                block = image[r:r + windowsize_r, c:c + windowsize_c][channel]
                block_color = np.average(block)
                tiny_img[tr][tc][channel] = block_color
                tc += 1
            tr += 1

    # Change tiny image to YCBCR color space
    Y = 0.299 * tiny_img[:, :, 0] + 0.587 * tiny_img[:, :, 1] + 0.114 * tiny_img[:, :, 2] - 128
    Cb = 0.169 * tiny_img[:, :, 0] - 0.331 * tiny_img[:, :, 1] + 0.500 * tiny_img[:, :, 2]
    Cr = 0.500 * tiny_img[:, :, 0] - 0.419 * tiny_img[:, :, 1] + 0.081 * tiny_img[:, :, 2]

    # Compute DCT coefficients for YCbCr channels
    DCT_Y = dct(dct(Y).T).T
    DCT_CB = dct(dct(Cb).T).T
    DCT_CR = dct(dct(Cr).T).T

    # Store DCT weighted coefficients as features
    features = [0, DCT_Y[0, 1], DCT_Y[1, 0], DCT_Y[2, 0], DCT_Y[1, 1], DCT_Y[0, 2], DCT_Y[0, 3], DCT_Y[1, 2],
                DCT_Y[2, 1], DCT_Y[0, 3], DCT_Y[0, 4], DCT_Y[1, 3], DCT_Y[2, 2], DCT_Y[3, 1], DCT_Y[4, 0],
                DCT_CB[0, 0], DCT_CB[0, 1], DCT_CB[1, 0], DCT_CB[2, 0], DCT_CB[1, 1], DCT_CB[0, 2], DCT_CB[0, 3],
                DCT_CB[1, 2], DCT_CB[2, 1], DCT_CB[0, 3], DCT_CB[0, 4], DCT_CB[1, 3], DCT_CB[2, 2], DCT_CB[3, 1],
                DCT_CB[4, 0],
                DCT_CR[0, 0], DCT_CR[0, 1], DCT_CR[1, 0], DCT_CR[2, 0], DCT_CR[1, 1], DCT_CR[0, 2], DCT_CR[0, 3],
                DCT_CR[1, 2], DCT_CR[2, 1], DCT_CR[0, 3], DCT_CR[0, 4], DCT_CR[1, 3], DCT_CR[2, 2], DCT_CR[3, 1],
                DCT_CR[4, 0]]

    return [np.array(features, dtype=np.float32)]


def dominant_colors_rgb(image, k=5):
    pixels = image.reshape((-1, 3))

    clt = KMeans(n_clusters=k)
    clt.fit(pixels)
    clusters = clt.cluster_centers_

    return [clusters.ravel().astype(np.float32)]


def glcm_texture(image):
    """
    Extract texture descriptors of an image by extracting properites from
     its gray-level co-occurrence matrix in 4 different directions (vertical, horizontal, left and right diagonals).

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        ndarray: 1D array of type np.float32 containing image descriptors, which correspond to the concatenation
            of the three channel histograms (Y, Cr and Cb).

    """
    # Convert image from RGB to Grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Compute grey-level co-ocurrence matrix
    glcm = greycomatrix(imageGray, distances=[1],
                        angles=[0, np.pi / 8, np.pi / 4, 3 * np.pi / 8],
                        levels=256, symmetric=True, normed=True)

    # Calculate texture properties of GLCM
    contrast = greycoprops(glcm, 'contrast').mean(axis=1)
    dissimilarity = greycoprops(glcm, 'dissimilarity').mean(axis=1)
    homogeneity = greycoprops(glcm, 'homogeneity').mean(axis=1)
    ASM = greycoprops(glcm, 'ASM').mean(axis=1)
    energy = greycoprops(glcm, 'energy').mean(axis=1)
    correlation = greycoprops(glcm, 'correlation').mean(axis=1)

    text_features = [contrast, dissimilarity, homogeneity, ASM, energy, correlation]
    return np.array(text_features, dtype=np.float32).ravel()


def gabor_texture(image):
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    shrink = (slice(0, None, 3), slice(0, None, 3))
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(np.float32(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))[shrink], kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()

    return feats.ravel().astype(np.float32)


def extract_global_descriptors(image, method):
    func = {
        'rgb_histogram': rgb_histogram,
        'hsv_histogram': hsv_histogram,
        'lab_histogram': lab_histogram,
        'ycrcb_histogram': ycrcb_histogram,
        'cld': cld,
        'rgb_histogram_pyramid': lambda image: pyramid_descriptor(image, rgb_histogram, 2),
        'hsv_histogram_pyramid': lambda image: pyramid_descriptor(image, hsv_histogram, 2),
        'lab_histogram_pyramid': lambda image: pyramid_descriptor(image, lab_histogram, 2),
        'ycrcb_histogram_pyramid': lambda image: pyramid_descriptor(image, ycrcb_histogram, 2),
        'gabor': gabor_texture,
        'glcm': glcm_texture
    }
    return func[method](image)


def sift_descriptors(image, keypoints):
    sift = cv2.xfeatures2d.SIFT_create()
    _, descriptors = sift.compute(image, keypoints)
    return descriptors


def extract_local_descriptors(image, keypoints, method):
    func = {
        'sift': sift_descriptors
    }
    return func[method](image, keypoints)
