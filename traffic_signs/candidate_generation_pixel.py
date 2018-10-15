#!/usr/bin/env python3
import numpy as np
import imageio
from skimage import color
import cv2

from color_utils import vrgb2ihsl as rgb2ihsl
from morphology_utils import fill_holes, filter_noise


def candidate_generation_pixel_normrgb(im):
    # convert input image to the normRGB color space
    normrgb_im = np.zeros(im.shape)
    eps_val = 0.00001
    norm_factor_matrix = im[:, :, 0] + im[:, :, 1] + im[:, :, 2] + eps_val

    normrgb_im[:, :, 0] = im[:, :, 0] / norm_factor_matrix
    normrgb_im[:, :, 1] = im[:, :, 1] / norm_factor_matrix
    normrgb_im[:, :, 2] = im[:, :, 2] / norm_factor_matrix

    # Develop your method here:
    # Example:
    pixel_candidates = normrgb_im[:, :, 1] > 100;

    return pixel_candidates


def candidate_generation_pixel_hsv(im):
    # convert input image to HSV color space
    hsv_im = color.rgb2hsv(im)

    # Develop your method here:
    # Example:
    pixel_candidates = hsv_im[:, :, 1] > 0.4;

    return pixel_candidates


# Create your own candidate_generation_pixel_xxx functions for other color spaces/methods
# Add them to the switcher dictionary in the switch_color_space() function
# These functions should take an image as input and output the pixel_candidates mask image


def candidate_generation_pixel_ihsl1(im):
    """
    Convert from RGB to IHLS and filter pixels by Euclidean distance to reference colors.
    Thresholds are dinamically computed using the global mean of the luminance.

    The method is explained in 'Color Detection And Segmentation For Road And Traffic Signs' (H. Fleyeh).
    """

    # convert input image to IHSL color space
    hsl = rgb2ihsl(im)
    H, S, L = (hsl[:, :, 0], hsl[:, :, 1], hsl[:, :, 2])

    # set color references as White, Red and Blue:
    SR = 0.815
    HR = 2.843

    SB = 0.6111
    HB = 0.0196

    # compute threshold given the mean lightness of the image
    m = len(L)
    n = len(L[0])
    suma = 0
    for i in range(m):
        for j in range(n):
            suma += L[i, j]
    mean = suma / (m * n)
    Nmean = mean / 256
    threshold = np.exp(-Nmean)

    # compute Euclidean distance from pixel of input image to color reference values (white, red and blue)
    Rx = (SR * np.cos(HR) - S * np.cos(H)) ** 2 + (SR * np.sin(HR) - S * np.sin(H)) ** 2
    Bx = (SB * np.cos(HB) - S * np.cos(H)) ** 2 + (SB * np.sin(HB) - S * np.sin(H)) ** 2
    Ed_R = np.nan_to_num(np.power(np.sqrt(Rx), 0.5))
    Ed_B = np.nan_to_num(np.power(np.sqrt(Bx), 0.5))

    # pixel candidates are those where the Euclidean distance is higher than the threshold
    Ed_R[Ed_R < threshold] = 0
    Ed_B[Ed_B < threshold] = 0

    pixel_candidates = Ed_R + Ed_B
    pixel_candidates[pixel_candidates > 0] = 1

    return pixel_candidates


def candidate_generation_pixel_ihsl2(im):
    """
    Convert from RGB to IHLS and normalize H and S channels. Then H and S values
    are discretized and pixels are filtered using the binary mask resulting from
    the logical AND between H and S channels.

    The method is explained in 'Color Detection And Segmentation For Road And Traffic Signs' (H. Fleyeh).
    """

    #convert input image to IHLS color space
    hsl = rgb2ihsl(im)
    H, S, L = (hsl[:, :, 0], hsl[:, :, 1], hsl[:, :, 2])
    size = im.shape #(n, m, channels)

    Hout = np.zeros((size[0], size[1]))
    Sout = np.zeros((size[0], size[1]))
    pixel_candidates = np.zeros((size[0], size[1]))

    #normalize S and H to [0, 255]

    S = 255*S/np.max(S)
    H = np.nan_to_num(H)
    H = 255*H/np.max(H)

    Smin = 51
    Smax = 170

    #the range of H defines the color (we want white, red and blue)
    #values obtained from hue histograms in train images, in color_ranges.py
    Hmin = [146, 2, 104]
    Hmax = [170, 7, 142]

    for i in range(size[0]):
        for j in range(size[1]):
            if S[i,j] <= Smin:
                Sout[i,j] = 0
            elif S[i,j] < Smax:
                Sout[i, j] = S[i, j]
            else:
                Sout[i, j] = 255
            for k in range(len(Hmin)):
                if H[i,j] >= Hmin[k] or H[i,j] <= Hmax[k]:
                    Hout[i, j] = max(255, Hout[i, j])

            pixel_candidates[i,j] = (Sout[i,j] and Hout[i,j])

    return pixel_candidates


def candidate_generation_pixel_hsv_euclidean(rgb):
    """
    Convert from RGB to HSV and filter pixels by Euclidean distance to reference
    values based on H and S. Thresholds are computed empirically.
    """

    ref_colors = [[0.0182, 0.6667, 1.0000], [0.6118, 0.6879, 1.0000]]
    thresholds = [0.25, 0.3]

    hsv = color.rgb2hsv(rgb)

    masks = []
    for ref, thresh in zip(ref_colors, thresholds):
        ref = np.tile(ref, hsv.shape[:2] + (1,))
        h1 = hsv[:,:,0]
        s1 = hsv[:,:,1]
        h2 = ref[:,:,0]
        s2 = ref[:,:,1]
        dist = ((h2-h1)**2 + (s2-s1)**2)**0.5
        mask = dist < thresh
        masks.append(mask)

    pixel_candidates = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for mask in masks:
        pixel_candidates += mask
    pixel_candidates = np.clip(pixel_candidates, 0, 1)
    return pixel_candidates


def candidate_generation_pixel_rgb(im):
    """
    Consider pixels in RGB space, pixels whose values are near the three most
    common RGB colors in the traffic signals are kept.
    """

    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    col0_candidates = (R > 19) & (R < 97) & (G > 61) & (G < 120) & (B > 94) & (B < 161);
    col1_candidates = (R > 143) & (R < 231) & (G > 186) & (G < 253) & (B > 178) & (B < 252);
    col2_candidates = (R > 18) & (R < 104) & (G > 23) & (G < 55) & (B > 23) & (B < 67);

    pixel_candidates = col0_candidates | col1_candidates | col2_candidates;

    return pixel_candidates

def morphological_filtering(mask):
    """
        Apply morphological operations to prepare pixel candidates to be selected as a traffic signal or not.
    """
    mask_filled = fill_holes(mask)
    mask_filtered = filter_noise(mask_filled)
    return mask_filtered

def candidate_generation_pixel_hsv_ranges(rgb):
    """
    Convert from RGB to HSV and filter pixels depending on whether they belong
    to a set of ranges. Ranges are computed empirically.
    """

    lower_red = np.array([[0, 100, 50], [10, 255, 255]])
    upper_red = np.array([[170, 100, 50], [179, 255, 255]])
    blue = np.array([[90, 100, 50], [135, 255, 255]])
    ranges = [lower_red, upper_red, blue]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h, w = hsv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in ranges:
        lowerb, upperb = r
        mask |= cv2.inRange(hsv, lowerb, upperb)
    mask = (mask / 255).astype(np.uint8)

    return mask


def candidate_generation_pixel_hsv_ranges(rgb):
    """
    Convert from RGB to HSV and filter pixels depending on whether they belong
    to a set of ranges. Ranges are computed empirically.
    """

    lower_red = np.array([[0, 100, 50], [10, 255, 255]])
    upper_red = np.array([[170, 100, 50], [179, 255, 255]])
    blue = np.array([[90, 100, 50], [135, 255, 255]])
    ranges = [lower_red, upper_red, blue]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h, w = hsv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in ranges:
        lowerb, upperb = r
        mask |= cv2.inRange(hsv, lowerb, upperb)
    mask = (mask / 255).astype(np.uint8)

    return mask


def morphological_filtering(mask):
    """
    Apply morphological operations to prepare pixel candidates to be selected as
    a traffic sign or not.
    """

    mask_filled = fill_holes(mask)
    mask_filtered = filter_noise(mask_filled)
    return mask_filtered


def switch_color_space(im, color_space):
    switcher = {
        'normrgb': candidate_generation_pixel_normrgb,
        'hsv': candidate_generation_pixel_hsv,
        'ihsl_1': candidate_generation_pixel_ihsl1,
        'ihsl_2': candidate_generation_pixel_ihsl2,
        'hsv_euclidean': candidate_generation_pixel_hsv_euclidean,
        'rgb': candidate_generation_pixel_rgb,
        'hsv_ranges': candidate_generation_pixel_hsv_ranges
    }

    # Get the function from switcher dictionary
    func = switcher.get(color_space, lambda: "Invalid color space")

    # Execute the function
    pixel_candidates = func(im)

    return pixel_candidates


def candidate_generation_pixel(im, color_space):
    mask = switch_color_space(im, color_space)
    pixel_candidates = morphological_filtering(mask)
    return pixel_candidates
