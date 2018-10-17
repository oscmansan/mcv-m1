#!/usr/bin/env python3
import cv2

def to_integral_image(img):
    """
    Tranforms the given the image to integral image using the library cv2.
    :param img: Image to be tranformed in Integral image to be classified
    :type img: numpy.ndarray
    :return: The integral image.
    :rtype: numpy.ndarray
    """
    return cv2.integral(img)

def sum_region(integral_img, top_left, bottom_right):
    """
    Gets the area summation of  given integral image (numpy array) using given boundaries.
    :param integral_img: Integral image to be classified
    :type integral_img: numpy.ndarray
    :param top_left: y,x points in the top left of area bounding.
    :type top_left: tuple
    :param bottom_right: y,x points in the bottom right of area bounding.
    :type bottom_right: tuple
    :return: The area of boundins selected(crop integral image).
    :rtype: numpy.int32
    """
    top_left = (top_left[1], top_left[0])
    bottom_right = (bottom_right[1], bottom_right[0])
    if top_left == bottom_right:
        return integral_img[top_left]

    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    return integral_img[bottom_right] - integral_img[top_right] - integral_img[bottom_left] + integral_img[top_left]