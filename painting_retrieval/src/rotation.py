import cv2
import numpy as np
import imageio
import math
from scipy import ndimage
import os
import glob
import matplotlib.pyplot as plt


def compute_rotation(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi/180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2-y1, x2-x1))
        angles.append(angle)

    median_angle = np.median(angles)
    img_rot = ndimage.rotate(img, median_angle)

    return median_angle, img_rot


if __name__ == '__main__':

    for filename in glob.glob(os.path.join('../data/w5_devel_random/*.jpg')):
        image = imageio.imread(filename)
        ang, img_rot = compute_rotation(image)
        print(ang)
        plt.imshow(img_rot)
        plt.show()