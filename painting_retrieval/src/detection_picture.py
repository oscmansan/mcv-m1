import math
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage.filters import roberts, sobel, scharr, prewitt

from skimage.filters import threshold_local

import cv2
import numpy as np
import numpy.linalg as la
import PIL
#Gaussian blur
#Canny edge detection
#Finding Hough lines
#Extrapolating the intersection of the Hough lines to get corners
import os
import imageio

import glob
import matplotlib.pyplot as plt


def FindAngle(a,b,c):
    """
    find the angle between three points
    """
    ab=b-a
    ac=c-a
    dot=np.dot(ab, ac)
    norm_ab=la.norm(ab)
    norm_ac=la.norm(ac)
    cos = max(min(1, dot/(norm_ac*norm_ab)), -1)
    angle=math.acos(cos)*180/np.pi
    return angle


def SortCorners(points):
    """
    sort corners respect to center of mass counterclockwise
    """
    C=np.mean(points,axis=0)[0]
    angle=[]
    for i in range(len(points)):
        a=(points[i])[0]
        angle.append(math.atan2(a[1]-C[1],a[0]-C[0]))
    ID=np.argsort(angle)
    sorted_p=points[ID]
    #print points
    #print ID
    #print sorted_p
    return sorted_p


def IsSquare(vcti):
    """"
    check whether the shape is square (or rectangle)
    """
    angles=[]
    vct=SortCorners(vcti)
    s_vct=len(vct)
    for i in range (s_vct):
        angles.append(FindAngle((vct[i%s_vct])[0], (vct[(i-1)%s_vct])[0], (vct[(i+1)%s_vct])[0]))
        #print angles
    #print angles
    min = 80
    max = 100
    if min<=angles[0]<=max and min<=angles[1]<=max and min<=angles[2]<=max:
        ## tolerance 90(+/-)2
        return True
    return False


def box_in_image(box, img):
    h = img.shape[0]
    w = img.shape[1]
    for i in range(len(box)):
        box[i][0] = max(0, min(h-1, box[i][0]))
        box[i][1] = max(0, min(w - 1, box[i][1]))
    return box


def bbox_characteristics(cnt, gray):
    rect = cv2.minAreaRect(cnt)
    bbox_center = rect[0]
    img_center = (round(gray.shape[0] / 2), round(gray.shape[1] / 2))
    dist = math.sqrt(((bbox_center[0] - img_center[0]) ** 2) + ((bbox_center[1] - img_center[1]) ** 2))
    area = cv2.contourArea(cnt)
    return dist, area


def detect_frame(gray):
    """
    Returns the 4 coordinates for the edges of the frame bounding box
    :param img: grayscale image
    :return: list of 4 points (x, y) denoting the coordinates of the image
    """

    # Preprocessing of image to minimize non-target edges

    # 1: Dilation
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(gray, kernel, iterations=1)

    imS = cv2.resize(img_dilation, (960, 540))  # Resize image

    # 2: Median Blur Filter
    img_median_blur = cv2.medianBlur(img_dilation, ksize=7)
    imS = cv2.resize(img_median_blur, (960, 540))  # Resize image

    # 3: Shrinking and enlarging ??????????????


    # Canny edge detection with dilation
    canny_gradient = cv2.Canny(img_median_blur, threshold1=0, threshold2=50, apertureSize=3)
    imS = cv2.resize(canny_gradient, (960, 540))  # Resize image

    img_canny_dilation = cv2.dilate(canny_gradient, kernel, iterations=1)
    imS = cv2.resize(img_canny_dilation, (960, 540))  # Resize image
    #plt.imshow(imS)
    #plt.show()


    # Contour detection along the edges
    ret, thresh = cv2.threshold(img_canny_dilation, 50, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    screenCnt = []

    # pick the best contour
    for c in cnts:

        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            if IsSquare(approx):
                screenCnt.append(approx)

        elif len(approx) > 4:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            screenCnt.append(box)

    screenCntFinal = sorted(screenCnt, key=cv2.contourArea, reverse=True)[:5]

    frame_bbox = screenCntFinal[0]
    for i in range(len(screenCntFinal)-1):
        dist_i, area_i = bbox_characteristics(frame_bbox, gray)
        dist_next, area_next = bbox_characteristics(screenCntFinal[i+1], gray)

        if dist_next/area_next < dist_i/area_i or (area_next > area_i*0.8 and dist_next < dist_i*1.2):
            frame_bbox = screenCntFinal[i+1]
        #if area_next > area_i*0.9:
        #    if dist_next < dist_i*1.1:
        #        frame_bbox = screenCntFinal[i+1]
        #elif dist_next < dist_i*1.5 and area_next > area_i*0.6:
        #    frame_bbox = screenCntFinal[i + 1]

    return frame_bbox


def rotate_and_crop(img, bbox):
    rect = cv2.minAreaRect(bbox)
    box = cv2.boxPoints(rect)
    angle = rect[2]

    # rotate img
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box

    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]
    return angle, img_crop


def crop_picture(img, gray):
    frame_bbox = detect_frame(gray)
    angle, img_crop = rotate_and_crop(img, frame_bbox)

    return img_crop


if __name__ == '__main__':
    frame_bboxes = []
    for filename in glob.glob(os.path.join('../data/w5_devel_random/*.jpg')):
        print(filename)
        img = imageio.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        frame_bbox = detect_frame(gray)

        m2 = cv2.drawContours(img, [frame_bbox], -1, (0, 255, 0), 3)
        imS = cv2.resize(img, (960, 540))  # Resize image
        plt.imshow(imS)
        plt.show()

        angle, img_crop = rotate_and_crop(img, frame_bbox)
        plt.imshow(img_crop)
        plt.show()


