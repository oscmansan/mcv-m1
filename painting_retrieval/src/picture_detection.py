import os
import glob
import math
import cv2
import numpy as np
import numpy.linalg as la
import imageio
import matplotlib.pyplot as plt
import pickle


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
    min = 80
    max = 100
    if min<=angles[0]<=max and min<=angles[1]<=max and min<=angles[2]<=max:
        return True
    return False


def box_in_image(box, img):
    h = img.shape[1]
    w = img.shape[0]
    for i in range(len(box)):
        if len(box[i]) > 1:
            box[i][0] = max(0, min(h - 1, box[i][0]))
            box[i][1] = max(0, min(w - 1, box[i][1]))
        else:
            box[i][0][0] = max(0, min(h-1, box[i][0][0]))
            box[i][0][1] = max(0, min(w - 1, box[i][0][1]))
    return box


def bbox_characteristics(cnt, gray):
    rect = cv2.minAreaRect(cnt)
    bbox_center = rect[0]
    img_center = (round(gray.shape[1] / 2), round(gray.shape[0] / 2))
    dist = math.sqrt(((bbox_center[0] - img_center[0]) ** 2) + ((bbox_center[1] - img_center[1]) ** 2))
    area = cv2.contourArea(cnt)
    angle = rect[2]
    return dist, area, angle


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

    # 2: Median Blur Filter
    img_median_blur = cv2.medianBlur(img_dilation, ksize=7)

    # Canny edge detection with dilation
    canny_gradient = cv2.Canny(img_median_blur, threshold1=0, threshold2=50, apertureSize=3)

    img_canny_dilation = cv2.dilate(canny_gradient, kernel, iterations=1)

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

    frame_bbox = box_in_image(screenCntFinal[0], gray)
    dist_i, area_i, angle_i = bbox_characteristics(frame_bbox, gray)
    for i in range(len(screenCntFinal)-1):
        dist_next, area_next, angle_next = bbox_characteristics(box_in_image(screenCntFinal[i+1], gray), gray)
        if area_next > area_i*0.6 and (abs(angle_next) < abs(angle_i)*3 or dist_next < dist_i*0.6):
            if dist_next*abs(angle_next)/(2*area_next) < dist_i*abs(angle_i)/(2*area_i) or (dist_next < dist_i*1.2):
                frame_bbox = screenCntFinal[i+1]
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


def crop_picture(gray):
    frame_bbox = detect_frame(gray)
    angle, img_crop = rotate_and_crop(gray, frame_bbox)

    return img_crop


def create_results(bbox, results):
    rect = cv2.minAreaRect(bbox)
    box = cv2.boxPoints(rect)
    coords = []
    for i in range(4):
        coords.append((box[i][0], box[i][1]))
    angle = rect[2]
    if rect[1][0] > rect[1][1]:
        angle = 90 -angle
    else:
        angle = -angle
    if angle >= 90 and angle <= 120:
        angle = angle - 90
    elif angle >= 180:
        angle = angle - 180
    elif angle >= 70 and angle < 90:
        angle = angle + 90
    results.append([angle, coords])
    return results


def save_results(results):
    if not os.path.exists('../frame_results/'):
        os.makedirs('../frame_results/')

    results_fn = os.path.join('../frame_results/frames.pkl')
    print('Saving results to {}'.format(results_fn))
    with open(results_fn, "wb") as fp:  # Pickling
        pickle.dump(results, fp)


if __name__ == '__main__':
    results = []
    for filename in glob.glob(os.path.join('../data/w5_devel_random/*.jpg')):
        print(filename)
        img = imageio.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        frame_bbox = detect_frame(gray)

        #m2 = cv2.drawContours(img, [frame_bbox], -1, (0, 255, 0), 3)
        #imS = cv2.resize(img, (960, 540))  # Resize image
        #plt.imshow(imS)
        #plt.show()

        angle, img_crop = rotate_and_crop(img, frame_bbox)
        plt.imshow(img_crop)
        plt.show()

        results = create_results(frame_bbox, results)
    print(results)
    save_results(results)
