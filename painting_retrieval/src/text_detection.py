import glob
import random

import numpy as np
import imutils
import cv2

from matplotlib import pyplot as plt


def imshow(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


image = random.choice(glob.glob('../data/w5_BBDD_random/*.jpg'))
#image = '../data/w5_BBDD_random/ima_000075.jpg'
#image = '../data/w5_BBDD_random/ima_000124.jpg'
#image = '../data/w5_BBDD_random/ima_000153.jpg'
#image = '../data/w5_BBDD_random/ima_000059.jpg'
print(image)
img = cv2.imread(image)
img = imutils.resize(img, width=512)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
imshow(tophat)
blackhat = cv2.cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
imshow(blackhat)

tophat = tophat if np.sum(tophat) > np.sum(blackhat) else blackhat

thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
imshow(thresh)

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
imshow(thresh)

contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    area = cv2.contourArea(cnt)
    rect_area = w * h
    extent = area / rect_area

    if extent > 0.2 and h > 10 and 2 < w / h:
        boxes.append((x, y, x + w, y + h))

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for bbox in boxes:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0))
imshow(rgb)
