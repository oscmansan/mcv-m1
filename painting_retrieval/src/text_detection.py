import glob
import argparse
import pickle
import random
import multiprocessing.dummy as mp

import numpy as np
import imutils
import cv2

from matplotlib import pyplot as plt


def imshow(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    #imshow(tophat)
    blackhat = cv2.cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    #imshow(blackhat)

    tophat = tophat if np.sum(tophat) > np.sum(blackhat) else blackhat

    thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #imshow(thresh)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    #imshow(thresh)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)
        rect_area = w * h
        extent = area / rect_area

        if extent > 0.2 and h > 10 and 2 < w / h:
            boxes.append((x, y, x + w, y + h))

    return boxes


def bbox_iou(bboxA, bboxB):
    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[0], bboxB[0])
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    iou = interArea / float(bboxAArea + bboxBArea - interArea)

    # return the intersection over union value
    return iou


def correct_boxes(boxes, orig_h, orig_w, h, w):
    w_ratio = orig_w / w
    h_ratio = orig_h / h
    return [(b[0]*w_ratio, b[1]*h_ratio, b[2]*w_ratio, b[3]*h_ratio) for b in boxes]


def test(image_file):
    print(image_file)

    image = cv2.imread(image_file)
    resized = imutils.resize(image, width=512)
    boxes = detect(resized)
    boxes = correct_boxes(boxes, *image.shape[:2], *resized.shape[:2])

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        print('({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(x1, y1, x2, y2))
        cv2.rectangle(rgb, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
    imshow(rgb)


def load_and_detect(image_file):
    print(image_file)
    image = cv2.imread(image_file)
    resized = imutils.resize(image, width=512)
    boxes = detect(resized)
    boxes = correct_boxes(boxes, *image.shape[:2], *resized.shape[:2])
    return boxes


def eval(image_files, iou_thresh=0.5):
    with mp.Pool(processes=4) as p:
        predicted = p.map(load_and_detect, image_files)

    with open('../w5_text_bbox_list.pkl', 'rb') as f:
        actual = pickle.load(f)

    tp = 0
    fp = 0
    npos = 0

    for pred, gt in zip(predicted, actual):
        npos += 1
        for det in pred:
            iou = bbox_iou(det, gt)
            if iou >= iou_thresh:
                tp += 1
            else:
                fp += 1

    prec = tp / (tp + fp)
    rec = tp / npos
    print('prec: {:.4f}, rec: {:.4f}'.format(prec, rec))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['eval', 'test'])
    args = parser.parse_args()

    if args.mode == 'test':
        image = random.choice(glob.glob('../data/w5_BBDD_random/*.jpg'))
        #image = '../data/w5_BBDD_random/ima_000075.jpg'
        #image = '../data/w5_BBDD_random/ima_000124.jpg'
        #image = '../data/w5_BBDD_random/ima_000153.jpg'
        #image = '../data/w5_BBDD_random/ima_000059.jpg'
        test(image)

    elif args.mode == 'eval':
        images = glob.glob('../data/w5_BBDD_random/*.jpg')
        #images = np.random.choice(images, 10)
        eval(images, iou_thresh=0.1)


if __name__ == '__main__':
    main()
