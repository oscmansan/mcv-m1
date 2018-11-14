import glob, os
import argparse
import pickle
import random
import multiprocessing.dummy as mp

import numpy as np
import imutils
import cv2

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def imshow(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


def find_text_ROI():
    with open("../w5_text_bbox_list.pkl", "rb") as fp:
        # tlx,tly, brx,bry
        bbox_gt = pickle.load(fp)

    top_limit_vector = []
    bottom_limit_vector = []
    area_vector = []
    for image in glob.glob('../data/w5_BBDD_random/*.jpg'):
        im = cv2.imread(image)
        index = int(os.path.split(image)[-1].split(".")[0].split("_")[1])
        tlx, tly, brx, bry = bbox_gt[index]
        H, W, _ = np.shape(im)
        h = bry - tly
        w = brx - tlx
        area_vector.append((h * w) / (H * W))
        if bry < H / 2:
            top_limit_vector.append(bry / H)
        else:
            if tly / H > 1:
                print(image)
                print(bbox_gt[index])
            bottom_limit_vector.append(tly / H)
    top_limit = max(top_limit_vector)
    bottom_limit = min(bottom_limit_vector)
    print("Top and bottom limits", top_limit, bottom_limit)
    print("Min and max areas", min(area_vector), max(area_vector))


def fill_holes(mask):
    im_floodfill = mask.astype(np.uint8).copy()
    h, w = im_floodfill.shape[:2]
    filling_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, filling_mask, (0, 0), 1)
    return mask.astype(np.uint8) | cv2.bitwise_not(im_floodfill)


def visualize_boxes(image, window_candidates):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    if len(window_candidates) > 0:
        for candidate in window_candidates:
            # tlx,tly, brx,bry
            minc, minr, maxc, maxr = candidate
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    plt.show()


def detect(img, method="difference"):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "tophat":

        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        blackhat = cv2.cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        tophat = tophat if np.sum(tophat) > np.sum(blackhat) else blackhat

        thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
        expansion = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        final_mask = expansion

    elif method == "difference":
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        blur = cv2.GaussianBlur(closing - opening, (7, 7), 0)

        thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        f1 = fill_holes(thresh2)
        # thresh2 = cv2.threshold(thresh4,250,255,cv2.THRESH_BINARY)[1]
        # imshow(thresh2,"thresh2")

        expansion = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3)))
        # expansion = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        final_mask = expansion

    imshow(final_mask)
    contours = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    boxes = []
    bad_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)
        H, W, _ = np.shape(img)
        rect_area = w * h
        image_area = H * W
        extent = area / rect_area
        tly = y
        bry = tly + h

        cond1 = extent > 0.2
        cond2 = h > 10
        cond3 = rect_area / image_area <= 0.2935
        cond4 = 1.75 < w / h
        cond5 = tly / H >= 0.5719 or bry / H <= 0.2974

        if cond1 and cond2 and cond3 and cond4 and cond5:
            boxes.append((x, y, x + w, y + h))
        else:
            bad_boxes.append((x, y, x + w, y + h))
            # print(cond1, cond2, cond3, cond4, cond5)
    visualize_boxes(img, bad_boxes)

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
    return [(b[0] * w_ratio, b[1] * h_ratio, b[2] * w_ratio, b[3] * h_ratio) for b in boxes]


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
        image = '../data/w5_BBDD_random/ima_000075.jpg'
        # image = '../data/w5_BBDD_random/ima_000124.jpg'
        # image = '../data/w5_BBDD_random/ima_000153.jpg'
        # image = '../data/w5_BBDD_random/ima_000059.jpg'
        test(image)

    elif args.mode == 'eval':
        images = glob.glob('../data/w5_BBDD_random/*.jpg')
        # images = np.random.choice(images, 10)
        eval(images, iou_thresh=0.1)


if __name__ == '__main__':
    main()
