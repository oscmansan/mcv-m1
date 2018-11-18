import os
import glob
import argparse
import pickle
import random
from collections import defaultdict

import numpy as np
import imutils
import cv2
from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt


def imshow(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


def draw_boxes(image, boxes, color=(0, 255, 0)):
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        #print('({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(x1, y1, x2, y2))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
    return image


def find_text_region():
    with open("../w5_text_bbox_list.pkl", "rb") as fp:
        # tlx, tly, brx, bry
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


def merge_boxes(boxes):
    y = np.array([[(b[1] + b[3]) / 2, b[3] - b[1]] for b in boxes])
    clt = DBSCAN(eps=22, min_samples=1, metric='l1').fit(y)
    labels = clt.labels_

    clusters = defaultdict(list)
    for box, label in zip(boxes, labels):
        if label != -1:
            clusters[label].append(box)
    clusters = clusters.values()

    merged_boxes = []
    areas = []
    for clt in clusters:
        x, y, w, h = cv2.boundingRect(points=np.concatenate(clt).reshape(-1, 2))
        merged_boxes.append((x, y, x + w, y + h))

        area = np.sum([(b[2]-b[0])*(b[3]-b[1]) for b in clt])
        areas.append(area)

    return merged_boxes, areas


def detect(img, method='difference', show=False):
    im_h, im_w = img.shape[:2]

    def tophat(img):
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        blackhat = cv2.cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        tophat = tophat if np.sum(tophat) > np.sum(blackhat) else blackhat
        if show:
            imshow(tophat)

        thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if show:
            imshow(thresh)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        return thresh

    def difference(img):
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3)))
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        tophat = closing - opening
        #blur = cv2.GaussianBlur(tophat, (7, 7), 0)

        #thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.threshold(tophat, 0.5*tophat.max(), 255, cv2.THRESH_BINARY)[1]

        #filled = fill_holes(thresh)
        #thresh = cv2.threshold(thresh4,250,255,cv2.THRESH_BINARY)[1]
        #imshow(thresh)

        dilation = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
        #expansion = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

        if show:
            imshow(closing)
            imshow(opening)
            imshow(tophat)
            imshow(thresh)

        return dilation

    func = {
        'tophat': tophat,
        'difference': difference
    }

    # find contours
    mask = func[method](img)
    if show:
        imshow(mask)

    # detect boxes from contours
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    boxes, bad_boxes = [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        image_area = im_h * im_w
        area = cv2.contourArea(cnt)
        rect_area = w * h
        extent = area / rect_area

        # filter boxes
        cond1 = extent > 0.2
        cond2 = h > 10
        cond3 = (rect_area / image_area) <= 0.2935
        cond4 = (w / h) > 1
        cond5 = (y / im_h) >= 0.5719 or ((y + h) / im_h) <= 0.2974
        #print(cond1, cond2, cond3, cond4, cond5)

        if all([cond1, cond2, cond3, cond4, cond5]):
            boxes.append((x, y, x + w, y + h))
        else:
            bad_boxes.append((x, y, x + w, y + h))
    if show:
        tmp = draw_boxes(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), bad_boxes, color=(255, 0, 0))
        tmp = draw_boxes(tmp, boxes, color=(0, 255, 0))
        imshow(tmp)

    # merge boxes
    if boxes:
        merged_boxes, areas = merge_boxes(boxes)
        if show:
            imshow(draw_boxes(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), merged_boxes))

        filtered_boxes = []
        filtered_areas = []
        for box, area in zip(merged_boxes, areas):
            tlx, tly, brx, bry = box
            h = bry-tly
            w = brx-tlx
            if 0.05 < h/w < 0.25:
                filtered_boxes.append(box)
                filtered_areas.append(area)
        if filtered_boxes:
            idx = np.argmax(filtered_areas)
            boxes = [filtered_boxes[idx]]
        else:
            boxes = []

    return boxes


def correct_boxes(boxes, orig_h, orig_w, h, w):
    w_ratio = orig_w / w
    h_ratio = orig_h / h

    corrected = []
    for b in boxes:
        tlx = int(np.floor(b[0] * w_ratio))
        tly = int(np.floor(b[1] * h_ratio))
        brx = int(np.ceil(b[2] * w_ratio))
        bry = int(np.ceil(b[3] * h_ratio))
        corrected.append((tlx, tly, brx, bry))

    return corrected


def filter_text_keypoints(img, keypoints):
    resized = imutils.resize(img, width=512)
    boxes = detect(resized)
    boxes = correct_boxes(boxes, *img.shape[:2], *resized.shape[:2])

    def inside(pt, box):
        # point = (x, y)
        # box = (tlx, tly, brx, bry)
        return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]

    filtered = []
    for kp in keypoints:
        for box in boxes:
            if inside(kp.pt, box):
                break
        else:
            filtered.append(kp)

    return filtered


def compute_text_mask(img, method='difference'):
    resized = imutils.resize(img, width=512)
    boxes = detect(resized, method)
    boxes = correct_boxes(boxes, *img.shape[:2], *resized.shape[:2])

    mask = np.full(img.shape[:2], 255, dtype=np.uint8)
    for box in boxes:
        tlx, tly, brx, bry = box
        mask[tly:bry, tlx:brx] = 0

    return mask


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


def save_results(predicted):
    results = [list(pred[0]) if pred else [0, 0, 0, 0] for pred in predicted]
    path = '../results'
    if not os.path.exists(path):
        os.mkdir(path)
    file_name = os.path.join(path, 'text.pkl')
    print('Saving text detection results to {}'.format(file_name))
    with open(file_name, "wb") as fp:
        pickle.dump(results, fp)


def eval(image_files, iou_thresh=0.5, save=False):
    predicted = []
    for image_file in image_files:
        print(image_file)
        gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        resized = imutils.resize(gray, width=512)
        boxes = detect(resized)
        boxes = correct_boxes(boxes, *gray.shape[:2], *resized.shape[:2])
        predicted.append(boxes)

    if save:
        save_results(predicted)

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

    overlap = [bbox_iou(det, gt) for pred, gt in zip(predicted, actual) for det in pred]
    mean_overlap = np.mean(overlap)
    print('mean IoU: {:.4f}'.format(mean_overlap))


def test(image_file):
    print(image_file)

    img = cv2.imread(image_file)
    resized = imutils.resize(img, width=512)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    boxes = detect(gray, method='difference', show=True)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    imshow(draw_boxes(rgb, boxes))


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
        #image = '../data/w5_BBDD_random/ima_000115.jpg'
        #image = '../data/w5_BBDD_random/ima_000108.jpg'
        #image = '../data/w5_BBDD_random/ima_000155.jpg'
        #image = '../data/w5_BBDD_random/ima_000052.jpg'
        #image = '../data/w5_BBDD_random/ima_000204.jpg'
        #image = '../data/w5_BBDD_random/ima_000055.jpg'
        #image = '../data/w5_BBDD_random/ima_000146.jpg'
        #image = '../data/w5_BBDD_random/ima_000061.jpg'
        #image = '../data/w5_BBDD_random/ima_000037.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000111.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000002.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000006.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000007.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000016.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000020.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000027.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000038.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000047.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000063.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000066.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000082.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000085.jpg'  # FIXME
        #image = '../data/w5_BBDD_random/ima_000101.jpg'  # FIXME
        test(image)

    elif args.mode == 'eval':
        images = glob.glob('../data/w5_BBDD_random/*.jpg')
        #images = np.random.choice(images, 10)
        eval(images, iou_thresh=0.3, save=True)


if __name__ == '__main__':
    main()
    '''
    from keypoints import detect_keypoints, Mode
    image_file = random.choice(glob.glob('../data/w5_BBDD_random/*.jpg'))
    gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    gray = imutils.resize(gray, width=512)
    keypoints = detect_keypoints(gray, 'orb', Mode.IMAGE, None)
    imshow(cv2.drawKeypoints(gray, keypoints, None))
    mask = compute_text_mask(gray)
    keypoints = detect_keypoints(gray, 'orb', Mode.IMAGE, mask)
    imshow(cv2.drawKeypoints(gray, keypoints, None))
    '''
