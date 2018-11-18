import pickle

import numpy as np
from imutils import resize
import cv2

from matplotlib import pyplot as plt

from keypoints import detect_keypoints, Mode
from descriptors import extract_local_descriptors
from picture_detection import crop_picture
from text_detection import compute_text_mask
from distances import _filter_matches


KEYPOINT_METHOD = 'orb'
DESCRIPTOR_METHOD = 'orb'
DISTANCE_METRIC = cv2.NORM_HAMMING
DISTANCE_RATIO = 0.5


def imshow(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


def _read_and_extract(image_file, keypoint_method, descriptor_method, mode):
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    if mode == Mode.QUERY:
        image = crop_picture(image)
    image = resize(image, width=512)

    if mode == Mode.IMAGE:
        mask = compute_text_mask(image, method='difference')
        keypoints = detect_keypoints(image, keypoint_method, mode, mask)
    else:
        keypoints = detect_keypoints(image, keypoint_method, mode)

    descriptors = extract_local_descriptors(image, keypoints, descriptor_method)

    return image, keypoints, descriptors


with open('../w5_query_devel.pkl', 'rb') as f:
    corresp = pickle.load(f)

num_matched_keypoints = []
distances_avg = []
for query_id, image_ids in corresp:
    if len(image_ids) == 1 and image_ids[0] == -1:
        continue

    query_file = '../data/w5_devel_random/ima_{:06d}.jpg'.format(query_id)
    query, query_kps, query_descs = _read_and_extract(query_file, KEYPOINT_METHOD, DESCRIPTOR_METHOD, Mode.QUERY)
    print('query: {} ({})'.format(query_file, len(query_descs)))

    for image_id in image_ids:
        image_file = '../data/w5_BBDD_random/ima_{:06d}.jpg'.format(image_id)
        image, image_kps, image_descs = _read_and_extract(image_file, KEYPOINT_METHOD, DESCRIPTOR_METHOD, Mode.IMAGE)
        print('image: {} ({})'.format(image_file, len(image_descs)))

        bf = cv2.BFMatcher(normType=DISTANCE_METRIC)
        matches = bf.knnMatch(query_descs, image_descs, k=2)
        good = _filter_matches(matches, ratio=DISTANCE_RATIO)

        num_matched_keypoints.append(len(good))
        distances_avg.append(np.mean([m.distance for m in good]))

        imshow(cv2.drawMatches(query, query_kps, image, image_kps, good, None))

print(sorted(num_matched_keypoints))
print(sorted(distances_avg, reverse=True))
