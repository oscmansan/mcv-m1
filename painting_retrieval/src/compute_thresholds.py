import numpy as np
import imageio
import cv2

from keypoints import detect_keypoints, Mode
from descriptors import extract_local_descriptors
from distances import _filter_matches


keypoint_method = 'sift'
descriptor_method = 'sift'
distance_metric = cv2.NORM_L1


def _read_and_extract(image_file, mode):
    image = imageio.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints = detect_keypoints(gray, keypoint_method, mode)
    descriptors = extract_local_descriptors(gray, keypoints, descriptor_method)
    return descriptors


corresp = [[0, [-1]], [1, [-1]], [2, [115]], [3, [-1]], [4, [-1]], [5, [99]], [6, [-1]], [7, [89]], [8, [19]], [9, [85]], [10, [90]], [11, [121, 117]], [12, [-1]], [13, [-1]], [14, [130]], [15, [6, 84]], [16, [35, 48, 52]], [17, [118]], [18, [-1]], [19, [-1]], [20, [-1]], [21, [-1]], [22, [60]], [23, [119, 128]], [24, [-1]], [25, [47]], [26, [-1]], [27, [41]], [28, [-1]], [29, [126, 123]]]

num_matched_keypoints = []
distances_avg = []
for query_id, image_ids in corresp:
    if len(image_ids) == 1 and image_ids[0] == -1:
        continue

    query_file = '../data/query_devel_W4/ima_{:06d}.jpg'.format(query_id)
    query_embd = _read_and_extract(query_file, Mode.QUERY)
    print('query: {} ({})'.format(query_file, len(query_embd)))

    for image_id in image_ids:
        image_file = '../data/BBDD_W4/ima_{:06d}.jpg'.format(image_id)
        image_embd = _read_and_extract(image_file, Mode.IMAGE)
        print('image: {} ({})'.format(image_file, len(image_embd)))

        bf = cv2.BFMatcher(normType=distance_metric)
        matches = bf.knnMatch(image_embd, query_embd, k=2)
        good = _filter_matches(matches, ratio=0.5)

        num_matched_keypoints.append(len(good))
        distances_avg.append(np.mean([m.distance for m in good]))

print(sorted(num_matched_keypoints))
print(sorted(distances_avg, reverse=True))
