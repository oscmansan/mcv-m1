import os
import multiprocessing.dummy as mp

import numpy as np
import imageio
import cv2

from keypoints import detect_keypoints
from descriptors import extract_local_descriptors
from distances import match_descriptors
from timer import Timer


def _read_and_extract(image_file, keypoint_method, descriptor_method):
    image = imageio.imread(image_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints = detect_keypoints(image_gray, keypoint_method)
    descriptors = extract_local_descriptors(image_gray, keypoints, descriptor_method)
    return descriptors


def _load_or_compute(image_file, keypoint_method, descriptor_method):
    print(image_file, keypoint_method, descriptor_method)

    directory, filename = image_file.rsplit(os.sep, 2)[-2:]
    name = os.path.splitext(filename)[0]
    path = '../descriptors/{}'.format(directory)
    file_path = '{}/{}_{}_{}.npy'.format(path, keypoint_method, descriptor_method, name)

    if os.path.exists(file_path):
        descriptors = np.load(file_path)
        return descriptors
    else:
        descriptors = _read_and_extract(image_file, keypoint_method, descriptor_method)
        if not os.path.exists(path): os.makedirs(path)
        np.save(file_path, descriptors)
        return descriptors


def query(query_file, image_files, keypoint_method, descriptor_method, match_method, distance_metric, k=10):
    query_embd = _read_and_extract(query_file, keypoint_method, descriptor_method)

    with mp.Pool(processes=20) as p:
        image_descriptors = p.starmap(_load_or_compute, [(image_file, keypoint_method, descriptor_method) for image_file in image_files])
        distances = p.starmap(match_descriptors, [(query_embd, image_embd, match_method, distance_metric) for image_embd in image_descriptors])

    inds = np.argsort(distances)[:k]
    result = [(image_files[i], distances[i]) for i in inds]
    return result


def query_batch(query_files, image_files, keypoint_method, descriptor_method, match_method, distance_metric, k=10):
    results = []
    with mp.Pool(processes=20) as p:
        with Timer('extract descriptors'):
            query_descriptors = p.starmap(_load_or_compute, [(query_file, keypoint_method, descriptor_method) for query_file in query_files])
            image_descriptors = p.starmap(_load_or_compute, [(image_file, keypoint_method, descriptor_method) for image_file in image_files])

        with Timer('match descriptors'):
            for query_embd in query_descriptors:
                distances = p.starmap(match_descriptors, [(query_embd, image_embd, match_method, distance_metric) for image_embd in image_descriptors])
                inds = np.argsort(distances)[:k]
                result = [(image_files[i], distances[i]) for i in inds]
                results.append(result)
    return results
