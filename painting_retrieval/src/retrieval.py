import os
import multiprocessing.dummy as mp

import numpy as np
import imageio
import cv2

from keypoints import detect_keypoints, Mode
from descriptors import extract_local_descriptors
from distances import match_descriptors
from timer import Timer
from detection_picture import crop_picture


def _read_and_extract(image_file, keypoint_method, descriptor_method, mode):
    image = imageio.imread(image_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mode == Mode.QUERY:
        cropped_image = crop_picture(image, image_gray)
    else:
        cropped_image = image_gray
    keypoints = detect_keypoints(cropped_image, keypoint_method, mode)
    descriptors = extract_local_descriptors(cropped_image, keypoints, descriptor_method)
    return descriptors


def _load_or_compute(image_file, keypoint_method, descriptor_method, mode):
    print(image_file, keypoint_method, descriptor_method)

    directory, filename = image_file.rsplit(os.sep, 2)[-2:]
    name = os.path.splitext(filename)[0]
    path = '../descriptors/{}'.format(directory)
    file_path = '{}/{}_{}_{}.npy'.format(path, keypoint_method, descriptor_method, name)

    if os.path.exists(file_path):
        descriptors = np.load(file_path)
        return descriptors
    else:
        descriptors = _read_and_extract(image_file, keypoint_method, descriptor_method, mode)
        if not os.path.exists(path): os.makedirs(path)
        np.save(file_path, descriptors)
        return descriptors


def query(query_file, image_files, keypoint_method, descriptor_method, match_method, distance_metric, k=10):
    query_embd = _load_or_compute(query_file, keypoint_method, descriptor_method, Mode.QUERY)
    with mp.Pool(processes=8) as p:
        image_descriptors = p.starmap(_load_or_compute, [(image_file, keypoint_method, descriptor_method, Mode.IMAGE) for image_file in image_files])

    scores = []
    for image_embd in image_descriptors:
        score = match_descriptors(query_embd, image_embd, match_method, distance_metric)
        scores.append(score)

    inds = np.argsort(scores)[::-1][:k]
    result = [(image_files[i], scores[i]) for i in inds]
    return result


def query_batch(query_files, image_files, keypoint_method, descriptor_method, match_method, distance_metric, k=10):
    results = []
    with Timer('extract descriptors'):
        with mp.Pool(processes=8) as p:
            query_descriptors = p.starmap(_load_or_compute, [(query_file, keypoint_method, descriptor_method, Mode.QUERY) for query_file in query_files])
            image_descriptors = p.starmap(_load_or_compute, [(image_file, keypoint_method, descriptor_method, Mode.IMAGE) for image_file in image_files])

    with Timer('match descriptors'):
        for i, (query_file, query_embd) in enumerate(zip(query_files, query_descriptors), 1):
            print('({}/{}) {}: {} descriptors'.format(i, len(query_files), query_file, len(query_embd)))

            scores = []
            for j, (image_file, image_embd) in enumerate(zip(image_files, image_descriptors), 1):
                print('({}/{}) {}: {} descriptors'.format(j, len(image_files), image_file, len(image_embd)))
                score = match_descriptors(query_embd, image_embd, match_method, distance_metric)
                scores.append(score)

            inds = np.argsort(scores)[::-1][:k]
            result = [(image_files[i], scores[i]) for i in inds]
            results.append(result)
    return results
