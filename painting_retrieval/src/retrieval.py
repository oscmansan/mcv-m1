import multiprocessing.dummy as mp

import numpy as np
import imageio

from keypoints import detect_keypoints
from descriptors import extract_local_descriptors
from distances import match_descriptors


def _read_and_extract(image_file, keypoint_method, descriptor_method):
    image = imageio.imread(image_file)
    keypoints = detect_keypoints(image, keypoint_method)
    descriptors = extract_local_descriptors(image, keypoints, descriptor_method)
    return descriptors


def query(query_file, image_files, keypoint_method, descriptor_method, match_method, distance_metric, k=10):
    query_embd = _read_and_extract(query_file, keypoint_method, descriptor_method)

    with mp.Pool(processes=20) as p:
        image_descriptors = p.starmap(_read_and_extract, [(image_file, keypoint_method, descriptor_method) for image_file in image_files])
        distances = p.starmap(match_descriptors, [(query_embd, image_embd, match_method, distance_metric) for image_embd in image_descriptors])

    inds = np.argsort(distances)[:k]
    result = [(image_files[i], distances[i]) for i in inds]
    return result


def query_batch(query_files, image_files, keypoint_method, descriptor_method, match_method, distance_metric, k=10):
    results = []
    with mp.Pool(processes=20) as p:
        query_descriptors = p.starmap(_read_and_extract, [(query_file, keypoint_method, descriptor_method) for query_file in query_files])
        image_descriptors = p.starmap(_read_and_extract, [(image_file, keypoint_method, descriptor_method) for image_file in image_files])

        for query_embd in query_descriptors:
            distances = p.starmap(match_descriptors, [(query_embd, image_embd, match_method, distance_metric) for image_embd in image_descriptors])
            inds = np.argsort(distances)[:k]
            result = [(image_files[i], distances[i]) for i in inds]
            results.append(result)
    return results
