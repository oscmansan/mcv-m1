import multiprocessing.dummy as mp

import numpy as np
import imageio

from descriptors import extract_descriptors
from distances import compute_distance


def _read_and_extract(image_file, method):
    image = imageio.imread(image_file)
    return extract_descriptors(image, method)


def query(query_file, image_files, method, metric, k=10):
    query_embd = _read_and_extract(query_file, method)

    with mp.Pool(processes=20) as p:
        image_descriptors = p.starmap(_read_and_extract, [(image_file, method) for image_file in image_files])
        distances = p.starmap(compute_distance, [(query_embd, image_embd, metric) for image_embd in image_descriptors])

    inds = np.argsort(distances)[:k]
    return [(image_files[i], distances[i]) for i in inds]


def query_batch(query_files, image_files, method, metric, k=10):
    results = []
    with mp.Pool(processes=20) as p:
        query_descriptors = p.starmap(_read_and_extract, [(query_file, method) for query_file in query_files])
        image_descriptors = p.starmap(_read_and_extract, [(image_file, method) for image_file in image_files])

        for query_embd in query_descriptors:
            distances = p.starmap(compute_distance, [(query_embd, image_embd, metric) for image_embd in image_descriptors])
            inds = np.argsort(distances)[:k]
            result = [(image_files[i], distances[i]) for i in inds]
            results.append(result)
    return results
