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


def query_batch(query_files, image_files, method, texture_method, metric, k=10):
    results = []
    with mp.Pool(processes=20) as p:
        query_descriptors = p.starmap(_read_and_extract, [(query_file, method) for query_file in query_files])
        image_descriptors = p.starmap(_read_and_extract, [(image_file, method) for image_file in image_files])

        if texture_method != 'None':
            query_descriptors_tex = p.starmap(_read_and_extract, [(query_file, texture_method) for query_file in query_files])
            image_descriptors_tex = p.starmap(_read_and_extract, [(image_file, texture_method) for image_file in image_files])

        for q in range(len(query_descriptors)):
            query_embd = query_descriptors[q]
            distances = p.starmap(compute_distance, [(query_embd, image_embd, metric) for image_embd in image_descriptors])
            if texture_method != 'None':
                query_embd_tex = query_descriptors_tex[q]
                dist_tex = p.starmap(compute_distance, [(query_embd_tex, image_embd, metric) for image_embd in image_descriptors_tex])
                distances = [0.6*d + 0.4*dt for (d, dt) in zip(distances, dist_tex)]
            inds = np.argsort(distances)[:k]
            result = [(image_files[i], distances[i]) for i in inds]
            results.append(result)
    return results
