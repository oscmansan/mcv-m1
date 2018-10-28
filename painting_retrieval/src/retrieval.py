import multiprocessing.dummy as mp

import imageio
import numpy as np

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


def query_batch(query_files, image_files, color_method, metric, texture_method=None, k=10):
    results = []
    with mp.Pool(processes=20) as p:
        query_color_descriptors = p.starmap(_read_and_extract, [(query_file, color_method) for query_file in query_files])
        image_color_descriptors = p.starmap(_read_and_extract, [(image_file, color_method) for image_file in image_files])

        if texture_method:
            query_texture_descriptors = p.starmap(_read_and_extract, [(query_file, texture_method) for query_file in query_files])
            image_texture_descriptors = p.starmap(_read_and_extract, [(image_file, texture_method) for image_file in image_files])

        for q in range(len(query_files)):
            query_embd = query_color_descriptors[q]
            distances = p.starmap(compute_distance, [(query_embd, image_embd, metric) for image_embd in image_color_descriptors])

            if texture_method:
                query_embd = query_texture_descriptors[q]
                texture_distances = p.starmap(compute_distance, [(query_embd, image_embd, metric) for image_embd in image_texture_descriptors])
                distances = [0.6 * d + 0.4 * td for d, td in zip(distances, texture_distances)]

            inds = np.argsort(distances)[:k]
            result = [(image_files[i], distances[i]) for i in inds]
            results.append(result)
    return results
