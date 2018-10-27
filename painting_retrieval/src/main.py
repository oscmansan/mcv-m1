import os
import glob

from retrieval import query_batch
from evaluation import mapk
from timer import Timer
import pickle


def _filename_to_id(filename):
    head, tail = os.path.split(filename)
    name, ext = os.path.splitext(tail)
    return int(name.split('_')[1])


def main():
    query_files = sorted(glob.glob('../data/query_devel_random/*.jpg'))
    image_files = sorted(glob.glob('../data/museum_set_random/*.jpg'))

    with open('../query_corresp_simple_devel.pkl', 'rb') as f:
        query_gt = pickle.load(f)

    methods = ['rgb_histogram', 'hsv_histogram', 'lab_histogram', 'ycrcb_histogram', 'cld', 'pyramid_rgb_histogram',
               'pyramid_hsv_histogram', 'pyramid_lab_histogram', 'pyramid_ycrcb_histogram']
    metrics = ['euclidean_distance', 'euclidean_distance', 'l1_distance', 'cosine_distance', 'correlation',
               'chi_square', 'intersection', 'hellinguer_distance', 'bhattacharya_distance']

    for method in methods:
        for metric in metrics:

            with Timer('query_batch'):
                results = query_batch(query_files, image_files, method, metric)

            actual = []
            predicted = []
            for query_file, result in zip(query_files, results):
                query_retrieval = [query_gt[_filename_to_id(query_file)]]
                predicted_ids = []
                for image_file, dist in result:
                    predicted_ids.append(_filename_to_id(image_file))
                actual.append(query_retrieval)
                predicted.append(predicted_ids)

            print('Result for ', method, " and ", metric, " :", mapk(actual, predicted))


if __name__ == '__main__':
    main()
