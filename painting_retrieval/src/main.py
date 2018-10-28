import os
import glob
import pickle
from itertools import product

from retrieval import query_batch
from evaluation import mapk
from timer import Timer


def _filename_to_id(filename):
    head, tail = os.path.split(filename)
    name, ext = os.path.splitext(tail)
    return int(name.split('_')[1])


def _save_results_in_pkl(method_number, results):
    try:
        os.mkdir('../results/method' + str(method_number))
    except:
        pass
    with open('../results/method' + str(method_number) + "result.pkl", "wb") as fp:  # Pickling
        pickle.dump(results, fp)


def main():
    query_files = sorted(glob.glob('../data/query_devel_random/*.jpg'))
    image_files = sorted(glob.glob('../data/museum_set_random/*.jpg'))

    with open('../query_corresp_simple_devel.pkl', 'rb') as f:
        query_gt = pickle.load(f)

    color_methods = ['rgb_histogram', 'hsv_histogram', 'lab_histogram', 'ycrcb_histogram', 'cld',
                     'rgb_histogram_pyramid', 'hsv_histogram_pyramid', 'lab_histogram_pyramid',
                     'ycrcb_histogram_pyramid']
    texture_methods = [None, 'gabor', 'glcm']
    dist_metrics = ['euclidean_distance', 'l1_distance', 'cosine_distance']
    hist_metrics = ['intersection', 'correlation', 'chi_square', 'hellinguer_distance', 'bhattacharya_distance']

    for texture_method, color_method, metric in product(texture_methods, color_methods, dist_metrics + hist_metrics):
        print('({}, {}, {})'.format(color_method, texture_method, metric))

        with Timer('query_batch'):
            results = query_batch(query_files, image_files, color_method, metric, texture_method)

        actual = []
        predicted = []
        for query_file, result in zip(query_files, results):
            query_retrieval = [query_gt[_filename_to_id(query_file)]]
            predicted_ids = [_filename_to_id(image_file) for image_file, dist in result]
            actual.append(query_retrieval)
            predicted.append(predicted_ids)
        print('MAP@K: {}'.format(mapk(actual, predicted)))


if __name__ == '__main__':
    main()
