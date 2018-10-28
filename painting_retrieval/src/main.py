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

    methods = ['rgb_histogram', 'hsv_histogram', 'lab_histogram', 'ycrcb_histogram', 'cld', 'rgb_histogram_pyramid',
               'hsv_histogram_pyramid', 'lab_histogram_pyramid', 'ycrcb_histogram_pyramid']
    texture_methods = ['gabor', 'glcm', 'None']
    dist_metrics = ['euclidean_distance', 'l1_distance', 'cosine_distance', 'chi_square', 'hellinguer_distance',
                    'bhattacharya_distance']
    hist_metrics = ['intersection', 'correlation']

    for method in methods:
        for texture_method in texture_methods:
            for metric in dist_metrics:

                with Timer('query_batch'):
                    results = query_batch(query_files, image_files, method, texture_method, metric)

                actual = []
                predicted = []
                for query_file, result in zip(query_files, results):
                    query_retrieval = [query_gt[_filename_to_id(query_file)]]
                    predicted_ids = []
                    for image_file, dist in result:
                        predicted_ids.append(_filename_to_id(image_file))
                    actual.append(query_retrieval)
                    predicted.append(predicted_ids)

                if texture_method != 'None':
                    print('Result for ', method, ', ', texture_method, ' and ', metric, ':', mapk(actual, predicted))
                else:
                    print('Result for ', method, ' and ', metric, ':', mapk(actual, predicted))

        for metric in hist_metrics:

            with Timer('query_batch'):
                results = query_batch(query_files, image_files, method, 'None', metric)

            actual = []
            predicted = []
            for query_file, result in zip(query_files, results):
                query_retrieval = [query_gt[_filename_to_id(query_file)]]
                predicted_ids = []
                for image_file, dist in result:
                    predicted_ids.append(_filename_to_id(image_file))
                actual.append(query_retrieval)
                predicted.append(predicted_ids)

            print('Result for ', method, ' and ', metric, ':', mapk(actual, predicted))



if __name__ == '__main__':
    main()
