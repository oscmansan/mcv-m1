import os
import glob
import pickle
import argparse
from itertools import product

from retrieval import query_batch
from evaluation import mapk
from timer import Timer


def _filename_to_id(filename):
    head, tail = os.path.split(filename)
    name, ext = os.path.splitext(tail)
    return int(name.split('_')[1])


def _save_results(results, dst_path, method):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    results_fn = os.path.join(dst_path, method + '.pkl')
    print('Saving results to {}'.format(results_fn))
    with open(results_fn, "wb") as fp:  # Pickling
        pickle.dump(results, fp)


def main(args):
    query_files = sorted(glob.glob(args.queries_path))
    image_files = sorted(glob.glob(args.images_path))

    if args.mode == 'eval':
        with open(args.corresp_file, 'rb') as f:
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

        if args.mode == 'eval':
            actual = []
            predicted = []
            for query_file, result in zip(query_files, results):
                actual.append([query_gt[_filename_to_id(query_file)]])
                predicted.append([_filename_to_id(image_file) for image_file, dist in result])
            print('MAP@K: {}'.format(mapk(actual, predicted)))

        elif args.mode == 'test':
            predicted = []
            for query_file, result in zip(query_files, results):
                predicted.append([_filename_to_id(image_file) for image_file, dist in result])
            _save_results(predicted, args.results_path, method='{}_{}_{}'.format(color_method, texture_method, metric))

        else:
            raise ValueError('Invalid mode.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['eval', 'test'])
    parser.add_argument('--queries_path', type=str, default='../data/query_devel_random/*.jpg')
    parser.add_argument('--images_path', type=str, default='../data/museum_set_random/*.jpg')
    parser.add_argument('--corresp_file', type=str, default='../query_corresp_simple_devel.pkl')
    parser.add_argument('--results_path', type=str, default='../results')
    args = parser.parse_args()
    main(args)
