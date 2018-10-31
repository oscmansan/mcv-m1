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

    keypoint_methods = ['sift']
    descriptor_methods = ['sift']
    match_methods = ['brute_force']
    distance_metrics = ['l2']

    for keypoint_method, descriptor_method, match_method,  distance_metric in product(keypoint_methods, descriptor_methods, match_methods, distance_metrics):
        print('({}, {}, {}, {})'.format(keypoint_method, descriptor_method, match_method, distance_metric))

        with Timer('query_batch'):
            results = query_batch(query_files, image_files, keypoint_method, descriptor_method, match_method, distance_metric)

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
            #save_results(predicted, args.results_path, method='{}_{}_{}_{}'.format(keypoint_method, descriptor_method, match_method, distance_metric))
            print('queries: {}'.format([_filename_to_id(q) for q in query_files]))
            print('predicted: {}'.format(predicted))

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
