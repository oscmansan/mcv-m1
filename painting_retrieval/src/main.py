import glob

import imageio
from matplotlib import pyplot as plt

from retrieval import query_batch
from timer import Timer


def main():
    query_files = sorted(glob.glob('../data/query_devel_random/*.jpg'))
    image_files = sorted(glob.glob('../data/museum_set_random/*.jpg'))

    method = 'pyramid_hsv_histogram'
    metric = 'euclidean_distance'
    with Timer('query_batch'):
        results = query_batch(query_files, image_files, method, metric)

    for query_file, result in zip(query_files, results):
        print('(query) {}'.format(query_file))
        for image_file, dist in result:
            print('({:.6f}) {}'.format(dist, image_file))

        #plt.figure()
        #plt.imshow(imageio.imread(query_file))
        #plt.figure()
        #plt.imshow(imageio.imread(result[0][0]))
    #plt.show()


if __name__ == '__main__':
    main()
