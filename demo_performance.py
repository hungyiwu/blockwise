import itertools
import multiprocessing as mp
import time

import numpy as np
import tifffile
import fire

import blockwise


class Timer(object):
    '''
    Modified from
    https://blog.usejournal.com/
    how-to-create-your-own-timing-context-manager-in-python-a0e944b48cf8
    '''
    def __init__(self, description):
        self.description = description
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, type, value, traceback):
        self.end = time.time()
        print('{}: {:.3f} sec'.format(
            self.description, self.end-self.start))


def sf_single(arr1, arr2, block_shape):
    """
    Straightforward approach, single process.
    """
    out_shape = [arr1.shape[ax] // block_shape[ax] for ax in range(arr1.ndim)]
    out = np.empty(out_shape)
    for coord in itertools.product(*[range(d) for d in out_shape]):
        s = [
            slice(coord[ax] * block_shape[ax],
                  (coord[ax] + 1) * block_shape[ax]) for ax in range(arr1.ndim)
        ]
        block1 = arr1[tuple(s)]
        block2 = arr2[tuple(s)]
        out[coord] = np.corrcoef(block1.flatten(), block2.flatten())[0, 1]
    return out


def sf_parallel_feed(arr1, arr2, block_shape):
    """
    Straightforward approach, parallelized.
    """
    out_shape = [arr1.shape[ax] // block_shape[ax] for ax in range(arr1.ndim)]
    for coord in itertools.product(*[range(d) for d in out_shape]):
        s = [
            slice(coord[ax] * block_shape[ax],
                  (coord[ax] + 1) * block_shape[ax]) for ax in range(arr1.ndim)
        ]
        block1 = arr1[tuple(s)]
        block2 = arr2[tuple(s)]
        yield coord, block1, block2


def sf_parallel_run(job):
    """
    Straightforward approach, parallelized.
    """
    coord, block1, block2 = job
    return coord, np.corrcoef(block1.flatten(), block2.flatten())[0, 1]


def sf_parallel(arr1, arr2, block_shape, nproc=4):
    """
    Straightforward approach, parallelized.
    """
    out_shape = [arr1.shape[ax] // block_shape[ax] for ax in range(arr1.ndim)]
    out = np.empty(out_shape)
    with mp.Pool(nproc) as wp:
        for coord, val in wp.imap_unordered(
            sf_parallel_run, sf_parallel_feed(arr1, arr2, block_shape)
        ):
            out[coord] = val
    return out


def bw_single(arr1, arr2, block_shape):
    """
    Blockwise approach, single process.
    """
    return blockwise.bw_corrcoef(arr1, arr2, block_shape)


def run(image1_filepath, image2_filepath):
    # load data
    arr1 = tifffile.imread(image1_filepath)
    arr2 = tifffile.imread(image2_filepath)
    block_shape = (10, 10)
    # preprocess data
    arr1 = blockwise.trim(arr1, block_shape)
    arr2 = blockwise.trim(arr2, block_shape)
    # describe data
    print('arr1: {}, arr2: {}, block shape: {}'.format(arr1.shape, arr2.shape,
        block_shape))
    # run
    with Timer('straightforward, 1 process'):
        sf_single(arr1, arr2, block_shape)
    with Timer('straightforward, 4 processes'):
        sf_parallel(arr1, arr2, block_shape, nproc=4)
    with Timer('blockwise, 1 process'):
        bw_single(arr1, arr2, block_shape)


if __name__ == "__main__":
    fire.Fire(run)
