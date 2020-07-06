#TODO: straightforward

def sf_single(arr1, arr2, block_shape):
    '''
    Straightforward approach, single process.
    '''
    out_shape = [arr1.shape[ax] // block_shape[ax] for ax in
            range(arr1.ndim)]
    out = np.empty(out_shape)
    for coord in itertools.product(*[range(d) for d in out_shape]):
        s = [slice(coord[ax]*block_shape[ax],
            (coord[ax]+1)*block_shape[ax]) for ax in range(arr1.ndim)]
        block1 = arr1[tuple(s)]
        block2 = arr2[tuple(s)]
        out[coord] = np.corrcoef(block1.flatten(), block2.flatten())[0,1]
    return out


def sf_parallel(arr1, arr2, block_shape):
    '''
    Straightforward approach, parallelized.
    '''


def bw_single(arr1, arr2, block_shape):
    '''
    Blockwise approach, single process.
    '''


def bw_parallel(arr1, arr2, block_shape):
    '''
    Blockwise approach, parallelized through dask.array.
    '''


def run(image1_filepath, image2_filepath):


if __name__ == '__main__':
    fire.Fire(run)
