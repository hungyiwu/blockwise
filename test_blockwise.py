import itertools

import numpy as np
import pytest

import blockwise

def test_shift():
    def shift_slow(arr, num, axis, fill_value=np.nan):
        out = np.empty_like(arr)
        if num == 0:
            out[:] = arr
            return out
        out = np.roll(arr, num, axis)
        s = [slice(None), ]*out.ndim
        if num > 0:
            s[ax] = slice(0, num)
        elif num < 0:
            s[ax] = slice(num, None)
        s = tuple(s)
        out[s] = fill_value
        return out

    arr_shape = (3, 3)
    arr = np.arange(np.prod(arr_shape)).reshape(arr_shape)
    for ax in range(arr.ndim):
        d = arr_shape[ax]
        for num in range(-(d-1), d):
            ref = shift_slow(arr, num, ax, fill_value=-1)
            test = blockwise.shift(arr, num, ax, fill_value=-1)
            assert np.array_equal(ref, test)


def test_repeat_block():
    arr = np.arange(4).reshape((2,2))
    ref = np.array([[0,0,1,1], ]*3 + [[2,2,3,3], ]*3)
    test = blockwise.repeat_block(arr, (3,2))
    assert np.array_equal(ref, test)


def test_trim():
    arr_shape, block_shape = (6, 5), (3, 2)
    arr = np.arange(np.prod(arr_shape)).reshape(arr_shape)
    test = blockwise.trim(arr, block_shape)
    ref = arr[:, :4]
    assert np.array_equal(ref, test)


@pytest.mark.parametrize(
    'funcname', [
        'sum',
        'mean',
        'std',
        ]
)
def test_bw_func(funcname):

    def bruteforce_bw(arr, block_shape):
        out_shape = [arr.shape[ax] // block_shape[ax] for ax in
                range(arr.ndim)]
        out = np.empty(out_shape)
        for coord in itertools.product(*[range(d) for d in out_shape]):
            s = [slice(coord[ax]*block_shape[ax],
                (coord[ax]+1)*block_shape[ax]) for ax in range(arr.ndim)]
            block = arr[tuple(s)]
            out[coord] = eval('np.'+funcname)(block)
        return out

    bw_func = getattr(blockwise, 'bw_'+funcname)

    arr_shape, block_shape = (4, 6), (2, 3)
    arr = np.arange(np.prod(arr_shape)).reshape(arr_shape)
    ref = bruteforce_bw(arr, block_shape)
    test = bw_func(arr, block_shape)
    assert np.array_equal(ref, test)


def test_bw_corrcoef():

    def bruteforce_bw(arr1, arr2, block_shape):
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

    arr_shape, block_shape = (4, 6), (2, 3)
    arr1 = np.arange(np.prod(arr_shape)).reshape(arr_shape)
    arr2 = np.roll(arr1, 1, 0)
    ref = bruteforce_bw(arr1, arr2, block_shape)
    test = blockwise.bw_corrcoef(arr1, arr2, block_shape)
    assert np.allclose(ref, test)
