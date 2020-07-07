import itertools

import numpy as np
import dask.array as da
import pytest

import blockwise


@pytest.mark.parametrize(
    "arr_shape,fill_value", [((3, 3), -1), ((3, 3), -2)],
)
def test_shift(arr_shape, fill_value):
    # less critical params
    seed = 42
    low, high = 0, 2
    # start test
    da.random.seed(seed)
    arr = da.random.randint(low=low, high=high, size=arr_shape)
    for ax in range(arr.ndim):
        d = arr_shape[ax]
        filled_slice = [":", ] * arr.ndim
        kept_slice = [":", ] * arr.ndim
        for num in range(-(d - 1), d):
            shifted = blockwise.shift(arr, num, ax, fill_value=fill_value)

            filled_slice = [":", ] * arr.ndim
            kept_slice = [":", ] * arr.ndim
            new_slice = [":", ] * arr.ndim

            if num == 0:
                assert da.allclose(arr, shifted)
            elif num > 0:
                kept_slice[ax] = "0:{}".format(-num)
                new_slice[ax] = "{}:".format(num)

                ref = eval("arr[" + ", ".join(kept_slice) + "]")
                test = eval("shifted[" + ", ".join(new_slice) + "]")

                assert da.allclose(ref, test)

                filled_slice[ax] = "0:{}".format(num)
                filled_shape = arr_shape[:ax] + (num,) + arr_shape[ax+1:]

                ref = da.full(shape=filled_shape, fill_value=fill_value)
                test = eval("shifted[" + ", ".join(filled_slice) + "]")

                assert da.allclose(ref, test)
            else:
                kept_slice[ax] = "{}:".format(-num)
                new_slice[ax] = ":{}".format(num)

                ref = eval("arr[" + ", ".join(kept_slice) + "]")
                test = eval("shifted[" + ", ".join(new_slice) + "]")

                assert da.allclose(ref, test)

                filled_slice[ax] = "{}:".format(num)
                filled_shape = arr_shape[:ax] + (-num,) + arr_shape[ax+1:]

                ref = da.full(shape=filled_shape, fill_value=fill_value)
                test = eval("shifted[" + ", ".join(filled_slice) + "]")

                assert da.allclose(ref, test)


def test_repeat_block():
    arr = da.arange(4).reshape((2, 2))
    ref = da.from_array(np.array([[0, 0, 1, 1], ] * 3 + [[2, 2, 3, 3], ] * 3))
    test = blockwise.repeat_block(arr, (3, 2))
    assert da.allclose(ref, test)


def test_trim():
    arr_shape, block_shape = (6, 5), (3, 2)
    arr = da.arange(np.prod(arr_shape)).reshape(arr_shape)
    test = blockwise.trim(arr, block_shape)
    ref = arr[:, :4]
    assert da.allclose(ref, test)


@pytest.mark.parametrize('funcname', ['sum', 'mean', 'std'])
def test_bw_func(funcname):
    # params perhaps less critical, can be parametrized in future
    arr_shape, block_shape = (4, 6), (2, 3)

    # straightforward implementation, takes np.ndarray
    def bruteforce_bw(arr, block_shape):
        out_shape = [arr.shape[ax] // block_shape[ax]
                     for ax in range(arr.ndim)]
        out = np.empty(out_shape)
        for coord in itertools.product(*[range(d) for d in out_shape]):
            s = [slice(coord[ax] * block_shape[ax],
                 (coord[ax] + 1) * block_shape[ax])
                 for ax in range(arr.ndim)]
            block = arr[tuple(s)]
            out[coord] = eval("np." + funcname)(block)
        return out

    # get function name
    bw_func = getattr(blockwise, "bw_" + funcname)

    # prepare identical data with different format
    arr_npy = np.arange(np.prod(arr_shape)).reshape(arr_shape)
    arr_da = da.from_array(arr_npy)

    # reference result
    ref_npy = bruteforce_bw(arr_npy, block_shape)
    ref_da = da.from_array(ref_npy)

    # test result
    test_da = bw_func(arr_da, block_shape)

    assert da.allclose(ref_da, test_da)


def test_bw_corrcoef():
    # params perhaps less critical, can be parametrized in future
    arr_shape, block_shape = (4, 6), (2, 3)

    # straightforward implementation, takes np.ndarray
    def bruteforce_bw(arr1, arr2, block_shape):
        out_shape = [arr1.shape[ax] // block_shape[ax]
                     for ax in range(arr1.ndim)]
        out = np.empty(out_shape)
        for coord in itertools.product(*[range(d) for d in out_shape]):
            s = [slice(coord[ax] * block_shape[ax],
                 (coord[ax] + 1) * block_shape[ax])
                 for ax in range(arr1.ndim)]
            block1 = arr1[tuple(s)]
            block2 = arr2[tuple(s)]
            out[coord] = np.corrcoef(block1.flatten(), block2.flatten())[0, 1]
        return out

    # prepare identical data with different format
    arr1_npy = np.arange(np.prod(arr_shape)).reshape(arr_shape)
    arr2_npy = np.roll(arr1_npy, 1, 0)
    arr1_da = da.from_array(arr1_npy)
    arr2_da = da.from_array(arr2_npy)

    # reference result
    ref_npy = bruteforce_bw(arr1_npy, arr2_npy, block_shape)
    ref_da = da.from_array(ref_npy)

    # test result
    test_da = blockwise.bw_corrcoef(arr1_da, arr2_da, block_shape)

    assert da.allclose(ref_da, test_da)
