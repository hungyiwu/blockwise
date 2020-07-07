import numpy as np
import dask.array as da


def shift(arr, num, axis, fill_value=np.nan):
    """
    Shift N-dim array.
    """
    if not num:
        return arr.copy()

    fill_shape = arr.shape[:axis] + (abs(num), ) + arr.shape[axis+1:]
    filled = da.full(shape=fill_shape, fill_value=fill_value)

    kept_slice = [':', ] * arr.ndim
    if num > 0:
        kept_slice[axis] = '0:{}'.format(-num)
        kept = eval('arr[' + ', '.join(kept_slice) + ']')
        result = da.concatenate([filled, kept], axis=axis)
    else:
        kept_slice[axis] = '{}:'.format(-num)
        kept = eval('arr[' + ', '.join(kept_slice) + ']')
        result = da.concatenate([kept, filled], axis=axis)

    return result


def repeat_block(image, block_shape):
    """
    da.repeat for n-dim.
    """
    rep = image.copy()
    for ax in range(image.ndim):
        rep = da.repeat(rep, repeats=block_shape[ax], axis=ax)
    return rep


def trim(image, block_shape):
    '''
    Trim off residues.
    '''
    s_list = []
    for ax in range(image.ndim):
        res = image.shape[ax] % block_shape[ax]
        if res:
            s = '0:{}'.format(-res)
        else:
            s = ':'
        s_list.append(s)
    return eval('image[' + ', '.join(s_list) + ']')


def bw_sum(image, block_shape, keep_shape=False):
    """
    Blockwise sum.

    Modified from
    https://github.com/dask/dask-image/pull/148#discussion_r444649473
    """
    # add up and subtract shifted copy, like np.diff but shifted
    integral_image = image.copy()
    for ax in range(image.ndim):
        integral_image = da.cumsum(integral_image, axis=ax)

    window_sums = integral_image
    for ax in range(image.ndim):
        window_sums -= shift(window_sums, block_shape[ax], ax, fill_value=0)

    # now sum is at the corner of each block, slice to get it
    s = ['{}::{}'.format(d-1, d) for d in block_shape]
    window_sums = eval('window_sums[' + ', '.join(s) + ']')

    # repeat to get same shape back
    if keep_shape:
        window_sums = repeat_block(window_sums, block_shape)

    return window_sums


def bw_mean(image, block_shape, keep_shape=False):
    """
    Blockwise mean.
    """
    bws = bw_sum(image, block_shape, keep_shape=keep_shape)
    return bws / np.prod(block_shape)


def bw_std(image, block_shape, ddof=0, keep_shape=False):
    """
    Blockwise standard deviation.
    """
    # zero-mean
    bwm = bw_mean(image, block_shape, keep_shape=True)
    image_zm = image - bwm
    # follow standard deviation formula
    bws = bw_sum(image_zm ** 2, block_shape, keep_shape=keep_shape)
    return da.sqrt(bws / (np.prod(block_shape) - ddof))


def bw_corrcoef(image1, image2, block_shape, keep_shape=False):
    """
    Blockwise Pearson correlation coefficient.
    """
    # blockwise zero-mean
    image1_zm = image1 - bw_mean(image1, block_shape, keep_shape=True)
    image2_zm = image2 - bw_mean(image2, block_shape, keep_shape=True)
    # follow Pearson correlation coefficient formula
    numerator = bw_mean(da.multiply(image1_zm, image2_zm), block_shape)
    image1_std = bw_std(image1, block_shape)
    image2_std = bw_std(image2, block_shape)
    denominator = da.multiply(image1_std, image2_std)
    bwcc = da.divide(numerator, denominator)

    if keep_shape:
        bwcc = repeat_block(bwcc, block_shape)

    return bwcc
