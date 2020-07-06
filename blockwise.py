import numpy as np


def shift(arr, num, axis, fill_value=np.nan):
    """
    Shift N-dim array.

    Modified from https://stackoverflow.com/a/42642326/6826902
    """
    # make slices
    def slice_maker(replace=None):
        s = [slice(None), ] * arr.ndim
        if replace is not None:
            s[axis] = replace
        return tuple(s)

    s0 = slice_maker()
    s1 = slice_maker(slice(0, num))
    s2 = slice_maker(slice(num, None))
    s3p = slice_maker(slice(0, -num))
    s3n = slice_maker(slice(-num, None))

    # move arrays using slices
    result = np.empty_like(arr)
    if num > 0:
        result[s1] = fill_value
        result[s2] = arr[s3p]
    elif num < 0:
        result[s2] = fill_value
        result[s1] = arr[s3n]
    else:
        result[s0] = arr

    return result


def repeat_block(image, block_shape):
    """
    numpy.repeat for n-dim.
    """
    rep = image
    for ax in range(image.ndim):
        rep = np.repeat(rep, repeats=block_shape[ax], axis=ax)
    return rep


def trim(image, block_shape):
    '''
    Trim off residues.
    '''
    s_list = []
    for ax in range(image.ndim):
        res = image.shape[ax] % block_shape[ax]
        if res:
            s = slice(0, -res)
        else:
            s = slice(None)
        s_list.append(s)
    return image[tuple(s_list)]


def bw_sum(image, block_shape, keep_shape=False):
    """
    Blockwise sum.

    Modified from
    https://github.com/dask/dask-image/pull/148#discussion_r444649473
    """
    # add up and subtract shifted copy, like np.diff but shifted
    integral_image = image
    for ax in range(image.ndim):
        integral_image = np.cumsum(integral_image, axis=ax)

    window_sums = integral_image
    for ax in range(image.ndim):
        window_sums -= shift(window_sums, block_shape[ax], ax, fill_value=0)

    # now sum is at the corner of each block, slice to get it
    s = [slice(d - 1, None, d) for d in block_shape]
    window_sums = window_sums[tuple(s)]

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
    return np.sqrt(bws / (np.prod(block_shape) - ddof))


def bw_corrcoef(image1, image2, block_shape, keep_shape=False):
    """
    Blockwise Pearson correlation coefficient.
    """
    # blockwise zero-mean
    image1_zm = image1 - bw_mean(image1, block_shape, keep_shape=True)
    image2_zm = image2 - bw_mean(image2, block_shape, keep_shape=True)
    # follow Pearson correlation coefficient formula
    numerator = bw_mean(np.multiply(image1_zm, image2_zm), block_shape)
    image1_std = bw_std(image1, block_shape)
    image2_std = bw_std(image2, block_shape)
    denominator = np.multiply(image1_std, image2_std)
    bwcc = np.divide(numerator, denominator)

    if keep_shape:
        bwcc = repeat_block(bwcc, block_shape)

    return bwcc
