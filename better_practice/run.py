import h5py
import dask.array as da
from dask.diagnostics import Profiler, ResourceProfiler, visualize, ProgressBar

import numpy as np
from blockwise_view import blockwise_view


def corrcoef(arr1, arr2, block_shape):
    # make blocks
    b1 = blockwise_view(
        arr1, block_shape, require_aligned_blocks=False, aslist=False
    )
    b2 = blockwise_view(
        arr2, block_shape, require_aligned_blocks=False, aslist=False
    )
    # subtract mean
    axes = tuple(np.arange(b1.ndim, dtype=int)[b1.ndim//2:])
    b1 -= b1.mean(axis=axes, keepdims=True)
    b2 -= b2.mean(axis=axes, keepdims=True)
    # numerator of corrcoef
    numerator = np.multiply(b1, b2).mean(axis=axes, keepdims=False)
    # denomenator of corrcoef
    dof = np.prod( b1.shape[slice(axes[0], axes[-1]+1)] )
    b1_std = np.sqrt( (b1**2).mean(axis=axes, keepdims=False) / dof )
    b2_std = np.sqrt( (b2**2).mean(axis=axes, keepdims=False) / dof )
    denominator = np.multiply(b1_std, b2_std)
    # divide
    out = np.divide(numerator, denominator)
    return out


if __name__ == '__main__':
    f1 = h5py.File("test.h5", "r")
    f2 = h5py.File("test2.h5", "r")
    arr1 = da.from_array(f1["arr"])
    arr2 = da.from_array(f2["arr"])

    block_shape = (10, 10)

    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof,\
            ProgressBar():
        out = da.map_blocks(corrcoef, arr1, arr2, block_shape,
                chunks=(400, 400))
        da.to_hdf5("out.h5", "/arr", out)
    visualize([prof, rprof])
