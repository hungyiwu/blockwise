import dask.array as da

arr_shape, chunk_shape = (int(1e5), int(1e5)), (1000, 1000)
dtype = float
arr = da.ones(shape=arr_shape, dtype=dtype, chunks=chunk_shape)
da.to_hdf5('test.h5', '/arr', arr)
