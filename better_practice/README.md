# Better practice

Use `blockwise_view.py` from `lazyflow` module of `ilastik` project ([link](https://github.com/ilastik/ilastik/blob/master/lazyflow/utility/blockwise_view.py)) and calculate correlation coefficient with numpy basic operations for vectorized performance. Minimal RAM and only moderate CPU usage on a machine with quad-core Intel i7-10510U and 16 GB RAM.

## files here
* `blockwise_view.py`  
   Neither `lazyflow` or `ilastik` can be found on [PyPI](https://pypi.org) so I posted an [issue](https://github.com/ilastik/ilastik/issues/2275) asking if it's possible to re-use this function programatically.
* `gen_data.py`  
   Generate large test data. The `test2.h5` input in `run.py` was a plain copy of `test.h5`.
* `run.py`  
   Demo code.
