# Blockwise
Fast blockwise operations of array data
## Rationale
Image registration is crucial for comparing/integrating images with spatial correspondence but don't exactly align due to complications like specimen deformation (ex. histological images) or optical distortion. There has been a thriving effort to develop powerful non-rigid image registrations (ex. [ANHIR](https://ieeexplore.ieee.org/document/9058666)), yet from an end user's perspective, the very first question is not "which algorithm to use" but rather:

> Do I need to register this particular dataset?

One quick and intuitive way to check it is to calculate the correlation coefficient. However, calculating one coefficient for the entir image does not capture local heterogeneity if the image is large (ex. 10k-by-10k pixels). A better way is to calculate one coefficient for each smaller regions (ex. [for each single cell](https://github.com/hungyiwu/tissue_integrity_dashboard)). This imposes a software engineering challenge: tasks are so small (`np.corrcoef` for two 10x10 arrays) that communication overhead roughly cancels parallelization speedup.

```
# demo code: straightforward approach, 1 core
# demo code: straightforward approach, 4 core
```

`blockwise` aims to perform blockwise operations (sum, mean, standard deviation, Pearson correlation coefficient) faster by replacing loops with vectorized operations. It's inspired by `skimage.feature.match_template` source code and [related discussion](https://github.com/dask/dask-image/pull/148#discussion_r444649473). 

```
# demo code: blockwise approach, 1 core
# demo code: blockwise approach, 4 core through dask
```
