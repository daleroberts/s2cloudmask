# s2cloudmask

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Release](https://img.shields.io/badge/Release-Private-ff69b4.svg)

The [s2cloudmask](https://github.com/daleroberts/s2cloudmask) Python package provides Machine Learning classifiers for **Cloud Detection** in [Sentinel-2](https://en.wikipedia.org/wiki/Sentinel-2) observations. The aim of this package is to open-source and demonstrate some of the tools being developed as part of the [Digital Earth Australia](https://www.ga.gov.au/dea) initiative and to push the state-of-art in the area of cloud classification.

The package currently provides two classifiers:
 * A spectral pixel-based approach
 * A spectral-temporal pixel-based approach

The *spectral classifier* is useful if you only have a couple of observations (i.e., satellite images) while the the *spectral-temporal classifier* (aka. *temporal classifier*) gives a better classification of clouds provided that you can supply it with a geomedian pixel-composite mosaic [Roberts et al. 2017] of the area (or a stack of data so that one can be created by this package).

We note the existence of Python package [s2cloudless](https://github.com/sentinel-hub/sentinel2-cloud-detector) developed by [Sentinel Hub](https://www.sentinel-hub.com/)'s research team that, as they argue in their [blog post](https://medium.com/sentinel-hub/improving-cloud-detection-with-machine-learning-c09dc5d7cf13), "didn't observe significant improvement using derived features instead of raw band values" so their "final classifier uses the following 10 bands as input: B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12". By releasing this package, we argue the contrary and demonstrate that this is incorrect and you can obtain a better classification of clouds by (thinking hard and) developing new derived features.

### Installation

```
$ pip install git+https://github.com/daleroberts/s2cloudmask
```

### Easy interface

This package has an easy interface. Given a numpy array `obs` ordered as (y,x,band) we can obtain a cloud `mask`.
```
>>> import s2cloudmask as s2cm
>>> mask = s2cm.cloud_mask(obs, model='spectral')
```

### References

 * Roberts, D., Mueller, N., McIntyre, A. (2017). [High-dimensional pixel composites from Earth observation time series](https://ieeexplore.ieee.org/document/8004469). *IEEE Transactions on Geoscience and Remote Sensing*, PP, 99. pp. 1--11.


<img src="http://www.ga.gov.au/__data/assets/image/0011/61589/GA-DEA-Logo-Inline-370x73.png" width=370>
