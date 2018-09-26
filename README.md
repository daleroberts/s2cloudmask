# s2cloudmask

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Release](https://img.shields.io/badge/Release-Private-ff69b4.svg)

The **s2cloudmask** Python package provides classifiers for clouds in Sentinel-2 observations. The package currently provides two classifiers:
 * A spectral pixel-based approach
 * A spectral-temporal pixel-based approach

The *spectral classifier* is useful if you only have a couple of observations (i.e., satellite images) while the the *spectral-temporal classifier* (aka. *temporal classifier*) gives a better classification of clouds provided that you can supply it with a geomedian pixel-composite mosaic [Roberts et al. 2018] of the area or a stack of data so that one can be created.

The aim of this package is to open-source and demonstrate some of the tools being developed as part of the [Digital Earth Australia](https://www.ga.gov.au/dea) initiative and to push the state-of-art in the area of cloud classification.


