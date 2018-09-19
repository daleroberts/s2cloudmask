import numpy as np
import xarray as xr

from typing import Union, Optional

Data = Union[xr.Dataset, np.array]


try:
    # Import the superfast version of geomedian pixel composites. 
    # Contact Dale Roberts <dale.roberts@anu.edu.au> if you want 
    # to discuss GitHub repo access to the development version.
    from pcm import gmpcm

except ImportError:
    # Use the slow public version
    from hdmedians import nangeomedian

    def gmpcm(data):
        result = np.empty(data.shape[:3], dtype=np.float32)
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                result[y, x, :] = nangeomedian(
                    data[y, x, :, :].astype(np.float32), axis=1, eps=1e-4, maxiters=2000
                )
        return result


class Classifier:
    def __init__(self):
        pass

    def mask_as_nan(self, data: Data, out: Optional[Data] = None) -> Optional[Data]:
        pass

    def prob_cloud(self, data: Data, out: Optional[Data] = None) -> Optional[Data]:
        pass

    def classify(self, data: Data, out: Optional[Data] = None) -> Optional[Data]:
        pass


class SpectralCloudClassifier(Classifier):
    def __init__(self):
        pass


class TemporalCloudClassifier(Classifier):
    def __init__(self):
        pass
