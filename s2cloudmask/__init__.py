import numpy as np
import xarray as xr

from typing import Union, Optional

Data = Union[xr.Dataset, np.array]


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
