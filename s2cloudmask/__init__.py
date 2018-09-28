import numpy as np
import xarray as xr
import logging
import joblib
import os

from typing import Union, Optional
from skimage.morphology import opening, square

CWD = os.path.dirname(__file__)
BANDNAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

LOG = logging.getLogger(__name__)

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


def cosine(X: np.array, Y: np.array) -> np.array:
    nX = 1 / np.sqrt(np.sum(np.square(X), axis=2))
    nY = 1 / np.sqrt(np.sum(np.square(Y), axis=2))
    XX = np.einsum("ij,ijk->ijk", nX, X)
    YY = np.einsum("ij,ijk->ijk", nY, Y)
    return 1.0 - np.einsum("ijk,ijk->ij", XX, YY)


def euclidean(X: np.array, Y: np.array) -> np.array:
    return np.sqrt(np.sum(np.square(X - Y), axis=2))


def braycurtis(X: np.array, Y: np.array) -> np.array:
    return np.sum(np.absolute(X - Y), axis=2) / np.sum(np.absolute(X + Y), axis=2)


def nldr(X: np.array, Y: np.array, i: int = 0, j: int = 1) -> np.array:
    XA, XB = X[:, :, i], X[:, :, j]
    YA, YB = Y[:, :, i], Y[:, :, j]
    numerX = 2 * (XA ** 2 + XB ** 2) + 1.5 * XB + 0.5 * XA
    denomX = XA + XB + 0.5
    numerY = 2 * (YA ** 2 + YB ** 2) + 1.5 * YB + 0.5 * YA
    denomY = YA + YB + 0.5
    return numerX / denomX - numerY / denomY


def features(obs, ftrexprs, ref=None):
    bandnos = {b: i for i, b in enumerate(BANDNAMES)}
    ffexprs = []
    for f in ftrexprs:
        if f.startswith("nldr"):
            for b in BANDNAMES:
                f = f.replace(b, str(bandnos[b]))
        ffexprs.append(f)
    env = {
        **{b: obs[:, :, bandnos[b]] for b in BANDNAMES},
        **{s: getattr(np, s) for s in dir(np)},
        **{
            "cosine": cosine,
            "euclidean": euclidean,
            "braycurtis": braycurtis,
            "nldr": nldr,
            "ref": ref,
            "obs": obs,
        },
    }
    return np.stack([eval(e, {"__builtins__": {}}, env) for e in ffexprs], axis=-1)


def cloud_probs(data, model="spectral", ref=None):
    cm = MODELS[model]()
    if len(data.shape) == 3:
        return cm.predict_proba(data, ref=ref)
    elif len(data.shape) == 4:
        probs = np.empty((data.shape[0], data.shape[1], data.shape[3]), dtype=np.float32)
        for t in range(data.shape[3]):
            obs = data[:, :, :, t]
            prb = cm.predict_proba(obs, ref=ref)
            probs[:, :, t] = prb.reshape(data.shape[0], data.shape[1])
        return probs
    else:
        raise DataDimensionalityError("data must have 3 or 4 dimensions.")


def shadow_probs(data, model="fast-shadow", ref=None):
    cm = MODELS[model]()
    if len(data.shape) == 3:
        return cm.predict_proba(data, ref=ref)
    elif len(data.shape) == 4:
        probs = np.empty((data.shape[0], data.shape[1], data.shape[3]), dtype=np.float32)
        for t in range(data.shape[3]):
            obs = data[:, :, :, t]
            prb = cm.predict_proba(obs, ref=ref)
            probs[:, :, t] = prb.reshape(data.shape[0], data.shape[1])
        return probs
    else:
        raise DataDimensionalityError("data must have 3 or 4 dimensions.")


def cloud_mask(data, model="spectral", ref=None):
    probs = cloud_probs(data, model=model, ref=ref)
    return probs > 0.5


def shadow_mask(data, model="fast-shadow", ref=None):
    probs = cloud_probs(data, model=model, ref=ref)
    return probs > 0.5


class DataDimensionalityError(ValueError):
    pass


def mask_cloud_as_nan(data, model="spectral", ref=None):
    LOG.debug("loading %s model", model)
    cm = MODELS[model]()
    LOG.debug("Masking observations")
    if len(data.shape) == 3:
        prob = cm.predict_proba(data, ref=ref)
        data[prob > 0.5] = np.nan
    elif len(data.shape) == 4:
        probs = np.empty((data.shape[0], data.shape[1], data.shape[3]), dtype=np.float32)
        for t in range(data.shape[3]):
            if t % 25 == 0:
                LOG.debug(
                    "Masking clouds in observation %s/%s containing %s pixels",
                    t,
                    data.shape[3],
                    data.shape[0] * data.shape[1],
                )
            obs = data[:, :, :, t]
            prb = cm.predict_proba(obs, ref=ref).reshape(data.shape[0], data.shape[1])
            obs[prb > 0.5] = np.nan
    else:
        raise DataDimensionalityError("data must have 3 or 4 dimensions.")


def mask_shadow_as_nan(data, model="fast-shadow", ref=None):
    LOG.debug("loading %s model", model)
    cm = MODELS[model]()
    LOG.debug("Masking observations")
    if len(data.shape) == 3:
        prob = cm.predict_proba(data, ref=ref)
        data[prob > 0.5] = np.nan
    elif len(data.shape) == 4:
        probs = np.empty((data.shape[0], data.shape[1], data.shape[3]), dtype=np.float32)
        for t in range(data.shape[3]):
            if t % 25 == 0:
                LOG.debug(
                    "Masking shadows in observation %s/%s containing %s pixels",
                    t,
                    data.shape[3],
                    data.shape[0] * data.shape[1],
                )
            obs = data[:, :, :, t]
            prb = cm.predict_proba(obs, ref=ref).reshape(data.shape[0], data.shape[1])
            obs[prb > 0.5] = np.nan
    else:
        raise DataDimensionalityError("data must have 3 or 4 dimensions.")


def mask_cloud_and_shadow_as_nan(data, cloudmodel="fast", shadowmodel="fast-shadow", ref=None):
    mask_shadow_as_nan(data, model=shadowmodel, ref=ref)
    mask_cloud_as_nan(data, model=cloudmodel, ref=ref)


class Classifier:
    def __init__(self):
        self.ftrexprs = None
        self.model = None


class SpectralCloudClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model, self.ftrexprs = joblib.load(os.path.join(CWD, "models", "spectral-model.pkl.xz"))

    def predict_proba(self, X: np.array, ref: Optional[np.array] = None) -> np.array:
        if len(X.shape) == 4 and X.shape[2] == 10:  # temporal stack (y,x,bands,time)
            XX = np.transpose(X, [0, 1, 3, 2]).reshape((X.shape[0], X.shape[1] * X.shape[3], X.shape[2]))
            ftr = features(XX, self.ftrexprs, ref=ref)
            prob = self.model.predict_proba(ftr.reshape((-1, ftr.shape[2])))[:, 1]
            prob = prob.reshape((X.shape[0], X.shape[1], X.shape[3]))
            return prob
        if len(X.shape) != 3 and X.shape[2] != 10:
            raise DataDimensionalityError(
                "Data shape should have length 3 and last dim should be equal to 10."
            )
        ftr = features(X, self.ftrexprs, ref=ref)
        prob = self.model.predict_proba(ftr.reshape((-1, ftr.shape[2])))[:, 1].reshape(X.shape[:2])
        opening(prob, square(3), out=prob)
        return prob

    def predict(self, X: np.array, ref: Optional[np.array] = None) -> np.array:
        prob = self.predict_proba(X)
        return prob > 0.5


class FastCloudClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model, self.ftrexprs = joblib.load(os.path.join(CWD, "models", "fast-spectral-model.pkl.xz"))

    def predict_proba(self, X: np.array, ref: Optional[np.array] = None) -> np.array:
        if len(X.shape) == 4 and X.shape[2] == 10:  # temporal stack (y,x,bands,time)
            XX = np.transpose(X, [0, 1, 3, 2]).reshape((X.shape[0], X.shape[1] * X.shape[3], X.shape[2]))
            ftr = features(XX, self.ftrexprs, ref=ref)
            ftr = ftr.reshape((-1, ftr.shape[2]))
            good = ~np.isnan(ftr).any(axis=1)
            prob = np.nan * np.ones((X.shape[0], X.shape[1]), dtype=np.float32).ravel()
            prob[good] = self.model.predict_proba(ftr[good])[:, 1]
            return prob.reshape((X.shape[0], X.shape[1]))
        if len(X.shape) != 3 and X.shape[2] != 10:
            raise DataDimensionalityError(
                "Data shape should have length 3 and last dim should be equal to 10."
            )
        ftr = features(X, self.ftrexprs, ref=ref)
        ftr = ftr.reshape((-1, ftr.shape[2]))
        good = ~np.isnan(ftr).any(axis=1)
        prob = np.nan * np.ones((X.shape[0], X.shape[1]), dtype=np.float32).ravel()
        prob[good] = self.model.predict_proba(ftr[good])[:, 1]
        prob = prob.reshape((X.shape[0], X.shape[1]))
        opening(prob, square(3), out=prob)
        prob[np.isnan(X).any(axis=2)] = np.nan
        return prob

    def predict(self, X: np.array, ref: Optional[np.array] = None) -> np.array:
        prob = self.predict_proba(X)
        return prob > 0.5


class TemporalCloudClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model, self.ftrexprs = joblib.load(os.path.join(CWD, "models", "temporal-model.pkl.xz"))

    def predict_proba(self, X: np.array, ref: np.array) -> np.array:
        if len(X.shape) != 3 and X.shape[2] != 10:
            raise DataDimensionalityError(
                "Data shape should have length 3 and last dim should be equal to 10."
            )
        ftr = features(X, self.ftrexprs, ref=ref)
        prob = self.model.predict_proba(ftr.reshape((-1, ftr.shape[2])))[:, 1].reshape(X.shape[:2])
        opening(prob, square(3), out=prob)
        return prob

    def predict(self, X: np.array, ref: np.array) -> np.array:
        prob = self.predict_proba(X, ref)
        return prob > 0.5


class FastShadowClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model, self.ftrexprs = joblib.load(os.path.join(CWD, "models", "fast-shadow-model.pkl.xz"))

    def predict_proba(self, X: np.array, ref: Optional[np.array] = None) -> np.array:
        if len(X.shape) == 4 and X.shape[2] == 10:  # temporal stack (y,x,bands,time)
            XX = np.transpose(X, [0, 1, 3, 2]).reshape((X.shape[0], X.shape[1] * X.shape[3], X.shape[2]))
            ftr = features(XX, self.ftrexprs, ref=ref)
            good = ~np.isnan(ftr).any(axis=2)
            prob = np.nan * np.ones((X.shape[0], X.shape[1]), dtype=np.float32)
            prob[good] = self.model.predict_proba(ftr[good].reshape((-1, ftr.shape[2])))[:, 1]
            return prob
        if len(X.shape) != 3 and X.shape[2] != 10:
            raise DataDimensionalityError(
                "Data shape should have length 3 and last dim should be equal to 10."
            )
        ftr = features(X, self.ftrexprs, ref=ref)
        good = ~np.isnan(ftr).any(axis=2)
        prob = np.nan * np.ones((X.shape[0], X.shape[1]), dtype=np.float32)
        prob[good] = self.model.predict_proba(ftr[good].reshape((-1, ftr.shape[2])))[:, 1].reshape(
            X.shape[:2]
        )
        return prob

    def predict(self, X: np.array, ref: Optional[np.array] = None) -> np.array:
        prob = self.predict_proba(X)
        return prob > 0.5


MODELS = {
    "fast": FastCloudClassifier,
    "fast-shadow": FastShadowClassifier,
    "spectral": SpectralCloudClassifier,
    "temporal": TemporalCloudClassifier,
}
