"""
Tests.
"""

import numpy as np
import s2cloudmask as s2cm
import joblib
import os

from numpy.testing import assert_equal, assert_array_almost_equal

CWD = os.path.dirname(__file__)
CLEAR = joblib.load(os.path.join(CWD, 'data', 'clear-wright-20160101.pkl.xz'))
REF = joblib.load(os.path.join(CWD, 'data', 'ref-wright.pkl.xz'))

def test_spectral_classifier():
        obs = CLEAR.copy()
        scc = s2cm.SpectralCloudClassifier()
        cloud = scc.predict_proba(obs)
        assert_equal(np.count_nonzero(cloud>0.5), 0)

def test_fast_cloud_classifier():
        obs = CLEAR.copy()
        scc = s2cm.FastCloudClassifier()
        cloud = scc.predict_proba(obs)
        assert_equal(cloud.shape[0], obs.shape[0])
        assert_equal(cloud.shape[1], obs.shape[1])
        assert(np.count_nonzero(cloud>0.5) < 1000)

def test_temporal_classifer():
        obs = CLEAR.copy()
        ref = REF
        tcc = s2cm.TemporalCloudClassifier()
        cloud = tcc.predict_proba(obs, ref)
        assert_equal(np.count_nonzero(cloud>0.5), 0)

def test_spectral_easy_probs():
        obs = CLEAR.copy()
        obss = np.stack([obs, obs], axis=-1)
        probs = s2cm.cloud_probs(obss, model='spectral')
        assert_equal(np.count_nonzero(probs>0.5), 0)

def test_spectral_easy_mask():
        obs = CLEAR.copy()
        obss = np.stack([obs, obs], axis=-1)
        mask = s2cm.cloud_mask(obss, model='spectral')
        assert_equal(np.count_nonzero(mask), 0)

def test_fast_easy_mask():
        obs = CLEAR.copy()
        mask = s2cm.cloud_mask(obs, model='fast')
        assert_equal(np.count_nonzero(mask), 0)

def test_fast_handle_nans():
        obs = CLEAR.copy()
        obs[10, 10, :] = np.nan
        prob = s2cm.cloud_probs(obs, model='fast')
        assert_equal(prob[10, 10], np.nan)
        mask = s2cm.cloud_mask(obs, model='fast')
        assert_equal(np.count_nonzero(mask), 0)

def test_spectral_mask_as_nan():
        obs = CLEAR.copy()
        mask = s2cm.cloud_mask(obs, model='spectral')
        mobs = obs.copy()
        s2cm.mask_cloud_as_nan(mobs, model='spectral')
        nanmask = np.isnan(mobs).all(axis=2)
        assert_equal(mask, nanmask)
