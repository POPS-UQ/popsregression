"""Tests of the loss-fluctuation identities used in the moment-condition study."""

import numpy as np
from comparisons.fluctuations import (
    GAUSSIAN_CALIBRATED_GAP,
    bennett_cgf,
    calibrated_projected_ball_gap,
    cgf,
    jensen_gap,
    soft_max,
)
from numpy.testing import assert_allclose
from scipy.stats import norm


def test_cgf_is_below_its_linear_jensen_bound():
    rng = np.random.RandomState(0)
    losses = rng.gamma(2.0, 1.0, size=5000) - 1.0
    t = np.linspace(0.01, 1.0, 25)
    psi = cgf(losses, t)
    assert np.all(psi >= -1e-12)
    assert np.all(psi <= t * jensen_gap(losses) + 1e-12)
    assert_allclose(psi[-1], jensen_gap(losses))


def test_calibrated_gaussian_gap():
    rng = np.random.RandomState(1)
    y = rng.randn(400000)
    losses = -norm.logpdf(y)
    assert_allclose(jensen_gap(losses), GAUSSIAN_CALIBRATED_GAP, atol=5e-3)


def test_calibrated_projected_ball_gap_limits():
    assert_allclose(calibrated_projected_ball_gap(1), 0.0, atol=1e-10)
    gaps = [calibrated_projected_ball_gap(p) for p in (2, 5, 35, 400)]
    assert np.all(np.diff(gaps) > 0)
    assert_allclose(gaps[-1], GAUSSIAN_CALIBRATED_GAP, atol=2e-3)


def test_bennett_bound_dominates_bounded_below_losses():
    rng = np.random.RandomState(2)
    losses = rng.exponential(1.0, size=200000)  # minimum loss a = 0, heavy upper tail
    G, v = losses.mean(), losses.var()
    t = np.linspace(0.05, 1.0, 20)
    assert np.all(cgf(losses, t) <= bennett_cgf(t, v, G - 0.0) + 1e-3)


def test_soft_max_between_mean_and_max():
    values = np.array([0.1, 0.2, 1.0])
    for lam in (0.1, 1.0, 100.0):
        s = soft_max(values, lam)
        assert values.mean() - 1e-12 <= s <= values.max() + 1e-12
