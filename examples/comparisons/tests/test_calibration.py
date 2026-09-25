"""Tests of the calibration curves and areas."""

import numpy as np
from comparisons.calibration import (
    error_pp_curve,
    group_weights,
    pit_curve,
    step_curve_areas,
)
from numpy.testing import assert_allclose
from scipy.integrate import quad


def test_step_curve_areas_match_quadrature():
    rng = np.random.RandomState(0)
    knots = np.concatenate([[0.0], np.sort(rng.rand(7)), [1.0]])
    values = rng.rand(8)

    def curve(u):
        k = np.searchsorted(knots, u, side="left") - 1
        return values[max(k, 0)]

    signed, unsigned = step_curve_areas(knots, values)
    pts = list(knots)
    assert_allclose(
        signed, quad(lambda u: curve(u) - u, 0, 1, points=pts)[0], atol=1e-8
    )
    assert_allclose(
        unsigned,
        quad(lambda u: abs(curve(u) - u), 0, 1, points=pts, limit=200)[0],
        atol=1e-8,
    )


def test_calibrated_predictive_has_small_areas():
    rng = np.random.RandomState(1)
    n = 4000
    scale = rng.uniform(0.5, 2.0, n)
    y = scale * rng.randn(n)
    draws = scale[:, None] * rng.randn(n, 200)
    _, _, s_pp, a_pp = error_pp_curve(y, draws)
    assert abs(s_pp) < 0.02 and a_pp < 0.02
    from scipy.stats import norm

    _, _, s_pit, a_pit = pit_curve(norm.cdf(y / scale))
    assert abs(s_pit) < 0.02 and a_pit < 0.02


def test_overconfident_predictive_has_positive_signed_area():
    rng = np.random.RandomState(2)
    y = rng.randn(3000)
    draws = 0.3 * rng.randn(3000, 100)  # far too narrow
    _, _, signed, unsigned = error_pp_curve(y, draws)
    assert signed > 0.2
    assert_allclose(unsigned, signed, atol=1e-3)  # C(u) >= u almost everywhere


def test_signed_area_can_vanish_without_agreement():
    """Zero signed area does not imply equal distributions."""
    pit = np.concatenate([np.full(500, 0.25), np.full(500, 0.75)])
    _, _, signed, unsigned = pit_curve(pit)
    assert abs(signed) < 1e-12
    assert unsigned > 0.1


def test_group_weights_equalize_groups():
    groups = np.array([0, 0, 0, 1])
    w = group_weights(groups)
    assert_allclose([w[:3].sum(), w[3:].sum()], [0.5, 0.5])
    assert group_weights(None) is None
