"""Statistical correctness of the projected-ball pushforward kernels."""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta, kstest

from popsregression._projected_ball import (
    log_norm_constant,
    projected_ball_logpdf,
    projected_ball_pdf,
    projected_ball_variance,
    smooth_log,
)


def _ball_pushforward_samples(n_dim, half_width, n_samples, seed):
    """Monte Carlo pushforward: phi @ theta for theta uniform on a ball.

    By rotational symmetry this equals half_width times the first
    coordinate of a uniform draw from the unit n_dim-ball.
    """
    rng = np.random.RandomState(seed)
    g = rng.randn(n_samples, n_dim)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    radius = rng.uniform(size=n_samples) ** (1.0 / n_dim)
    return half_width * g[:, 0] * radius


@pytest.mark.parametrize("n_dim", [2, 5, 9])
def test_pushforward_density_matches_mc(n_dim):
    """MC histogram of phi @ theta matches C_P/a (1 - t^2/a^2)^k."""
    half_width = 1.7
    t = _ball_pushforward_samples(n_dim, half_width, 1_000_000, seed=n_dim)

    # t/a has density C_P (1-u^2)^k on (-1, 1), i.e. (u+1)/2 is
    # Beta((P+1)/2, (P+1)/2) distributed.
    alpha = 0.5 * (n_dim + 1)
    result = kstest((t / half_width + 1.0) / 2.0, beta(alpha, alpha).cdf)
    assert result.pvalue > 1e-3

    # Variance of the pushforward is a^2 / (P + 2), to 1%.
    expected = projected_ball_variance(half_width, n_dim)
    assert np.var(t) == pytest.approx(expected, rel=1e-2)


@pytest.mark.parametrize("n_dim", [1, 2, 5, 9, 200])
def test_density_normalization(n_dim):
    """The pdf integrates to 1 (gammaln keeps large P stable)."""
    half_width = 0.8
    total = quad(
        lambda t: projected_ball_pdf(t, half_width, n_dim),
        -half_width,
        half_width,
    )[0]
    assert total == pytest.approx(1.0, rel=1e-6)
    assert np.isfinite(log_norm_constant(n_dim))


def test_logpdf_support():
    logp = projected_ball_logpdf(np.array([-2.0, 0.0, 2.0]), 1.0, 4)
    assert np.isneginf(logp[0]) and np.isneginf(logp[2])
    assert np.isfinite(logp[1])


def test_smooth_log_matches_log_above_threshold():
    q = np.array([0.011, 0.5, 1.0, 3.0])
    value, grad, hess = smooth_log(q, 1e-2)
    np.testing.assert_allclose(value, np.log(q), rtol=1e-14)
    np.testing.assert_allclose(grad, 1.0 / q, rtol=1e-14)
    np.testing.assert_allclose(hess, -1.0 / q**2, rtol=1e-14)


def test_smooth_log_c1_continuity():
    """Value, first and second derivative are continuous at q = rho."""
    rho = 1e-2
    eps = 1e-10
    below = smooth_log(np.array([rho - eps]), rho)
    above = smooth_log(np.array([rho + eps]), rho)
    for lo, hi in zip(below, above):
        np.testing.assert_allclose(lo, hi, rtol=1e-6)


def test_smooth_log_finite_for_negative_q():
    value, grad, hess = smooth_log(np.array([-1e6, -1.0, 0.0]), 1e-3)
    assert np.all(np.isfinite(value))
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(hess))
    # The barrier keeps pushing: gradient grows as q decreases.
    assert grad[0] > grad[1] > grad[2] > 0


def test_smooth_log_converges_to_log():
    """L_rho(q) -> log(q) pointwise as rho -> 0 for q > 0."""
    q = np.array([1e-3, 0.1, 1.0])
    for rho in [1e-1, 1e-2, 1e-4]:
        value, _, _ = smooth_log(q, rho)
        mask = q >= rho
        np.testing.assert_allclose(value[mask], np.log(q[mask]), rtol=1e-12)
