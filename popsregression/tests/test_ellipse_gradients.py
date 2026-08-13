"""Finite-difference exactness tests for the ellipse objective."""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from popsregression._ellipse import _ellipse_nll, _ellipse_nll_hess_diag

N, P, R = 40, 7, 3


def _make_problem(seed, scale):
    """Random small problem; scale sets which L_rho branch dominates."""
    rng = np.random.RandomState(seed)
    Z = rng.randn(N, P)
    y = 2.0 * rng.randn(N)
    b0 = 0.1 * rng.rand(N)
    weights = 0.5 + rng.rand(N)
    psi = np.concatenate([scale * rng.randn(P), scale * rng.randn(P * R)])
    psi0 = 0.1 * rng.randn(P * (1 + R))
    return Z, y, b0, weights, psi, psi0


def _central_diff_grad(fun, psi, h=1e-6):
    grad = np.zeros_like(psi)
    for j in range(psi.size):
        e = np.zeros_like(psi)
        e[j] = h
        grad[j] = (fun(psi + e) - fun(psi - e)) / (2.0 * h)
    return grad


# Small parameter scale leaves most q_i < rho (continued branch); large
# scale puts most q_i on the exact-log branch.
BRANCHES = [(1e-1, 0.05), (1e-1, 2.0), (1e-3, 2.0), (1e-3, 0.05)]


@pytest.mark.parametrize("rho,scale", BRANCHES)
def test_gradient_exactness(rho, scale):
    Z, y, b0, weights, psi, psi0 = _make_problem(3, scale)
    delta = 1e-2

    _, grad = _ellipse_nll(psi, Z, y, b0, weights, delta, rho)
    fd = _central_diff_grad(
        lambda p: _ellipse_nll(p, Z, y, b0, weights, delta, rho)[0], psi
    )
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("rho,scale", BRANCHES)
def test_gradient_exactness_with_prior(rho, scale):
    Z, y, b0, weights, psi, psi0 = _make_problem(7, scale)
    delta = 1e-2
    prec = 0.7

    _, grad = _ellipse_nll(psi, Z, y, b0, weights, delta, rho, psi0, prec)
    fd = _central_diff_grad(
        lambda p: _ellipse_nll(p, Z, y, b0, weights, delta, rho, psi0, prec)[0],
        psi,
    )
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("rho,scale", BRANCHES)
def test_hessian_diagonal_exactness(rho, scale):
    """Diagonal Hessian formulas vs finite differences of the gradient."""
    Z, y, b0, weights, psi, _ = _make_problem(11, scale)
    delta = 1e-2

    hess_diag = _ellipse_nll_hess_diag(psi, Z, y, b0, weights, delta, rho)

    h = 1e-6
    fd = np.zeros_like(psi)
    for j in range(psi.size):
        e = np.zeros_like(psi)
        e[j] = h
        _, gp = _ellipse_nll(psi + e, Z, y, b0, weights, delta, rho)
        _, gm = _ellipse_nll(psi - e, Z, y, b0, weights, delta, rho)
        fd[j] = (gp[j] - gm[j]) / (2.0 * h)
    np.testing.assert_allclose(hess_diag, fd, rtol=1e-5, atol=1e-6)


def test_zero_weights_drop_points():
    """Zero-weighted points must not contribute to value or gradient."""
    Z, y, b0, weights, psi, _ = _make_problem(5, 1.0)
    delta = 1e-2
    rho = 1e-2

    weights = np.ones(N)
    weights[N // 2 :] = 0.0
    val, grad = _ellipse_nll(psi, Z, y, b0, weights, delta, rho)
    val_sub, grad_sub = _ellipse_nll(
        psi,
        Z[: N // 2],
        y[: N // 2],
        b0[: N // 2],
        np.ones(N // 2),
        delta,
        rho,
    )
    np.testing.assert_allclose(val, val_sub, rtol=1e-12)
    np.testing.assert_allclose(grad, grad_sub, rtol=1e-12, atol=1e-14)
