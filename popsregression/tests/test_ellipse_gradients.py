"""Exactness tests for the compiled ellipse objective and its optimizer."""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from popsregression._ellipse import _fit_ellipsoid, _hess_diag, _nll
from popsregression._optimize import lbfgs
from popsregression._projected_ball import ball_constants, projected_ball_logpdf

N, P, R = 40, 7, 3
DELTA2 = 1e-4
CONSTANTS = ball_constants(P + 1)


def _make_problem(seed, scale):
    """Random small problem; scale sets which L_rho branch dominates."""
    rng = np.random.RandomState(seed)
    Z = rng.randn(N, P)
    y = 2.0 * rng.randn(N)
    b0 = 0.1 * rng.rand(N)
    weights = 0.5 + rng.rand(N)
    center = scale * rng.randn(P)
    U = scale * rng.randn(P, R)
    return center, U, (Z, y, b0, weights, DELTA2)


def _packed(data, rho):
    """Objective as a function of the packed vector ``(center, vec U)``."""

    def loss(psi):
        return _nll(psi[:P], psi[P:].reshape(P, R), *data, rho, *CONSTANTS)

    return loss


def _central_difference(fun, x, h=1e-6):
    out = []
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = h
        out.append((np.asarray(fun(x + e)) - np.asarray(fun(x - e))) / (2.0 * h))
    return np.array(out)


# Small parameter scale leaves most q_i < rho (continued branch); large
# scale puts most q_i on the exact-log branch.
BRANCHES = [(1e-1, 0.05), (1e-1, 2.0), (1e-3, 2.0), (1e-3, 0.05)]


def test_objective_is_the_negative_log_projected_ball_likelihood():
    """Where every point is well inside the support the barrier is exact."""
    center, U, data = _make_problem(0, 0.05)
    Z, y, _, weights, delta2 = data
    b0 = np.full(N, 1e3)  # wide ellipsoid: q_i close to 1
    data = (Z, y, b0, weights, delta2)
    half = np.sqrt(b0 + np.sum((Z @ U) ** 2, axis=1) + delta2)
    q = 1.0 - ((y - Z @ center) / half) ** 2
    assert q.min() > 1e-3
    expected = -weights @ projected_ball_logpdf(y - Z @ center, half, P + 1)
    value = _nll(center, U, *data, 1e-3, *CONSTANTS)
    np.testing.assert_allclose(value, expected, rtol=1e-12)


@pytest.mark.parametrize("rho,scale", BRANCHES)
def test_gradient_exactness(rho, scale):
    center, U, data = _make_problem(3, scale)
    psi = np.concatenate([center, U.ravel()])
    loss = _packed(data, rho)
    np.testing.assert_allclose(
        jax.grad(loss)(psi), _central_difference(loss, psi), rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("rho,scale", BRANCHES)
def test_hessian_diagonal_exactness(rho, scale):
    """Chained diagonal Hessian vs finite differences of the gradient."""
    center, U, data = _make_problem(11, scale)
    psi = np.concatenate([center, U.ravel()])
    hd_center, hd_U = _hess_diag(center, U, *data, rho, *CONSTANTS)
    hess_diag = np.concatenate([hd_center, np.ravel(hd_U)])
    fd = np.diag(_central_difference(jax.grad(_packed(data, rho)), psi))
    np.testing.assert_allclose(hess_diag, fd, rtol=1e-5, atol=1e-6)


def test_zero_weights_drop_points():
    """Zero-weighted points must not contribute to value or gradient."""
    center, U, (Z, y, b0, _, delta2) = _make_problem(5, 1.0)
    psi = np.concatenate([center, U.ravel()])
    weights = np.ones(N)
    weights[N // 2 :] = 0.0
    half = slice(0, N // 2)
    full = _packed((Z, y, b0, weights, delta2), 1e-2)
    subset = _packed((Z[half], y[half], b0[half], np.ones(N // 2), delta2), 1e-2)
    np.testing.assert_allclose(full(psi), subset(psi), rtol=1e-12)
    np.testing.assert_allclose(
        jax.grad(full)(psi), jax.grad(subset)(psi), rtol=1e-12, atol=1e-14
    )


# --- L-BFGS ---


def test_lbfgs_minimizes_rosenbrock():
    def rosenbrock(x):
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    x, value, n_iter = lbfgs(
        jax.value_and_grad(rosenbrock), jnp.zeros(6), max_iter=500, tol=1e-12
    )
    np.testing.assert_allclose(x, np.ones(6), atol=1e-5)
    assert value < 1e-10 and 0 < n_iter < 500


def test_lbfgs_respects_max_iter_and_starts_converged():
    quadratic = jax.value_and_grad(lambda x: 0.5 * jnp.sum(x * x))
    _, _, n_iter = lbfgs(quadratic, jnp.zeros(3), max_iter=10, tol=1e-8)
    assert n_iter == 0
    ill = jax.value_and_grad(lambda x: jnp.sum(jnp.arange(1.0, 51.0) ** 3 * x**4))
    _, _, n_iter = lbfgs(ill, jnp.ones(50), max_iter=3, tol=1e-30)
    assert n_iter == 3


def test_lbfgs_batches_with_vmap():
    targets = jnp.asarray(np.random.RandomState(0).randn(5, 4))
    scales = jnp.asarray([1.0, 3.0, 10.0, 30.0])

    def solve(target):
        fun = jax.value_and_grad(lambda x: jnp.sum(scales * (x - target) ** 2))
        return lbfgs(fun, jnp.zeros(4), max_iter=200, tol=1e-14)[0]

    np.testing.assert_allclose(jax.vmap(solve)(targets), targets, atol=1e-6)


def _continuation_problem():
    rng = np.random.RandomState(2)
    Z = rng.randn(200, P)
    y = Z @ rng.randn(P) + 0.3 * np.sin(2 * Z[:, 0])
    center0 = np.linalg.lstsq(Z, y, rcond=None)[0]
    U0 = 1e-3 * rng.randn(P, R)
    data = (Z, y, np.zeros(200), np.ones(200), DELTA2)
    return center0, U0, data, np.array([1e-1, 1e-2, 1e-3])


def test_continuation_matches_scipy():
    """The compiled continuation reaches the optimum SciPy's L-BFGS-B finds."""
    center0, U0, data, rhos = _continuation_problem()
    center, U, value, n_iter = _fit_ellipsoid(
        center0, U0, *data, rhos, *CONSTANTS, 1e-10, 2000, False
    )
    assert n_iter.shape == (3,) and int(n_iter.sum()) > 0
    np.testing.assert_array_equal(center, center0)

    psi = np.concatenate([center0, U0.ravel()])
    for rho in rhos:

        def objective(x):
            full = np.concatenate([center0, x])
            value, grad = jax.value_and_grad(_packed(data, rho))(full)
            return float(value), np.asarray(grad)[P:]

        psi[P:] = minimize(objective, psi[P:], jac=True, method="L-BFGS-B", tol=1e-10).x
    reference = float(_packed(data, rhos[-1])(psi))
    assert float(value) <= reference + 1e-5 * abs(reference)


def test_continuation_with_free_center_reaches_a_stationary_point():
    """The joint problem is non-convex: check stationarity, not one optimum."""
    center0, U0, data, rhos = _continuation_problem()
    center, U, value, _ = _fit_ellipsoid(
        center0, U0, *data, rhos, *CONSTANTS, 1e-12, 5000, True
    )
    loss = _packed(data, rhos[-1])
    start = np.concatenate([center0, U0.ravel()])
    end = np.concatenate([center, np.ravel(U)])
    assert not np.allclose(center, center0)
    np.testing.assert_allclose(value, loss(end), rtol=1e-12)
    assert float(value) < float(loss(start))
    gradient = lambda psi: np.abs(np.asarray(jax.grad(loss)(psi))).max()
    assert gradient(end) < 1e-6 * gradient(start)
