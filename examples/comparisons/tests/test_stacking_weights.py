"""Tests for the held-out predictive-stacking solver."""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from comparisons.stacking_weights import (
    mixture_log_loss,
    solve_stacking_weights,
)
from numpy.testing import assert_allclose
from scipy.optimize import check_grad


def _random_log_density(n_rows, n_candidates, seed):
    rng = np.random.RandomState(seed)
    return np.log(rng.uniform(0.02, 1.0, size=(n_rows, n_candidates)))


def test_objective_matches_direct_evaluation_and_gradient():
    L = _random_log_density(40, 4, seed=0)
    q = np.array([0.1, 0.2, 0.3, 0.4])
    value, grad = mixture_log_loss(q, L)
    direct = -np.mean(np.log(np.exp(L) @ q))
    assert value == pytest.approx(direct, rel=1e-12)
    err = check_grad(
        lambda v: mixture_log_loss(v, L)[0], lambda v: mixture_log_loss(v, L)[1], q
    )
    assert err < 1e-6

    # Weighted rows (equal group weights) are honoured in value and gradient.
    w = np.linspace(1.0, 2.0, 40)
    w /= w.sum()
    value_w, grad_w = mixture_log_loss(q, L, w)
    assert value_w == pytest.approx(-float(w @ np.log(np.exp(L) @ q)), rel=1e-12)
    err = check_grad(
        lambda v: mixture_log_loss(v, L, w)[0],
        lambda v: mixture_log_loss(v, L, w)[1],
        q,
    )
    assert err < 1e-6


def test_zero_weights_are_stable():
    """Boundary weights are allowed: no NaN from log(0)."""
    L = _random_log_density(20, 3, seed=1)
    value, grad = mixture_log_loss(np.array([1.0, 0.0, 0.0]), L)
    assert np.isfinite(value) and np.all(np.isfinite(grad))
    assert value == pytest.approx(-np.mean(L[:, 0]))


def test_simplex_constraints_and_status():
    L = _random_log_density(200, 6, seed=2)
    sol = solve_stacking_weights(L)
    assert sol.weights.shape == (6,)
    assert np.all(sol.weights >= 0.0)
    assert sol.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert sol.status["success"]
    assert sol.status["kkt_residual"] <= 1e-6
    assert sol.status["simplex_residual"] <= 1e-8
    assert sol.status["method"] in ("slsqp", "em_fallback")


def test_agrees_with_dense_grid_for_two_candidates():
    L = _random_log_density(80, 2, seed=3)
    sol = solve_stacking_weights(L)
    grid = np.linspace(0.0, 1.0, 200001)
    values = np.array([mixture_log_loss(np.array([g, 1.0 - g]), L)[0] for g in grid])
    best = grid[np.argmin(values)]
    assert sol.weights[0] == pytest.approx(best, abs=2e-5)
    assert sol.objective <= values.min() + 1e-10


def test_mixing_improves_on_every_single_candidate():
    """Candidate 0 fits the first half of the rows, candidate 1 the rest."""
    n = 60
    L = np.full((n, 2), np.log(0.05))
    L[: n // 2, 0] = np.log(0.9)
    L[n // 2 :, 1] = np.log(0.9)
    sol = solve_stacking_weights(L)
    single = [mixture_log_loss(np.eye(2)[k], L)[0] for k in range(2)]
    assert sol.objective < min(single) - 0.1
    assert_allclose(sol.weights, [0.5, 0.5], atol=1e-6)


def test_duplicated_candidate_keeps_the_optimum_value():
    L = _random_log_density(120, 4, seed=4)
    base = solve_stacking_weights(L)
    dup = solve_stacking_weights(np.column_stack([L, L[:, :1]]))
    assert dup.objective == pytest.approx(base.objective, abs=1e-9)
    # The attainable mixtures are the same: merged weights agree.
    merged = dup.weights[:4].copy()
    merged[0] += dup.weights[4]
    assert_allclose(merged, base.weights, atol=1e-5)


def test_invalid_inputs():
    with pytest.raises(ValueError, match="finite entry"):
        solve_stacking_weights(np.full((3, 2), -np.inf))
    with pytest.raises(ValueError, match="row_weights"):
        mixture_log_loss([0.5, 0.5], np.zeros((3, 2)), np.array([1.0, 1.0, 1.0]))
