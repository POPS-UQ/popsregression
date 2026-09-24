"""Tests for the standalone Bayesian subset-regression stacking baseline."""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import inspect

import comparisons.bayesian_stacking as module
import numpy as np
import pytest
from comparisons.bayesian_stacking import (
    BayesianLinearRegression,
    BayesianStacking,
)
from numpy.testing import assert_allclose
from scipy.integrate import quad


def _poly(x, degree):
    return np.column_stack([x**d for d in range(degree + 1)])


def _data(n, seed, degree=4):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, n)
    return _poly(x, degree), np.sin(3 * x) + 0.3 * x


COMPONENTS = [np.arange(d + 1) for d in range(1, 5)]


def test_no_pops_dependencies():
    source = inspect.getsource(module)
    for forbidden in ("_pops", "_ellipse", "_pac_certificate", "_projected_ball"):
        assert forbidden not in source
    assert "_predictive_stacking" in source  # only the generic weight solver


def test_direct_small_matrix_calculation():
    X = np.array([[1.0, 0.5], [1.0, -1.0], [1.0, 2.0]])
    y = np.array([0.7, -0.4, 1.9])
    model = BayesianLinearRegression(
        prior_precision=0.1, a0=2.0, b0=0.5, standardize=False
    ).fit(X, y)
    Lambda0 = 0.1 * np.eye(2)
    Lambda_n = Lambda0 + X.T @ X
    m_n = np.linalg.solve(Lambda_n, X.T @ y)
    a_n = 2.0 + 1.5
    b_n = 0.5 + 0.5 * (y @ y - m_n @ Lambda_n @ m_n)
    assert_allclose(model.m_n_, m_n)
    assert model.a_n_ == pytest.approx(a_n)
    assert model.b_n_ == pytest.approx(b_n)
    x = np.array([[1.0, 0.3]])
    loc, scale, scale_param = model._predictive(x)
    assert loc[0] == pytest.approx(x[0] @ m_n)
    quad_form = x[0] @ np.linalg.solve(Lambda_n, x[0])
    assert scale[0] ** 2 == pytest.approx((b_n / a_n) * (1 + quad_form))
    assert scale_param[0] ** 2 == pytest.approx((b_n / a_n) * quad_form)
    assert model.df_ == pytest.approx(2 * a_n)
    assert_allclose(model.coef_, m_n)
    assert model.intercept_ == 0.0


def test_residual_update_is_stable_when_ill_conditioned():
    rng = np.random.RandomState(0)
    x = rng.uniform(0, 1, 40)
    X = np.column_stack([np.ones(40), x, x + 1e-9 * rng.randn(40), x**2])
    y = 2 + x - 0.5 * x**2
    model = BayesianLinearRegression(prior_precision=1e-8).fit(X, y)
    assert model.b_n_ > model.b0 > 0
    assert np.isfinite(model.residual_scale_) and model.residual_scale_ > 0
    assert np.all(np.isfinite(model.predict_logpdf(X, y)))


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("include_residual", [False, True])
def test_predictive_normalization_and_moments(standardize, include_residual):
    X, y = _data(30, seed=1)
    model = BayesianLinearRegression(standardize=standardize).fit(X, y)
    x = X[:1]
    mean, std = model.predict(x, return_std=True, include_residual=include_residual)
    loc, scale_full, scale_param = model._predictive(x)
    scale = scale_full if include_residual else scale_param
    df = model.df_
    total = quad(
        lambda t: np.exp(model.predict_logpdf(x, np.array([t]), include_residual)[0]),
        loc[0] - 60 * scale[0],
        loc[0] + 60 * scale[0],
        limit=500,
    )[0]
    assert total == pytest.approx(1.0, abs=1e-6)
    first = quad(
        lambda t: t
        * np.exp(model.predict_logpdf(x, np.array([t]), include_residual)[0]),
        loc[0] - 60 * scale[0],
        loc[0] + 60 * scale[0],
        limit=500,
    )[0]
    assert first == pytest.approx(mean[0], abs=1e-6)
    assert std[0] ** 2 == pytest.approx(scale[0] ** 2 * df / (df - 2))
    assert scale_param[0] < scale_full[0]
    assert model.predict_cdf(x, loc, include_residual)[0] == pytest.approx(0.5)
    # The default is parameter-only: no residual term enters.
    assert_allclose(model.predict_logpdf(x, loc), model.predict_logpdf(x, loc, False))


def test_standardization_jacobian_and_scaling_equivariance():
    X, y = _data(30, seed=2)
    a = BayesianLinearRegression(standardize=True).fit(X, y)
    b = BayesianLinearRegression(standardize=True).fit(X, 1000.0 * y)
    # The same model in rescaled units: log densities shift by log(1000).
    assert_allclose(
        b.predict_logpdf(X, 1000.0 * y), a.predict_logpdf(X, y) - np.log(1000.0)
    )
    assert b.residual_scale_ == pytest.approx(1000.0 * a.residual_scale_)


def test_leave_one_out_is_fold_isolated():
    X, y = _data(25, seed=3)
    stack = BayesianStacking(COMPONENTS, cv="loo").fit(X, y)
    assert stack.oof_log_density_.shape == (25, 4)
    i, k = 7, 2
    keep = np.arange(25) != i
    manual = BayesianLinearRegression(COMPONENTS[k]).fit(X[keep], y[keep])
    assert stack.oof_log_density_[i, k] == pytest.approx(
        manual.predict_logpdf(X[i : i + 1], y[i : i + 1])[0], rel=1e-12
    )
    full = BayesianStacking(COMPONENTS, cv="loo", include_residual=True).fit(X, y)
    assert full.oof_log_density_[i, k] == pytest.approx(
        manual.predict_logpdf(X[i : i + 1], y[i : i + 1], True)[0], rel=1e-12
    )
    assert full.oof_log_density_[i, k] != pytest.approx(stack.oof_log_density_[i, k])
    # Preprocessing is fold-internal: the held-out row did not enter the
    # standardization of the component that scored it.
    assert manual.y_mean_ == pytest.approx(y[keep].mean())
    assert manual.y_mean_ != pytest.approx(y.mean())


def test_grouped_folds_keep_cases_indivisible():
    X, y = _data(30, seed=4)
    groups = np.repeat(np.arange(10), 3)
    stack = BayesianStacking(COMPONENTS, cv=5).fit(X, y, group_ids=groups)
    for train, test in stack._folds(30, groups):
        assert not np.intersect1d(groups[train], groups[test]).size
    assert_allclose(stack.row_weights_, 1.0 / 30)
    assert stack.weights_.sum() == pytest.approx(1.0)


def test_stacking_weights_and_controls():
    X, y = _data(60, seed=5)
    stack = BayesianStacking(COMPONENTS).fit(X, y)
    q = stack.weights_
    assert np.all(q >= 0) and q.sum() == pytest.approx(1.0)
    assert stack.solver_status_["success"]
    assert set(stack.weights_by_rule_) == {"stacking", "best", "uniform"}
    scores = stack.mixture_cv_scores_
    assert scores["stacking"] <= scores["best"] + 1e-9
    assert scores["stacking"] <= scores["uniform"] + 1e-9
    assert scores["best"] == pytest.approx(stack.cv_scores_.min())
    assert stack.residual_scales_.shape == (4,) and np.all(stack.residual_scales_ > 0)
    assert len(stack.components_) == 4
    assert stack.fit_time_ > 0


def test_mixture_predictive_is_a_density_with_exact_quantiles():
    X, y = _data(60, seed=6)
    stack = BayesianStacking(COMPONENTS).fit(X, y)
    x = X[:3]
    for i in range(3):
        xi = x[i : i + 1]
        mean, std = stack.predict(xi, return_std=True)
        total = quad(
            lambda t: np.exp(stack.predict_logpdf(xi, np.array([t]))[0]),
            mean[0] - 60 * std[0],
            mean[0] + 60 * std[0],
            limit=500,
        )[0]
        assert total == pytest.approx(1.0, abs=1e-5)
    level = 0.9545
    lo, hi = stack.predict_interval(x, level=level)
    assert_allclose(stack.predict_cdf(x, lo), 0.5 * (1 - level), atol=1e-8)
    assert_allclose(stack.predict_cdf(x, hi), 0.5 * (1 + level), atol=1e-8)
    # Parameter-only (the default) is smaller than the full predictive.
    _, std_param = stack.predict(x, return_std=True)
    _, std_full = stack.predict(x, return_std=True, include_residual=True)
    assert np.all(std_param < std_full)
    lo_f, hi_f = stack.predict_interval(x, level=level, include_residual=True)
    assert np.all(lo_f < lo) and np.all(hi_f > hi)
    # Uniform weights are accepted by name.
    assert np.isfinite(stack.predict_logpdf(x, y[:3], weights="uniform")).all()


def test_parameter_draws_embed_into_the_reference_space():
    X, y = _data(60, seed=7)
    stack = BayesianStacking(COMPONENTS).fit(X, y)
    coef, intercept, component, sigma2 = stack.sample_parameters(4000, random_state=0)
    assert coef.shape == (5, 4000) and np.all(sigma2 > 0)
    # A degree-d component leaves higher columns at zero.
    for k in range(4):
        sel = component == k
        assert np.all(coef[k + 2 :, sel] == 0.0)
    # One draw is one field: propagated location variance matches the
    # parameter-only predictive variance.
    X_new = _poly(np.linspace(-1, 1, 21), 4)
    field = X_new @ coef + intercept[None, :]
    _, std_param = stack.predict(X_new, return_std=True)
    assert_allclose(field.var(axis=1), std_param**2, rtol=0.15)
    freq = np.bincount(component, minlength=4) / 4000
    assert_allclose(freq, stack.weights_, atol=0.03)


def test_invalid_inputs():
    X, y = _data(10, seed=8)
    with pytest.raises(ValueError, match="outside X"):
        BayesianStacking([np.arange(9)]).fit(X, y)
    with pytest.raises(ValueError, match="positive"):
        BayesianLinearRegression(a0=-1.0).fit(X, y)
    stack = BayesianStacking(COMPONENTS).fit(X, y)
    with pytest.raises(ValueError, match="probability vector"):
        stack.predict(X, weights=np.array([1.0, 1.0, 0.0, 0.0]))
