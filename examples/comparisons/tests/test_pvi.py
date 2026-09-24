"""Tests of the predictive variational inference comparator."""

import numpy as np
import pytest
from comparisons.pvi import PredictiveVI
from numpy.testing import assert_allclose
from scipy.stats import norm


def _data(n=30, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, n)
    X = np.vander(x, 4, increasing=True)
    return X, np.sin(3 * x)


@pytest.mark.parametrize("lam", [0.0, 0.7])
def test_gradient_matches_finite_differences(lam):
    X, y = _data()
    model = PredictiveVI(sigma=0.05, lam=lam)
    Z, ys = model._standardize_fit(X, y)
    rng = np.random.RandomState(1)
    p = Z.shape[1]
    params = rng.randn(p + p * (p + 1) // 2) * 0.3
    _, grad = model.objective(params, Z, ys, lam)
    eps = 1e-6
    fd = np.array(
        [
            (
                model.objective(params + eps * e, Z, ys, lam)[0]
                - model.objective(params - eps * e, Z, ys, lam)[0]
            )
            / (2 * eps)
            for e in np.eye(params.size)
        ]
    )
    assert_allclose(grad, fd, rtol=1e-5, atol=1e-8)


def test_kl_term_matches_direct_formula():
    X, y = _data()
    model = PredictiveVI(sigma=0.1, prior_scale=2.0)
    Z, ys = model._standardize_fit(X, y)
    p = Z.shape[1]
    params = np.random.RandomState(2).randn(p + p * (p + 1) // 2) * 0.2
    v1, _ = model.objective(params, Z, ys, 1.0)
    v0, _ = model.objective(params, Z, ys, 0.0)
    mu, L = model._unpack(params, p)
    S = L @ L.T
    tau2 = 4.0
    kl = 0.5 * (
        np.trace(S) / tau2
        + mu @ mu / tau2
        - p
        + p * np.log(tau2)
        - np.linalg.slogdet(S)[1]
    )
    assert_allclose((v1 - v0) * Z.shape[0], kl, rtol=1e-10)


def test_exact_predictive_equals_monte_carlo_average():
    X, y = _data()
    model = PredictiveVI(sigma=0.1, lam=1.0).fit(X, y)
    Z, ys = model._transform(X[:3]), (y[:3] - model.y_mean_) / model.y_scale_
    rng = np.random.RandomState(3)
    theta = model.mu_[:, None] + model.L_ @ rng.randn(model.mu_.size, 200000)
    mc = norm.pdf(ys[:, None], Z @ theta, model.sigma).mean(axis=1)
    exact = np.exp(model._log_score(model.mu_, model.L_, Z, ys))
    assert_allclose(mc, exact, rtol=2e-2)


def test_predictions_are_parameter_only():
    X, y = _data()
    model = PredictiveVI(sigma=0.5, lam=1.0).fit(X, y)
    _, std = model.predict(X, return_std=True)
    ZL = model._transform(X) @ model.L_
    assert_allclose(std, model.y_scale_ * np.sqrt(np.sum(ZL**2, axis=1)))
    lo, hi = model.predict_interval(X, level=0.9)
    assert_allclose(model.predict_cdf(X, hi) - model.predict_cdf(X, lo), 0.9)
    coef, offset = model.sample_parameters(4000, random_state=0)
    draws = X @ coef + offset
    assert_allclose(draws.mean(axis=1), model.predict(X), atol=0.05 * std.max() + 1e-6)


def test_cross_validation_selects_from_grid():
    X, y = _data(40)
    model = PredictiveVI(sigma=0.05, lam="cv", lam_grid=(0.0, 1.0)).fit(X, y)
    assert model.lam_ in (0.0, 1.0)
    assert set(model.cv_scores_) == {0.0, 1.0}
