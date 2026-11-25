"""Tests of the port of the published PVI code."""

import numpy as np
import pytest
from comparisons.pvi import PredictiveVI
from numpy.testing import assert_allclose
from scipy.special import logsumexp
from scipy.stats import norm


def _data(n=30, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, n)
    return np.vander(x, 4, increasing=True), np.sin(3 * x) + 0.3


def _fd(fun, params, eps=1e-6):
    return np.array(
        [
            (fun(params + eps * e) - fun(params - eps * e)) / (2 * eps)
            for e in np.eye(params.size)
        ]
    )


@pytest.mark.parametrize("posterior", ["Basic", "BasicFullRank"])
def test_objective_gradients_match_finite_differences(posterior):
    X, y = _data()
    model = PredictiveVI(sigma=0.3, posterior=posterior)
    Z, ys = model._standardize_fit(X, y)
    p = Z.shape[1]
    rng = np.random.RandomState(1)
    params = 0.2 * rng.randn(model._n_params(p))
    u = rng.randn(5, p)
    _, g = model._log_score(params, Z, ys, u)
    assert_allclose(
        g, _fd(lambda q: model._log_score(q, Z, ys, u)[0], params), rtol=1e-5, atol=1e-6
    )
    _, g = model._kl_prior(params, u)
    assert_allclose(
        g, _fd(lambda q: model._kl_prior(q, u)[0], params), rtol=1e-5, atol=1e-7
    )


def test_log_score_is_the_published_log_mean_exp():
    X, y = _data()
    model = PredictiveVI(sigma=0.2)
    Z, ys = model._standardize_fit(X, y)
    p = Z.shape[1]
    params = np.random.RandomState(2).randn(2 * p) * 0.1
    u = np.random.RandomState(3).randn(7, p)
    theta = params[:p] + np.exp(params[p:]) * u
    lls = norm.logpdf(ys[:, None], Z @ theta.T, model._width())
    expected = np.sum(logsumexp(lls, axis=1) - np.log(7))
    assert_allclose(model._log_score(params, Z, ys, u)[0], expected)


def test_no_implicit_intercept_and_sigma_in_target_std_units():
    X, y = _data()
    model = PredictiveVI(sigma=0.1)
    Z, ys = model._standardize_fit(X, y)
    assert np.all(model.x_mean_ == 0.0) and model.y_mean_ == 0.0
    assert_allclose(model._width() * model.y_scale_, 0.1 * y.std())


@pytest.mark.parametrize("optimizer", ["sgd", "nesterov", "rmsprop"])
def test_training_increases_the_objective(optimizer):
    X, y = _data(60)
    model = PredictiveVI(
        sigma=0.1,
        s=8,
        lamb=0.01,
        iterations=3000,
        learning_rate=1e-2,
        optimizer=optimizer,
    ).fit(X, y)
    v = np.asarray(model.values_)
    assert np.mean(v[-100:]) > np.mean(v[:100])
    assert model.converged_


def test_predictions_are_parameter_only():
    X, y = _data()
    model = PredictiveVI(sigma=0.5, s=4, iterations=500).fit(X, y)
    _, std = model.predict(X, return_std=True)
    ZL = model._transform(X) @ model.L_
    assert_allclose(std, model.y_scale_ * np.sqrt(np.sum(ZL**2, axis=1)))
    lo, hi = model.predict_interval(X, level=0.9)
    assert_allclose(model.predict_cdf(X, hi) - model.predict_cdf(X, lo), 0.9)
    coef, offset = model.sample_parameters(4000, random_state=0)
    draws = X @ coef + offset
    assert_allclose(draws.mean(axis=1), model.predict(X), atol=0.05 * std.max() + 1e-6)


@pytest.mark.parametrize("lamb", [0.0, 1.0])
def test_fresh_objective_matches_training_history(lamb):
    """The fresh-draw objective at the final q agrees with the average of
    the last training steps (which used their own fresh draws), and the
    recorded parts add up to the recorded step values."""
    X, y = _data(40)
    model = PredictiveVI(
        sigma=0.3,
        s=4,
        lamb=lamb,
        iterations=4000,
        learning_rate=1e-2,
        optimizer="rmsprop",
    ).fit(X, y)
    v, sc, kl = map(np.asarray, (model.values_, model.score_values_, model.kl_values_))
    assert_allclose(v, sc + lamb * kl)
    assert model.n_fev_ == model.n_iter_ == 4000
    assert model.termination_ == "iteration_budget"
    data, kl_term, total = model.evaluate_objective(X, y, n_groups=4000, random_state=1)
    n = y.size
    assert total == pytest.approx(data + kl_term)
    assert data == pytest.approx(-np.mean(sc[-500:]) / n, rel=0.05, abs=0.05)
    if lamb:
        # The published KLPrior estimates -KL(q || prior) without bias.
        assert -np.mean(kl[-500:]) == pytest.approx(model.kl_divergence(), rel=0.1)
    else:
        assert kl_term == 0.0
