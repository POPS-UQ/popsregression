"""Tests for POPSRegression."""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.estimator_checks import parametrize_with_checks

from popsregression import POPSRegression


def _make_low_noise_data(n_samples=50, n_features=5, noise=0.01, seed=42):
    """Generate low-noise polynomial regression data."""
    rng = np.random.RandomState(seed)
    x = np.sort(rng.uniform(-1, 1, n_samples)) * 10
    f = lambda x: (x**3 + 0.01 * x**4) * 0.1 + np.sin(x) * x * 10.0

    poly = PolynomialFeatures(degree=n_features - 1, include_bias=True)
    X = poly.fit_transform(x.reshape(-1, 1))
    y = f(x) + noise * rng.randn(n_samples)
    return X, y, poly


# --- Basic functionality ---


def test_fit_returns_self():
    X, y, _ = _make_low_noise_data()
    model = POPSRegression()
    result = model.fit(X, y)
    assert result is model


def test_fitted_attributes():
    X, y, _ = _make_low_noise_data()
    model = POPSRegression().fit(X, y)

    assert hasattr(model, "coef_")
    assert hasattr(model, "sigma_")
    assert hasattr(model, "misspecification_sigma_")
    assert hasattr(model, "posterior_samples_")
    assert hasattr(model, "alpha_")
    assert hasattr(model, "lambda_")
    assert hasattr(model, "n_iter_")

    n_features = X.shape[1]
    assert model.coef_.shape == (n_features,)
    assert model.sigma_.shape == (n_features, n_features)
    assert model.misspecification_sigma_.shape == (n_features, n_features)
    assert model.posterior_samples_.shape[0] == n_features


def test_predict_mean_only():
    X, y, _ = _make_low_noise_data()
    model = POPSRegression().fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_predict_return_std():
    X, y, _ = _make_low_noise_data()
    model = POPSRegression().fit(X, y)
    y_pred, y_std = model.predict(X, return_std=True)
    assert y_pred.shape == (X.shape[0],)
    assert y_std.shape == (X.shape[0],)
    assert np.all(y_std >= 0)


def test_predict_return_bounds():
    X, y, _ = _make_low_noise_data()
    model = POPSRegression().fit(X, y)
    y_pred, y_max, y_min = model.predict(X, return_bounds=True)
    assert y_pred.shape == (X.shape[0],)
    assert_array_less(y_min, y_max + 1e-10)


def test_predict_return_all():
    X, y, _ = _make_low_noise_data()
    model = POPSRegression().fit(X, y)
    result = model.predict(
        X, return_std=True, return_bounds=True, return_epistemic_std=True
    )
    assert len(result) == 5
    y_pred, y_std, y_max, y_min, y_epi_std = result
    assert y_pred.shape == (X.shape[0],)
    assert y_std.shape == (X.shape[0],)
    assert y_epi_std.shape == (X.shape[0],)


def test_predict_return_epistemic_std():
    X, y, _ = _make_low_noise_data()
    model = POPSRegression().fit(X, y)
    y_pred, y_epi_std = model.predict(X, return_epistemic_std=True)
    assert np.all(y_epi_std >= 0)


# --- Posterior types ---


@pytest.mark.parametrize("posterior", ["hypercube", "ensemble"])
def test_posterior_types(posterior):
    X, y, _ = _make_low_noise_data()
    model = POPSRegression(posterior=posterior).fit(X, y)
    y_pred, y_std = model.predict(X, return_std=True)
    assert y_pred.shape == (X.shape[0],)
    assert np.all(y_std >= 0)


# --- Resampling methods ---


@pytest.mark.parametrize("method", ["uniform", "sobol", "latin", "halton"])
def test_resampling_methods(method):
    X, y, _ = _make_low_noise_data()
    model = POPSRegression(resampling_method=method).fit(X, y)
    y_pred, y_std = model.predict(X, return_std=True)
    assert y_pred.shape == (X.shape[0],)
    assert np.all(y_std >= 0)


# --- POPS vs BayesianRidge uncertainty ---


def test_misspecification_sigma_larger_than_epistemic():
    """POPS misspecification uncertainty should generally be larger than
    epistemic-only uncertainty for low-noise misspecified models."""
    X, y, _ = _make_low_noise_data(n_samples=30, n_features=5, noise=0.001)
    model = POPSRegression().fit(X, y)

    misspec_trace = np.trace(model.misspecification_sigma_)
    epistemic_trace = np.trace(model.sigma_)
    assert misspec_trace > epistemic_trace


def test_pops_wider_uncertainty_than_bayesian_ridge():
    """POPS combined uncertainty should be wider than BayesianRidge
    epistemic-only uncertainty for misspecified low-noise data."""
    X, y, _ = _make_low_noise_data(n_samples=30, n_features=5, noise=0.001)

    pops = POPSRegression().fit(X, y)
    br = BayesianRidge(fit_intercept=False).fit(X, y)

    _, pops_std = pops.predict(X, return_std=True)
    br_epistemic_std = np.sqrt(np.sum(np.dot(X, br.sigma_) * X, axis=1))

    assert np.mean(pops_std) > np.mean(br_epistemic_std)


# --- fit_intercept ---


def test_fit_intercept():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 3)
    y = X @ np.array([1, 2, 3]) + 5.0 + 0.01 * rng.randn(50)

    model = POPSRegression(fit_intercept=True).fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == (50,)
    assert np.mean((y - y_pred) ** 2) < 1.0


def test_fit_intercept_get_params_consistency():
    """fit_intercept should be correctly reported after fit."""
    model = POPSRegression(fit_intercept=True)
    assert model.get_params()["fit_intercept"] is True

    X, y, _ = _make_low_noise_data()
    model.fit(X, y)
    assert model.get_params()["fit_intercept"] is True


# --- Parameter validation ---


def test_invalid_posterior():
    with pytest.raises(ValueError):
        POPSRegression(posterior="invalid").fit(*_make_low_noise_data()[:2])


def test_invalid_resampling_method():
    with pytest.raises(ValueError):
        POPSRegression(resampling_method="invalid").fit(*_make_low_noise_data()[:2])


# --- sample_weight ---


def test_sample_weight():
    X, y, _ = _make_low_noise_data()
    weights = np.ones(X.shape[0])
    weights[:10] = 2.0

    model = POPSRegression().fit(X, y, sample_weight=weights)
    y_pred = model.predict(X)
    assert y_pred.shape == (X.shape[0],)


# --- compute_score ---


def test_compute_score():
    X, y, _ = _make_low_noise_data()
    model = POPSRegression(compute_score=True).fit(X, y)
    assert hasattr(model, "scores_")
    assert len(model.scores_) > 0


# --- Minimum relative error ---


@pytest.mark.parametrize("minimum_relative_error", [0.0, 0.01, 0.5])
def test_minimum_relative_error(minimum_relative_error):
    X, y, _ = _make_low_noise_data()
    model = POPSRegression(minimum_relative_error=minimum_relative_error).fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_minimum_relative_error_filters_points():
    """Points with residuals below the threshold are excluded."""
    X, y, _ = _make_low_noise_data()

    all_points = POPSRegression(minimum_relative_error=0.0).fit(X, y)
    assert all_points._filtering_mask.all()

    filtered = POPSRegression(minimum_relative_error=0.5).fit(X, y)
    assert filtered._filtering_mask.sum() < X.shape[0]


def test_minimum_relative_error_is_relative_to_rmse():
    """The threshold is minimum_relative_error * RMSE of the mean fit."""
    X, y, _ = _make_low_noise_data()
    model = POPSRegression(minimum_relative_error=0.5).fit(X, y)

    errors = y - X @ model.coef_
    rmse = np.sqrt(np.mean(errors**2))
    expected = np.abs(errors) >= 0.5 * rmse
    assert np.array_equal(model._filtering_mask, expected)


def test_minimum_relative_error_is_scale_invariant():
    """Rescaling y leaves the selected points unchanged."""
    X, y, _ = _make_low_noise_data()
    model = POPSRegression(minimum_relative_error=0.5).fit(X, y)
    scaled = POPSRegression(minimum_relative_error=0.5).fit(X, 1000.0 * y)
    assert np.array_equal(model._filtering_mask, scaled._filtering_mask)


def test_minimum_relative_error_falls_back_when_all_filtered():
    """If no point passes the threshold, all points are used."""
    X, y, _ = _make_low_noise_data()
    model = POPSRegression(minimum_relative_error=1e12).fit(X, y)
    assert model._filtering_mask.all()


# --- Deprecation of leverage_percentile ---


def test_leverage_percentile_deprecated():
    X, y, _ = _make_low_noise_data()
    model = POPSRegression(leverage_percentile=50.0)
    with pytest.warns(FutureWarning, match="leverage_percentile"):
        model.fit(X, y)


def test_leverage_percentile_is_ignored():
    """The deprecated parameter no longer affects the fit."""
    X, y, _ = _make_low_noise_data()

    reference = POPSRegression(posterior="ensemble").fit(X, y)
    with pytest.warns(FutureWarning):
        deprecated = POPSRegression(posterior="ensemble", leverage_percentile=90.0).fit(
            X, y
        )

    assert_allclose(
        deprecated.misspecification_sigma_, reference.misspecification_sigma_
    )


def test_no_warning_without_leverage_percentile():
    X, y, _ = _make_low_noise_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        POPSRegression().fit(X, y)


# --- Cloning and get_params/set_params ---


def test_clone():
    from sklearn.base import clone

    model = POPSRegression(
        posterior="ensemble",
        resample_density=2.0,
        minimum_relative_error=0.05,
    )
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()


def test_get_set_params():
    model = POPSRegression()
    params = model.get_params()
    assert "posterior" in params
    assert "resample_density" in params
    assert "minimum_relative_error" in params

    model.set_params(posterior="ensemble", resample_density=5.0)
    assert model.posterior == "ensemble"
    assert model.resample_density == 5.0


# --- sklearn estimator checks ---


@parametrize_with_checks([POPSRegression()])
def test_sklearn_compatible(estimator, check):
    check(estimator)
