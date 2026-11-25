"""Tests for POPSRegression."""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from popsregression import POPSRegression
from popsregression._base import NotFittedError


def _make_low_noise_data(n_samples=50, n_features=5, noise=0.01, seed=42):
    """Generate low-noise polynomial regression data."""
    rng = np.random.RandomState(seed)
    x = np.sort(rng.uniform(-1, 1, n_samples)) * 10
    f = lambda x: (x**3 + 0.01 * x**4) * 0.1 + np.sin(x) * x * 10.0
    X = np.vander(x, n_features, increasing=True)
    y = f(x) + noise * rng.randn(n_samples)
    return X, y


def _make_low_leverage_data(scale, n_samples=2000, n_features=8, n_low=5, seed=0):
    """Misspecified data in which a few rows barely excite the model."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    X[:n_low] *= scale
    y = X @ rng.randn(n_features) + 0.05 * np.sin(3 * X[:, 0])
    y += 0.02 * np.cos(5 * X[:, 1])
    return X, y, rng.randn(500, n_features)


def _gram(model, X, weights=None):
    """Regularized Gram matrix of the mean fit."""
    weights = np.ones(X.shape[0]) if weights is None else weights
    ratio = model.lambda_ / model.alpha_
    return (X * weights[:, None]).T @ X + ratio * np.eye(X.shape[1])


# --- Basic functionality ---


def test_fit_returns_self():
    X, y = _make_low_noise_data()
    model = POPSRegression()
    assert model.fit(X, y) is model


def test_fitted_attributes():
    X, y = _make_low_noise_data()
    model = POPSRegression().fit(X, y)

    n_features = X.shape[1]
    assert model.coef_.shape == (n_features,)
    assert model.intercept_ == 0.0
    assert model.sigma_.shape == (n_features, n_features)
    assert model.misspecification_sigma_.shape == (n_features, n_features)
    assert model.posterior_samples_.shape[0] == n_features
    assert model.pointwise_corrections_.shape == X.shape
    assert model.leverage_.shape == (X.shape[0],)
    assert model.selected_.dtype == bool
    assert model.n_features_in_ == n_features
    assert model.alpha_ > 0 and model.lambda_ > 0 and model.n_iter_ >= 1
    for name in ("coef_", "sigma_", "posterior_samples_", "leverage_"):
        assert isinstance(getattr(model, name), np.ndarray)


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        POPSRegression().predict(np.ones((2, 3)))


def test_predict_shapes_and_signs():
    X, y = _make_low_noise_data()
    model = POPSRegression(random_state=0).fit(X, y)
    assert model.predict(X).shape == (X.shape[0],)

    y_pred, y_std, y_max, y_min, y_epi_std = model.predict(
        X, return_std=True, return_bounds=True, return_epistemic_std=True
    )
    for out in (y_pred, y_std, y_max, y_min, y_epi_std):
        assert out.shape == (X.shape[0],)
    assert np.all(y_std >= y_epi_std) and np.all(y_epi_std >= 0)
    assert_array_less(y_min, y_max + 1e-10)
    assert_allclose(model.predict(X, return_epistemic_std=True)[1], y_epi_std)
    assert_allclose(model.predict(X, return_bounds=True)[1], y_max)


def test_predict_checks_the_feature_count():
    X, y = _make_low_noise_data()
    model = POPSRegression().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(X[:, :-1])


def test_score_is_r_squared():
    X, y = _make_low_noise_data()
    model = POPSRegression().fit(X, y)
    residual = y - model.predict(X)
    expected = 1.0 - residual @ residual / np.sum((y - y.mean()) ** 2)
    assert_allclose(model.score(X, y), expected)


# --- Mean fit ---


def test_mean_fit_is_the_ridge_solution_of_its_precisions():
    X, y = _make_low_noise_data()
    model = POPSRegression().fit(X, y)
    A = _gram(model, X)
    assert_allclose(model.coef_, np.linalg.solve(A, X.T @ y), rtol=1e-8)
    assert_allclose(model.sigma_, np.linalg.inv(A) / model.alpha_, rtol=1e-8)


@pytest.mark.parametrize("shape", [(60, 5), (20, 30)])
@pytest.mark.parametrize("weighted", [False, True])
def test_mean_fit_matches_scikit_learn(shape, weighted):
    linear_model = pytest.importorskip("sklearn.linear_model")
    rng = np.random.RandomState(1)
    X = rng.randn(*shape)
    y = X @ rng.randn(shape[1]) + 0.1 * np.sin(3 * X[:, 0])
    weights = rng.uniform(0.5, 2.0, shape[0]) if weighted else None

    model = POPSRegression().fit(X, y, sample_weight=weights)
    reference = linear_model.BayesianRidge(fit_intercept=False)
    reference.fit(X, y, sample_weight=weights)
    assert model.n_iter_ == reference.n_iter_
    assert_allclose(model.coef_, reference.coef_, rtol=1e-7, atol=1e-10)
    assert_allclose(model.alpha_, reference.alpha_, rtol=1e-7)
    assert_allclose(model.lambda_, reference.lambda_, rtol=1e-7)
    assert_allclose(model.sigma_, reference.sigma_, rtol=1e-6, atol=1e-12)


def test_initial_precisions_are_used():
    X, y = _make_low_noise_data()
    default = POPSRegression(max_iter=1).fit(X, y)
    started = POPSRegression(max_iter=1, alpha_init=1e3, lambda_init=1e-3).fit(X, y)
    assert not np.isclose(default.alpha_, started.alpha_)


# --- Pointwise corrections and leverage ---


def test_unregularized_corrections_fit_every_point_exactly():
    X, y = _make_low_noise_data()
    model = POPSRegression(leverage_regularization=0.0).fit(X, y)
    fitted = np.einsum("ij,ij->i", X, model.coef_ + model.pointwise_corrections_)
    assert_allclose(fitted, y, rtol=1e-7, atol=1e-7 * np.abs(y).max())


@pytest.mark.parametrize("regularization", [0.0, 1e-2, 1.0])
def test_corrections_follow_the_regularized_leverage_formula(regularization):
    X, y = _make_low_noise_data()
    model = POPSRegression(leverage_regularization=regularization).fit(X, y)
    solved = np.linalg.solve(_gram(model, X), X.T).T
    leverage = np.einsum("ij,ij->i", X, solved)
    errors = y - X @ model.coef_
    eps = regularization * leverage.mean()
    expected = (errors / (leverage + eps))[:, None] * solved
    assert_allclose(model.leverage_, leverage, rtol=1e-7)
    scale = np.abs(expected).max()
    assert_allclose(
        model.pointwise_corrections_, expected, rtol=1e-6, atol=1e-9 * scale
    )


def test_low_leverage_points_destabilize_the_exact_pointwise_fit():
    """A few rows of vanishing leverage dominate the unregularized posterior."""
    X_ref, y_ref, X_test = _make_low_leverage_data(scale=1.0)
    X_low, y_low, _ = _make_low_leverage_data(scale=1e-4)

    def mean_std(X, y, regularization):
        model = POPSRegression(
            leverage_regularization=regularization, random_state=0
        ).fit(X, y)
        return model.predict(X_test, return_std=True)[1].mean()

    reference = mean_std(X_ref, y_ref, 0.0)
    assert mean_std(X_low, y_low, 0.0) > 100 * reference
    default = POPSRegression().leverage_regularization
    assert default > 0
    assert_allclose(mean_std(X_low, y_low, default), reference, rtol=0.2)
    # On a well-conditioned design the default changes little.
    assert_allclose(mean_std(X_ref, y_ref, default), reference, rtol=0.02)


def test_regularized_corrections_are_bounded_in_the_fit_metric():
    """||d_i||_A <= |e_i| / (2 sqrt(eps)) whatever the leverage."""
    X, y, _ = _make_low_leverage_data(scale=1e-6)
    regularization = 1e-2
    model = POPSRegression(leverage_regularization=regularization).fit(X, y)
    d = model.pointwise_corrections_
    norm_A = np.sqrt(np.einsum("ij,ij->i", d @ _gram(model, X), d))
    eps = regularization * model.leverage_.mean()
    bound = np.abs(y - X @ model.coef_) / (2.0 * np.sqrt(eps))
    assert np.all(norm_A <= bound * (1 + 1e-8))


# --- Selection by correction norm ---


def test_selection_is_relative_to_the_rms_correction_norm():
    X, y = _make_low_noise_data()
    model = POPSRegression(minimum_relative_correction=0.5).fit(X, y)
    norm = np.linalg.norm(model.pointwise_corrections_, axis=1)
    expected = norm >= 0.5 * np.sqrt(np.mean(norm**2))
    assert np.array_equal(model.selected_, expected)
    assert 0 < model.selected_.sum() < X.shape[0]


def test_selection_keeps_everything_at_zero_threshold():
    X, y = _make_low_noise_data()
    assert POPSRegression(minimum_relative_correction=0.0).fit(X, y).selected_.all()


def test_selection_is_scale_invariant():
    X, y = _make_low_noise_data()
    model = POPSRegression(minimum_relative_correction=0.5).fit(X, y)
    scaled = POPSRegression(minimum_relative_correction=0.5).fit(X, 1000.0 * y)
    assert np.array_equal(model.selected_, scaled.selected_)


def test_selection_falls_back_when_all_filtered():
    X, y = _make_low_noise_data()
    model = POPSRegression(minimum_relative_correction=1e12).fit(X, y)
    assert model.selected_.all()


def test_pruned_points_do_not_enter_the_posterior():
    X, y = _make_low_noise_data()
    model = POPSRegression(minimum_relative_correction=0.5, posterior="ensemble")
    model.fit(X, y)
    kept = model.pointwise_corrections_[model.selected_]
    assert_allclose(model.posterior_samples_, kept.T)
    assert_allclose(model.misspecification_sigma_, kept.T @ kept / kept.shape[0])


def test_hypercube_is_the_bounding_box_of_the_selected_corrections():
    X, y = _make_low_noise_data()
    model = POPSRegression(minimum_relative_correction=0.3).fit(X, y)
    support = model._hypercube_support
    low, high = model._hypercube_bounds
    assert_allclose(support.T @ support, np.eye(support.shape[1]), atol=1e-10)
    projected = model.pointwise_corrections_[model.selected_] @ support
    assert_allclose(projected.min(axis=0), low, rtol=1e-8, atol=1e-12)
    assert_allclose(projected.max(axis=0), high, rtol=1e-8, atol=1e-12)
    inside = support.T @ model.posterior_samples_
    assert np.all(inside >= low[:, None] - 1e-9) and np.all(
        inside <= high[:, None] + 1e-9
    )


def test_percentile_clipping_shrinks_the_hypercube():
    X, y = _make_low_noise_data(n_samples=200)
    full = POPSRegression(minimum_relative_correction=0.0).fit(X, y)
    clipped = POPSRegression(minimum_relative_correction=0.0, percentile_clipping=10.0)
    clipped.fit(X, y)
    width = lambda m: np.sum(m._hypercube_bounds[1] - m._hypercube_bounds[0])
    assert width(clipped) < width(full)


def test_minimum_relative_error_is_deprecated_and_forwarded():
    X, y = _make_low_noise_data()
    with pytest.warns(FutureWarning, match="minimum_relative_error"):
        old = POPSRegression(minimum_relative_error=0.5).fit(X, y)
    new = POPSRegression(minimum_relative_correction=0.5).fit(X, y)
    assert np.array_equal(old.selected_, new.selected_)


def test_no_warning_by_default():
    X, y = _make_low_noise_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        POPSRegression().fit(X, y)


# --- Posterior types and resampling ---


@pytest.mark.parametrize("posterior", ["hypercube", "ensemble"])
def test_posterior_types(posterior):
    X, y = _make_low_noise_data()
    model = POPSRegression(posterior=posterior, random_state=0).fit(X, y)
    n_features = X.shape[1]
    assert model.posterior_samples_.shape[0] == n_features
    assert model.misspecification_sigma_.shape == (n_features, n_features)
    _, y_std, y_max, y_min = model.predict(X, return_std=True, return_bounds=True)
    assert np.all(y_std >= 0) and np.all(y_max >= y_min)


def test_random_state_makes_resampling_reproducible():
    X, y = _make_low_noise_data()
    first = POPSRegression(random_state=7).fit(X, y).posterior_samples_
    second = POPSRegression(random_state=7).fit(X, y).posterior_samples_
    assert_allclose(first, second)


def test_random_state_none_uses_the_global_rng():
    """The historical np.random.seed reproducibility must be preserved."""
    X, y = _make_low_noise_data()
    np.random.seed(0)
    first = POPSRegression().fit(X, y).posterior_samples_
    np.random.seed(0)
    second = POPSRegression().fit(X, y).posterior_samples_
    assert_allclose(first, second)


@pytest.mark.parametrize("method", ["uniform", "sobol", "latin", "halton"])
def test_resampling_methods(method):
    X, y = _make_low_noise_data()
    model = POPSRegression(resampling_method=method, random_state=0).fit(X, y)
    assert model.posterior_samples_.shape[1] >= 64
    assert np.all(model.predict(X, return_std=True)[1] >= 0)


def test_resample_density_sets_the_number_of_samples():
    X, y = _make_low_noise_data(n_samples=200)
    model = POPSRegression(resample_density=3.0, random_state=0).fit(X, y)
    assert model.posterior_samples_.shape[1] == 600


# --- POPS vs epistemic uncertainty ---


def test_misspecification_dominates_epistemic_uncertainty():
    """In the low-noise misspecified regime POPS is far wider than sigma_."""
    X, y = _make_low_noise_data(n_samples=30, n_features=5, noise=0.001)
    model = POPSRegression(random_state=0).fit(X, y)
    assert np.trace(model.misspecification_sigma_) > np.trace(model.sigma_)
    _, std, epistemic_std = model.predict(X, return_std=True, return_epistemic_std=True)
    assert std.mean() > epistemic_std.mean()


# --- fit_intercept ---


def test_fit_intercept_separates_the_intercept():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 3)
    y = X @ np.array([1.0, 2.0, 3.0]) + 5.0 + 0.01 * rng.randn(50)

    model = POPSRegression(fit_intercept=True, random_state=0).fit(X, y)
    assert model.coef_.shape == (3,)
    assert_allclose(model.coef_, [1.0, 2.0, 3.0], atol=0.02)
    assert_allclose(model.intercept_, 5.0, atol=0.02)
    assert_allclose(model.predict(X), X @ model.coef_ + model.intercept_)
    # The intercept takes part in the posterior: it is the last coordinate.
    assert model.sigma_.shape == (4, 4)
    assert model.posterior_samples_.shape[0] == 4
    assert model.get_params()["fit_intercept"] is True

    explicit = POPSRegression(random_state=0).fit(np.hstack([X, np.ones((50, 1))]), y)
    assert_allclose(model.coef_, explicit.coef_[:-1], rtol=1e-10)
    assert_allclose(model.intercept_, explicit.coef_[-1], rtol=1e-10)


# --- sample_weight ---


def test_integer_weights_repeat_rows_in_the_mean_fit():
    X, y = _make_low_noise_data()
    weights = np.ones(X.shape[0])
    weights[:10] = 2.0
    weighted = POPSRegression().fit(X, y, sample_weight=weights)
    repeated = POPSRegression().fit(np.vstack([X, X[:10]]), np.concatenate([y, y[:10]]))
    assert_allclose(weighted.coef_, repeated.coef_, rtol=1e-6)
    assert_allclose(weighted.alpha_, repeated.alpha_, rtol=1e-6)


def test_zero_weight_points_are_excluded_from_the_posterior():
    X, y = _make_low_noise_data()
    y_outlier = y.copy()
    y_outlier[3] += 1e4
    weights = np.ones(X.shape[0])
    weights[3] = 0.0

    kwargs = dict(minimum_relative_correction=0.0, posterior="ensemble")
    dropped = POPSRegression(**kwargs).fit(X, y_outlier, sample_weight=weights)
    deleted = POPSRegression(**kwargs).fit(np.delete(X, 3, axis=0), np.delete(y, 3))
    assert not dropped.selected_[3]
    assert_allclose(dropped.coef_, deleted.coef_, rtol=1e-6)
    assert_allclose(
        dropped.misspecification_sigma_, deleted.misspecification_sigma_, rtol=1e-5
    )


def test_invalid_sample_weight():
    X, y = _make_low_noise_data()
    with pytest.raises(ValueError):
        POPSRegression().fit(X, y, sample_weight=-np.ones(X.shape[0]))
    with pytest.raises(ValueError):
        POPSRegression().fit(X, y, sample_weight=np.ones(3))


# --- Input and parameter validation ---


@pytest.mark.parametrize(
    "params",
    [
        {"posterior": "invalid"},
        {"resampling_method": "invalid"},
        {"percentile_clipping": 60.0},
        {"leverage_regularization": -1.0},
        {"minimum_relative_correction": -0.1},
        {"mode_threshold": 0.0},
    ],
)
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        POPSRegression(**params).fit(*_make_low_noise_data())


def test_invalid_inputs():
    X, y = _make_low_noise_data()
    with pytest.raises(ValueError):
        POPSRegression().fit(X, y[:-1])
    with pytest.raises(ValueError):
        POPSRegression().fit(X[:, 0], y)
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        POPSRegression().fit(X_bad, y)


def test_more_features_than_samples():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 30)
    y = X @ rng.randn(30)
    model = POPSRegression(random_state=0).fit(X, y)
    assert np.all(np.isfinite(model.predict(X, return_std=True)[1]))


# --- Parameters ---


def test_get_set_params_and_repr():
    model = POPSRegression(posterior="ensemble")
    params = model.get_params()
    assert params["posterior"] == "ensemble"
    assert "leverage_regularization" in params
    assert "minimum_relative_correction" in params
    assert repr(model) == "POPSRegression(posterior='ensemble')"

    assert model.set_params(posterior="hypercube", resample_density=5.0) is model
    assert model.resample_density == 5.0
    assert repr(model) == "POPSRegression(resample_density=5.0)"
    with pytest.raises(ValueError):
        model.set_params(not_a_parameter=1)

    rebuilt = POPSRegression(**model.get_params())
    assert rebuilt.get_params() == model.get_params()


def test_importing_the_package_does_not_import_scikit_learn():
    code = "import sys, popsregression; sys.exit('sklearn' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# --- Prior shape ---


def _prior_problem(seed=0, n_samples=80, n_features=6):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = X @ rng.randn(n_features) + 0.3 * np.sin(3 * X[:, 0])
    root = rng.randn(n_features, n_features)
    return X, y, root @ root.T + 0.1 * np.eye(n_features), rng


@pytest.mark.parametrize("diagonal", [True, False])
def test_prior_precision_sets_the_metric_of_the_fit(diagonal):
    """A = X^T X + (lambda / alpha) Lambda for the mean and the corrections."""
    X, y, precision, _ = _prior_problem()
    prior = np.diag(precision).copy() if diagonal else precision
    Lambda = np.diag(prior) if diagonal else prior
    model = POPSRegression(prior_precision=prior, leverage_regularization=0.0)
    model.fit(X, y)

    A = X.T @ X + (model.lambda_ / model.alpha_) * Lambda
    assert_allclose(model.coef_, np.linalg.solve(A, X.T @ y), rtol=1e-8)
    assert_allclose(model.sigma_, np.linalg.inv(A) / model.alpha_, rtol=1e-7)
    solved = np.linalg.solve(A, X.T).T
    leverage = np.einsum("ij,ij->i", X, solved)
    expected = ((y - X @ model.coef_) / leverage)[:, None] * solved
    assert_allclose(model.leverage_, leverage, rtol=1e-7)
    assert_allclose(model.pointwise_corrections_, expected, rtol=1e-6, atol=1e-10)
    # The corrections are still the exact pointwise fits.
    fitted = np.einsum("ij,ij->i", X, model.coef_ + model.pointwise_corrections_)
    assert_allclose(fitted, y, rtol=1e-7, atol=1e-8)


def test_prior_precision_is_the_isotropic_fit_of_the_whitened_design():
    X, y, precision, _ = _prior_problem()
    factor = np.linalg.cholesky(precision)
    T = np.linalg.inv(factor).T
    kwargs = dict(posterior="ensemble", minimum_relative_correction=0.0)
    prior = POPSRegression(prior_precision=precision, **kwargs).fit(X, y)
    whitened = POPSRegression(**kwargs).fit(X @ T, y)
    assert prior.n_iter_ == whitened.n_iter_
    assert_allclose(prior.lambda_, whitened.lambda_, rtol=1e-8)
    assert_allclose(prior.coef_, T @ whitened.coef_, rtol=1e-8)
    assert_allclose(prior.predict(X), whitened.predict(X @ T), rtol=1e-8)
    assert_allclose(
        prior.pointwise_corrections_, whitened.pointwise_corrections_ @ T.T, atol=1e-9
    )


def test_prior_precision_scale_is_learned():
    """Only the shape of Lambda matters; identity is the default prior."""
    X, y, precision, _ = _prior_problem()
    kwargs = dict(posterior="ensemble", minimum_relative_correction=0.0)
    base = POPSRegression(prior_precision=precision, **kwargs).fit(X, y)
    scaled = POPSRegression(prior_precision=100.0 * precision, **kwargs).fit(X, y)
    assert_allclose(scaled.lambda_, base.lambda_ / 100.0, rtol=1e-6)
    assert_allclose(scaled.coef_, base.coef_, rtol=1e-8)
    assert_allclose(
        scaled.misspecification_sigma_, base.misspecification_sigma_, rtol=1e-7
    )

    default = POPSRegression(**kwargs).fit(X, y)
    identity = POPSRegression(prior_precision=np.ones(X.shape[1]), **kwargs).fit(X, y)
    assert_allclose(identity.coef_, default.coef_, rtol=1e-10)
    assert_allclose(
        identity.misspecification_sigma_, default.misspecification_sigma_, rtol=1e-8
    )


def test_prior_precision_shrinks_the_penalized_direction():
    rng = np.random.RandomState(0)
    X = rng.randn(8, 4)
    y = X @ np.array([1.0, 1.0, 1.0, 1.0]) + 0.5 * rng.randn(8)
    flat = POPSRegression().fit(X, y)
    stiff = POPSRegression(prior_precision=[1.0, 1.0, 1.0, 1e6]).fit(X, y)
    assert abs(stiff.coef_[3]) < 0.05 * abs(flat.coef_[3])
    assert stiff.sigma_[3, 3] < 1e-3 * flat.sigma_[3, 3]


def test_hypercube_with_a_prior_is_the_box_of_its_corrections():
    X, y, precision, _ = _prior_problem()
    model = POPSRegression(prior_precision=precision, random_state=0).fit(X, y)
    support = model._hypercube_support
    low, high = model._hypercube_bounds
    assert_allclose(support.T @ support, np.eye(support.shape[1]), atol=1e-10)
    projected = model.pointwise_corrections_[model.selected_] @ support
    assert_allclose(projected.min(axis=0), low, rtol=1e-8, atol=1e-12)
    assert_allclose(projected.max(axis=0), high, rtol=1e-8, atol=1e-12)


def test_prior_precision_with_intercept():
    X, y, precision, _ = _prior_problem()
    y = y + 4.0
    design = np.hstack([X, np.ones((X.shape[0], 1))])
    augmented = np.eye(X.shape[1] + 1)
    augmented[:-1, :-1] = precision

    # Given for the features only, the intercept gets unit prior precision.
    model = POPSRegression(prior_precision=precision, fit_intercept=True).fit(X, y)
    explicit = POPSRegression(prior_precision=augmented).fit(design, y)
    assert_allclose(model.coef_, explicit.coef_[:-1], rtol=1e-8)
    assert_allclose(model.intercept_, explicit.coef_[-1], rtol=1e-8)
    assert model.posterior_samples_.shape[0] == X.shape[1] + 1

    # It can also be given with a row and column for the intercept.
    loose = augmented.copy()
    loose[-1, -1] = 1e-8
    free = POPSRegression(prior_precision=loose, fit_intercept=True).fit(X, y)
    explicit = POPSRegression(prior_precision=loose).fit(design, y)
    assert_allclose(free.intercept_, explicit.coef_[-1], rtol=1e-8)
    assert not np.isclose(free.intercept_, model.intercept_, rtol=1e-6)


@pytest.mark.parametrize(
    "prior",
    [
        np.ones(3),
        -np.ones(6),
        np.ones((6, 5)),
        np.arange(36.0).reshape(6, 6),
        -np.eye(6),
        np.full(6, np.nan),
    ],
)
def test_invalid_prior_precision(prior):
    X, y, _, _ = _prior_problem()
    with pytest.raises(ValueError, match="prior_precision"):
        POPSRegression(prior_precision=prior).fit(X, y)
