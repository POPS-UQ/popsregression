"""Tests for POPSRegressionEllipse."""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import os
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.estimator_checks import parametrize_with_checks

from popsregression import POPSRegression, POPSRegressionEllipse


def _target_function(x):
    """Oscillatory target from SimpleExample.ipynb (misspecified setup)."""
    return (x**3 + 0.01 * x**4) * 0.1 + np.sin(x) * x * 10.0


def _make_misspecified_data(n_samples, seed=42, degree=4):
    """Quartic-polynomial-fits-oscillatory-function setup."""
    rng = np.random.RandomState(seed)
    x_train = (
        np.sort(np.append(rng.uniform(-1, 1, n_samples), np.linspace(-1, 1, 2))) * 10
    )
    x_dense = np.linspace(-1.1, 1.1, 51) * 10
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_train = poly.fit_transform(x_train.reshape(-1, 1))
    X_dense = poly.fit_transform(x_dense.reshape(-1, 1))
    return X_train, _target_function(x_train), X_dense, _target_function(x_dense)


def _make_well_specified_data(n_samples=60, n_features=5, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    theta = rng.randn(n_features)
    return X, X @ theta, theta


# --- Basic functionality ---


def test_fit_returns_self():
    X, y, _ = _make_well_specified_data()
    model = POPSRegressionEllipse(random_state=0)
    assert model.fit(X, y) is model


def test_fitted_attributes():
    X, y, _ = _make_well_specified_data()
    model = POPSRegressionEllipse(random_state=0).fit(X, y)

    n_features = X.shape[1]
    assert model.coef_.shape == (n_features,)
    assert model.center_whitened_.shape == (n_features,)
    assert model.U_.shape == (n_features, model.rank_)
    assert model.ellipsoid_B_.shape == (n_features, n_features)
    assert model.baseline_B0_.shape == (n_features, n_features)
    assert 0.0 <= model.coverage_fraction_ <= 1.0
    assert np.isfinite(model.objective_)
    assert model.n_iter_ > 0
    assert model.n_outer_iter_ == 1


def test_predict_shapes():
    X, y, _ = _make_well_specified_data()
    model = POPSRegressionEllipse(random_state=0).fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == (X.shape[0],)

    y_pred, y_std = model.predict(X, return_std=True)
    assert y_std.shape == (X.shape[0],)
    assert np.all(y_std >= 0)

    y_pred, y_std, y_max, y_min = model.predict(X, return_std=True, return_bounds=True)
    assert_array_less(y_min, y_max + 1e-12)

    y_pred, y_max, y_min = model.predict(X, return_bounds=True)
    assert_array_less(y_min, y_max + 1e-12)


def test_std_is_pushforward_std_not_half_width():
    """std = sqrt(v/(P+2)); bounds = mean +/- sqrt(v)."""
    X, y, _ = _make_well_specified_data()
    model = POPSRegressionEllipse(random_state=0).fit(X, y)
    y_pred, y_std, y_max, y_min = model.predict(X, return_std=True, return_bounds=True)
    half_width = 0.5 * (y_max - y_min)
    n_dim = X.shape[1]
    assert_allclose(y_std, half_width / np.sqrt(n_dim + 2.0), rtol=1e-10)


def test_ellipsoid_B_consistent_with_predict():
    """x^T B x + delta^2 must equal the squared predictive half-width."""
    X, y, _ = _make_well_specified_data()
    for fit_intercept in [False, True]:
        model = POPSRegressionEllipse(random_state=0, fit_intercept=fit_intercept).fit(
            X, y
        )
        y_pred, y_max, y_min = model.predict(X, return_bounds=True)
        v = (0.5 * (y_max - y_min)) ** 2
        B = model.ellipsoid_B_
        if fit_intercept:
            D = np.hstack([X - X.mean(axis=0), np.ones((X.shape[0], 1))])
        else:
            D = X
        v_from_B = np.einsum("ij,jk,ik->i", D, B, D) + model.delta**2
        assert_allclose(v_from_B, v, rtol=1e-8, atol=1e-12)


# --- Well-specified limit: ellipsoid collapses (test 3) ---


def test_well_specified_limit():
    X, y, theta = _make_well_specified_data()
    model = POPSRegressionEllipse(random_state=0, tol=1e-12).fit(X, y)

    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert_allclose(model.coef_, ols, atol=1e-6)
    assert model.coverage_fraction_ == 1.0

    # The optimizer drives the widths s_i to ~0 (ellipsoid collapse):
    # the predictive squared half-width reduces to the delta floor.
    _, y_max, y_min = model.predict(X, return_bounds=True)
    s = (0.5 * (y_max - y_min)) ** 2 - model.delta**2
    assert np.max(np.abs(s)) < 1e-8


@pytest.mark.parametrize("baseline", ["pops", "ridge", "zero"])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_baselines_well_specified(baseline, fit_intercept):
    X, y, _ = _make_well_specified_data()
    model = POPSRegressionEllipse(
        random_state=0, baseline=baseline, fit_intercept=fit_intercept
    ).fit(X, y)
    assert model.coverage_fraction_ == 1.0
    assert model.score(X, y) > 0.99

    # Exercise the width/bounds and dense-matrix paths of every baseline.
    _, y_std, y_max, y_min = model.predict(X, return_std=True, return_bounds=True)
    assert np.all(y_std >= 0) and np.all(y_max >= y_min)
    n_dim = X.shape[1] + int(fit_intercept)
    assert model.ellipsoid_B_.shape == (n_dim, n_dim)
    assert model.baseline_B0_.shape == (n_dim, n_dim)
    if baseline == "ridge":
        assert_allclose(
            model.baseline_B0_, model.baseline_ridge * np.eye(n_dim), rtol=1e-12
        )
    elif baseline == "zero":
        assert_allclose(model.baseline_B0_, 0.0, atol=1e-15)


# --- Misspecified limit: honest bounds that do not shrink (test 4) ---


def test_misspecified_coverage_and_width_retention():
    widths = {}
    for n_samples in [50, 500]:
        X_train, y_train, X_dense, y_dense = _make_misspecified_data(n_samples)
        model = POPSRegressionEllipse(random_state=0).fit(X_train, y_train)

        # (a) every training point is covered at the final rho
        assert model.coverage_fraction_ == 1.0

        # (b) predictive [y_min, y_max] covers >= 95% of dense targets
        _, y_max, y_min = model.predict(X_dense, return_bounds=True)
        covered = np.mean((y_dense >= y_min) & (y_dense <= y_max))
        assert covered >= 0.95

        widths[n_samples] = np.mean(y_max - y_min)

    # (c) bounds do NOT shrink as N grows: misspecification is retained
    assert widths[500] / widths[50] > 0.8


# --- Anchoring against POPSRegression (test 5) ---


def test_predictive_std_anchored_to_pops():
    X_train, y_train, X_dense, _ = _make_misspecified_data(50)
    ellipse = POPSRegressionEllipse(random_state=0).fit(X_train, y_train)
    pops = POPSRegression().fit(X_train, y_train)

    _, ellipse_std = ellipse.predict(X_dense, return_std=True)
    _, pops_std = pops.predict(X_dense, return_std=True)
    ratio = np.mean(ellipse_std) / np.mean(pops_std)
    assert 1.0 / 3.0 < ratio < 3.0


# --- sklearn compliance (test 6) ---


@parametrize_with_checks([POPSRegressionEllipse()])
def test_sklearn_compatible(estimator, check):
    check(estimator)


def test_pipeline_polynomial_features():
    x = np.linspace(-2, 2, 40)
    y = np.sin(2 * x) * x
    pipe = make_pipeline(
        PolynomialFeatures(degree=3),
        POPSRegressionEllipse(random_state=0),
    )
    pipe.fit(x.reshape(-1, 1), y)
    y_pred = pipe.predict(x.reshape(-1, 1))
    assert y_pred.shape == (40,)


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_fit_intercept(fit_intercept):
    rng = np.random.RandomState(42)
    X = rng.randn(50, 3)
    y = X @ np.array([1.0, 2.0, 3.0]) + 5.0 * fit_intercept
    model = POPSRegressionEllipse(random_state=0, fit_intercept=fit_intercept).fit(X, y)
    assert model.score(X, y) > 0.999
    if fit_intercept:
        assert model.intercept_ == pytest.approx(5.0, abs=1e-3)
        assert model.coverage_fraction_ == 1.0
    else:
        assert model.intercept_ == 0.0


def test_weights_zero_drop_coverage_requirement():
    """Zero-weighted points do not have to be covered by the ellipsoid."""
    X, y, _ = _make_well_specified_data()
    y_outlier = y.copy()
    y_outlier[0] += 50.0

    weights = np.ones(X.shape[0])
    weights[0] = 0.0
    dropped = POPSRegressionEllipse(random_state=0, weights=weights).fit(X, y_outlier)
    kept = POPSRegressionEllipse(random_state=0).fit(X, y_outlier)

    _, std_dropped = dropped.predict(X, return_std=True)
    _, std_kept = kept.predict(X, return_std=True)
    # Covering the outlier requires a much wider ellipsoid.
    assert np.mean(std_kept) > 5.0 * np.mean(std_dropped)
    assert kept.coverage_fraction_ == 1.0
    assert dropped.coverage_fraction_ < 1.0


# --- Parameter validation and warnings ---


def test_invalid_rho_schedule():
    X, y, _ = _make_well_specified_data()
    with pytest.raises(ValueError, match="rho_schedule"):
        POPSRegressionEllipse(rho_schedule=(0.1, -0.1)).fit(X, y)
    with pytest.raises(ValueError, match="rho_schedule"):
        POPSRegressionEllipse(rho_schedule=()).fit(X, y)


def test_delta_zero_with_tiny_rho_warns():
    X, y, _ = _make_well_specified_data()
    with pytest.warns(UserWarning, match="delta=0"):
        POPSRegressionEllipse(delta=0.0, rho_schedule=(1e-1, 1e-9), max_iter=5).fit(
            X, y
        )


def test_invalid_baseline():
    X, y, _ = _make_well_specified_data()
    with pytest.raises(ValueError):
        POPSRegressionEllipse(baseline="invalid").fit(X, y)


# --- Determinism (test 7) ---


def test_determinism():
    X_train, y_train, _, _ = _make_misspecified_data(50)
    model_a = POPSRegressionEllipse(random_state=3).fit(X_train, y_train)
    model_b = POPSRegressionEllipse(random_state=3).fit(X_train, y_train)
    assert np.array_equal(model_a.coef_, model_b.coef_)
    assert np.array_equal(model_a.U_, model_b.U_)


# --- Posterior sampling ---


def test_sample_semantics():
    X, y, _ = _make_well_specified_data()
    model = POPSRegressionEllipse(random_state=0).fit(X, y)
    samples = model.sample(2000, random_state=0)
    assert samples.shape == (X.shape[1], 2000)

    # Every sampled parameter predicts inside the pushforward support.
    y_pred, y_max, y_min = model.predict(X, return_bounds=True)
    y_samples = X @ samples
    assert np.all(y_samples <= y_max[:, None] + 1e-10)
    assert np.all(y_samples >= y_min[:, None] - 1e-10)

    model_i = POPSRegressionEllipse(random_state=0, fit_intercept=True).fit(X, y + 3.0)
    samples_i = model_i.sample(100, random_state=0)
    assert samples_i.shape == (X.shape[1] + 1, 100)


# --- PAC-Bayes layer (test 8) ---


def test_pac_bayes_finite_components():
    X_train, y_train, _, _ = _make_misspecified_data(50)
    model = POPSRegressionEllipse(random_state=0, pac_bayes=True).fit(X_train, y_train)
    assert np.isfinite(model.kl_) and model.kl_ >= 0.0
    assert np.isfinite(model.bound_)
    assert np.isfinite(model.empirical_H_)
    # hyperprior_scale is relative: tau2_ = scale * ||psi0||^2 / d.
    psi0 = model._psi0
    assert model.tau2_ == pytest.approx(psi0 @ psi0 / psi0.size)
    d = model.center_whitened_.size * (1 + model.rank_)
    assert model.hyper_sigma_diag_.shape == (d,)
    assert np.all(model.hyper_sigma_diag_ > 0)
    # The bound dominates the hyperposterior-averaged empirical error.
    assert model.bound_ > model.empirical_H_


def test_pac_bayes_predictive_spread_added():
    """predict follows the documented delta-method formulas exactly."""
    X_train, y_train, X_dense, _ = _make_misspecified_data(50)
    pac = POPSRegressionEllipse(random_state=0, pac_bayes=True).fit(X_train, y_train)
    _, std_pac, max_pac, min_pac, bstd = pac.predict(
        X_dense, return_std=True, return_bounds=True, return_bound_std=True
    )

    # Bounds are the support of the FITTED ellipsoid: mean +/- sqrt(v)
    # with v = x^T B x + delta^2 (no hyperposterior folding).
    v_point = (
        np.einsum("ij,jk,ik->i", X_dense, pac.ellipsoid_B_, X_dense) + pac.delta**2
    )
    assert_allclose(0.5 * (max_pac - min_pac), np.sqrt(v_point), rtol=1e-8)

    # std averages over the hyperposterior: sqrt((v+dv)/(P+2) + z^2 Sc);
    # bound_std is the delta-method std of the bound curves:
    # sqrt(z^2 Sc + Var[v]/(4v)), Var[v] = 4 sum_m (zU_m)^2 (z^2 SU_m).
    _, Z = pac._whitened_design(X_dense)
    n_dim = X_dense.shape[1]
    sigma_u = pac.hyper_sigma_diag_[n_dim:].reshape(n_dim, -1)
    sigma_u_proj = (Z * Z) @ sigma_u
    d_v = np.sum(sigma_u_proj, axis=1)
    mean_var = (Z * Z) @ pac.hyper_sigma_diag_[:n_dim]
    var_v = 4.0 * np.sum((Z @ pac.U_) ** 2 * sigma_u_proj, axis=1)
    assert_allclose(
        std_pac, np.sqrt((v_point + d_v) / (n_dim + 2.0) + mean_var), rtol=1e-8
    )
    assert_allclose(bstd, np.sqrt(mean_var + var_v / (4.0 * v_point)), rtol=1e-8)
    assert np.all(d_v > 0) and np.all(bstd > 0)

    # A model fitted without pac_bayes has zero bound spread.
    bare = POPSRegressionEllipse(random_state=0).fit(X_train, y_train)
    _, bstd_bare = bare.predict(X_dense, return_bound_std=True)
    assert_allclose(bstd_bare, 0.0, atol=1e-15)


def test_pac_bayes_infinite_tau2_recovers_phase1():
    """tau2 -> inf is exactly the pac_bayes=False optimum (test 8c)."""
    X_train, y_train, _, _ = _make_misspecified_data(50)
    plain = POPSRegressionEllipse(random_state=0).fit(X_train, y_train)
    with warnings.catch_warnings():
        # The improper tau2=inf limit may trip the Hessian-floor warning.
        warnings.simplefilter("ignore", UserWarning)
        infinite = POPSRegressionEllipse(
            random_state=0, pac_bayes=True, hyperprior_scale=np.inf
        ).fit(X_train, y_train)
    assert_allclose(infinite.coef_, plain.coef_, atol=1e-8)
    assert np.isinf(infinite.kl_)


def test_pac_bayes_never_narrower_than_bare():
    """With hyperprior_center='phase1' the PAC layer only broadens.

    The hyperposterior is centered on the phase-1 optimum, so the MAP
    (and hence the ellipse max/min bounds) coincides with the bare fit
    exactly; the hyperposterior enters as a strictly positive predictive
    std inflation and bound spread, whose relative size decays with N
    (rate-N concentration on the bare values).
    """
    rel_broadening = {}
    for n_samples in [10, 500]:
        X_train, y_train, X_dense, _ = _make_misspecified_data(n_samples)
        bare = POPSRegressionEllipse(random_state=0).fit(X_train, y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pac = POPSRegressionEllipse(random_state=0, pac_bayes=True).fit(
                X_train, y_train
            )
        assert np.array_equal(bare.coef_, pac.coef_)
        assert np.array_equal(bare.U_, pac.U_)

        # Identical fitted ellipsoid -> identical support bounds...
        _, b_std, b_max, b_min = bare.predict(
            X_dense, return_std=True, return_bounds=True
        )
        _, p_std, p_max, p_min, p_bstd = pac.predict(
            X_dense, return_std=True, return_bounds=True, return_bound_std=True
        )
        assert_allclose(p_max, b_max, rtol=1e-12)
        assert_allclose(p_min, b_min, rtol=1e-12)
        # ...while std and the 2-sigma bound envelope strictly broaden.
        assert np.all(p_std > b_std)
        assert np.all(p_bstd > 0)
        envelope = (p_max + 2 * p_bstd) - (p_min - 2 * p_bstd)
        rel_broadening[n_samples] = np.mean(envelope / (b_max - b_min) - 1.0)
    assert 0.0 < rel_broadening[500] < rel_broadening[10]


def test_pac_bayes_update_hyperprior_converges():
    X_train, y_train, _, _ = _make_misspecified_data(50)
    model = POPSRegressionEllipse(
        random_state=0,
        pac_bayes=True,
        hyperprior_center="warm_start",
        update_hyperprior=True,
        n_outer=4,
    ).fit(X_train, y_train)
    assert np.isfinite(model.tau2_) and model.tau2_ > 0
    assert 1 <= model.n_outer_iter_ <= 4
    assert np.isfinite(model.bound_)


def test_update_hyperprior_ignored_with_phase1_centering():
    X_train, y_train, _, _ = _make_misspecified_data(50)
    with pytest.warns(UserWarning, match="ill-posed"):
        model = POPSRegressionEllipse(
            random_state=0, pac_bayes=True, update_hyperprior=True
        ).fit(X_train, y_train)
    assert model.n_outer_iter_ == 1


def test_pac_bayes_low_data_regime():
    """The PAC-Bayes layer is operative at N/P ~ 2 and tightens with N."""
    bounds = {}
    for n_samples in [10, 500]:
        X_train, y_train, X_dense, _ = _make_misspecified_data(n_samples)
        with warnings.catch_warnings():
            # The exact Hessian need not be PSD at a barrier-active
            # optimum; the floor warning is expected at tiny N.
            warnings.simplefilter("ignore", UserWarning)
            pac = POPSRegressionEllipse(random_state=0, pac_bayes=True).fit(
                X_train, y_train
            )
        assert pac.coverage_fraction_ == 1.0
        assert np.isfinite(pac.bound_) and np.isfinite(pac.kl_)
        assert 0.0 < pac.gamma_ < pac.hyper_sigma_diag_.size
        _, y_std = pac.predict(X_dense, return_std=True)
        assert np.all(np.isfinite(y_std)) and np.all(y_std > 0)
        bounds[n_samples] = pac.bound_
    # The hyperposterior concentrates at rate N: the PAC bound is loose
    # in the scarce-data regime and tightens as data accumulates.
    assert bounds[500] < bounds[10]


def test_pac_bayes_gamma_effective_dof():
    """0 < gamma < d on well-specified data (test 8e)."""
    X, y, _ = _make_well_specified_data()
    model = POPSRegressionEllipse(random_state=0, pac_bayes=True).fit(X, y)
    d = model.hyper_sigma_diag_.size
    assert 0.0 < model.gamma_ < d


# --- Frozen center ---


def test_optimize_center_false_keeps_pops_mean():
    """Frozen center reproduces the POPS pre-fit mean exactly."""
    X_train, y_train, X_dense, y_dense = _make_misspecified_data(50)
    frozen = POPSRegressionEllipse(random_state=0, optimize_center=False).fit(
        X_train, y_train
    )
    pops = POPSRegression(fit_intercept=False).fit(X_train, y_train)
    assert_allclose(frozen.coef_, pops.coef_, rtol=1e-12, atol=1e-12)
    assert frozen.coverage_fraction_ == 1.0

    # Widths still adapt: the bounds cover the dense targets.
    _, y_max, y_min = frozen.predict(X_dense, return_bounds=True)
    assert np.mean((y_dense >= y_min) & (y_dense <= y_max)) >= 0.95


def test_optimize_center_false_pac_bayes():
    """With a frozen center the hyperposterior covers the widths only."""
    X_train, y_train, _, _ = _make_misspecified_data(50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = POPSRegressionEllipse(
            random_state=0, optimize_center=False, pac_bayes=True
        ).fit(X_train, y_train)
    n_dim = model.center_whitened_.size
    assert_allclose(model.hyper_sigma_diag_[:n_dim], 0.0, atol=1e-15)
    assert np.all(model.hyper_sigma_diag_[n_dim:] > 0)
    assert np.isfinite(model.kl_) and np.isfinite(model.bound_)
    assert 0.0 < model.gamma_ < n_dim * model.rank_


def test_low_n_conservatism_recipe():
    """Frozen center + PAC is the conservative low-N configuration.

    At N/P ~ 2 the bare ellipse is the minimum covering support and
    deliberately tight; freezing the center and adding the 2-sigma
    hyperposterior envelope around the ellipse bounds restores
    conservatism (the PAC layer's main motivation), covering the dense
    truth far better than the bare fit at a fraction of the hypercube
    width.
    """
    X_train, y_train, X_dense, y_dense = _make_misspecified_data(10)
    x_dense = X_dense[:, 1]
    interp = np.abs(x_dense) <= 10.0

    bare = POPSRegressionEllipse(random_state=0).fit(X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        recipe = POPSRegressionEllipse(
            random_state=0, optimize_center=False, pac_bayes=True
        ).fit(X_train, y_train)
    hypercube = POPSRegression(leverage_percentile=0.0).fit(X_train, y_train)

    def coverage_and_width(y_max, y_min):
        covered = np.mean(((y_dense >= y_min) & (y_dense <= y_max))[interp])
        return covered, np.mean(0.5 * (y_max - y_min)[interp])

    _, b_max, b_min = bare.predict(X_dense, return_bounds=True)
    cov_bare, hw_bare = coverage_and_width(b_max, b_min)

    # Conservative envelope: ellipse max/min +/- 2 sigma hyperposterior.
    _, r_max, r_min, r_bstd = recipe.predict(
        X_dense, return_bounds=True, return_bound_std=True
    )
    cov_recipe, hw_recipe = coverage_and_width(r_max + 2 * r_bstd, r_min - 2 * r_bstd)

    _, h_max, h_min = hypercube.predict(X_dense, return_bounds=True)
    cov_box, hw_box = coverage_and_width(h_max, h_min)

    assert cov_recipe >= 0.95
    assert cov_recipe >= cov_box
    assert cov_recipe > cov_bare + 0.1
    assert hw_recipe > 1.2 * hw_bare  # conservatism costs width...
    assert hw_recipe < hw_box  # ...but far less than the box support


# --- Cloning ---


def test_clone_and_get_set_params():
    from sklearn.base import clone

    model = POPSRegressionEllipse(rank=4, baseline="ridge", pac_bayes=True)
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    model.set_params(rank=8, delta=1e-2)
    assert model.rank == 8 and model.delta == 1e-2


# --- Scaling: O(N P r) fit without dense P x P optimization ---


@pytest.mark.skipif(
    not os.environ.get("POPS_RUN_SLOW"),
    reason="slow scaling test; set POPS_RUN_SLOW=1 to run",
)
def test_large_problem_memory_and_time():
    """P=2000, r=32, N=20000 fit stays O(N*P) + whitening in memory.

    The optimizer must never allocate O(P^2) per-iteration state or an
    (N, P**2)-ish intermediate; the only dense (P, P) arrays are the
    whitening transform and its eigendecomposition. The tracemalloc peak
    bound (a few multiples of the design matrix) enforces this.
    """
    import time
    import tracemalloc

    n_samples, n_features = 20000, 2000
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    y = X @ rng.randn(n_features) + 0.1 * np.sin(X[:, 0] * 3)

    model = POPSRegressionEllipse(
        rank=32,
        baseline="ridge",
        rho_schedule=(1e-1,),
        max_iter=20,
        random_state=0,
    )
    tracemalloc.start()
    start = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    design_bytes = X.nbytes  # 320 MB
    assert peak < 4 * design_bytes, f"peak memory {peak / 1e9:.2f} GB"
    assert elapsed < 60.0, f"fit took {elapsed:.1f} s"
    assert model.U_.shape == (n_features, 32)
