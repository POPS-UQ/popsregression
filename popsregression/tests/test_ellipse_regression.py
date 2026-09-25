"""Tests of the public POPSEllipseRegression estimator."""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import trapezoid
from scipy.stats import beta as beta_dist
from scipy.stats import binom
from sklearn.utils.estimator_checks import parametrize_with_checks

from popsregression import POPSEllipseRegression
from popsregression._ellipse import DiagnosticBoundWarning, _EllipsoidPosterior
from popsregression._projected_ball import projected_ball_logpdf

Y_BOUNDS = (-4.0, 4.0)


def _quartic_data(n, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, n)
    X = np.vander(x, 5, increasing=True)
    y = np.sin(3.0 * x) + 0.5 * x
    return X, y


@parametrize_with_checks([POPSEllipseRegression()])
def test_sklearn_compatible(estimator, check):
    check(estimator)


# --- regularization=None and 'empirical-bayes' -------------------------------


def test_bare_matches_engine():
    X, y = _quartic_data(40)
    model = POPSEllipseRegression(random_state=0).fit(X, y)
    rng = np.random.RandomState(0)
    engine = _EllipsoidPosterior(random_state=int(rng.randint(2**31 - 1))).fit(X, y)
    mean, y_max, y_min = model.predict(X, return_bounds=True)
    ref_mean, ref_max, ref_min = engine.predict(X, return_bounds=True)
    assert_allclose(mean, ref_mean, atol=1e-10)
    assert_allclose(y_max, ref_max, atol=1e-10)
    assert_allclose(y_min, ref_min, atol=1e-10)
    assert_allclose(model.coef_, engine.coef_, atol=1e-10)
    assert model.certificate_status_ == "none"


def test_bare_interval_is_exact_projected_ball_quantile():
    X, y = _quartic_data(40)
    model = POPSEllipseRegression(random_state=0).fit(X, y)
    mean, y_max, _ = model.predict(X, return_bounds=True)
    half = y_max - mean
    lo, hi = model.predict_interval(X, level=0.9)
    a = 0.5 * (model._ball_dim + 1)
    expected = (2.0 * beta_dist.ppf(0.95, a, a) - 1.0) * half
    assert_allclose(hi - mean, expected, rtol=1e-8, atol=1e-10)
    assert_allclose(mean - lo, expected, rtol=1e-8, atol=1e-10)


def test_empirical_bayes_matches_engine_and_broadens():
    X, y = _quartic_data(30)
    rng = np.random.RandomState(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiagnosticBoundWarning)
        engine = _EllipsoidPosterior(
            random_state=int(rng.randint(2**31 - 1)), pac_bayes=True
        ).fit(X, y)
    model = POPSEllipseRegression(regularization="empirical-bayes", random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DiagnosticBoundWarning)
        model.fit(X, y)
    assert model.certificate_status_ == "diagnostic_empirical_bayes"
    assert not hasattr(model, "bound_")
    assert_allclose(model.diagnostic_bound_, engine.bound_)
    bare = POPSEllipseRegression(random_state=0).fit(X, y)
    lo_b, hi_b = bare.predict_interval(X)
    lo, hi = model.predict_interval(X)
    assert np.mean(hi - lo) > np.mean(hi_b - lo_b)


@pytest.mark.parametrize("regularization", [None, "empirical-bayes"])
def test_logpdf_normalized_and_cdf_consistent(regularization):
    X, y = _quartic_data(25)
    model = POPSEllipseRegression(
        regularization=regularization, n_hyper_samples=64, random_state=0
    ).fit(X, y)
    x_star = X[:3]
    _, y_max, y_min = model.predict(x_star, return_bounds=True)
    for k in range(3):
        grid = np.linspace(y_min[k], y_max[k], 20001)
        dens = np.exp(
            model.predict_logpdf(np.repeat(x_star[k : k + 1], grid.size, 0), grid)
        )
        assert_allclose(trapezoid(dens, grid), 1.0, atol=2e-3)
    lo, hi = model.predict_interval(x_star, level=0.8)
    assert_allclose(model.predict_cdf(x_star, lo), 0.1, atol=1e-6)
    assert_allclose(model.predict_cdf(x_star, hi), 0.9, atol=1e-6)


def test_logpdf_is_minus_inf_outside_bare_support():
    X, y = _quartic_data(25)
    model = POPSEllipseRegression(random_state=0).fit(X, y)
    _, y_max, _ = model.predict(X[:2], return_bounds=True)
    assert np.all(np.isneginf(model.predict_logpdf(X[:2], y_max + 1.0)))


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_parameter_samples_lie_in_the_predictive_support(fit_intercept):
    X, y = _quartic_data(40)
    X = X[:, 1:] if fit_intercept else X
    model = POPSEllipseRegression(fit_intercept=fit_intercept, random_state=0).fit(X, y)
    theta, offset = model.sample(500, random_state=1, return_offset=True)
    assert np.all(np.abs(offset) <= model.delta + 1e-15)
    design = np.hstack([X, np.ones((X.shape[0], 1))]) if fit_intercept else X
    preds = design @ theta + offset
    _, y_max, y_min = model.predict(X, return_bounds=True)
    assert np.all(preds <= y_max[:, None] + 1e-8)
    assert np.all(preds >= y_min[:, None] - 1e-8)
    mean = model.predict(X)
    assert_allclose(preds.mean(axis=1), mean, atol=0.1 * np.std(y))
    # sample() without offsets is the coefficient marginal of the same draw.
    assert_allclose(model.sample(500, random_state=1), theta)


@pytest.mark.parametrize("regularization", [None, "empirical-bayes", "PAC"])
def test_propagated_draws_follow_the_scored_predictive(regularization):
    """Joint draws have exactly the scored marginal (floor included)."""
    X, y = _quartic_data(40)
    kwargs = {"y_bounds": Y_BOUNDS} if regularization == "PAC" else {}
    model = POPSEllipseRegression(
        regularization=regularization, delta=0.3, random_state=0, **kwargs
    ).fit(X, y)
    x = X[:3]
    draws = model.sample_predictions(x, 20000, random_state=2)
    for i in range(3):
        grid = np.quantile(draws[i], [0.1, 0.3, 0.5, 0.7, 0.9])
        assert_allclose(
            model.predict_cdf(np.repeat(x[i : i + 1], grid.size, 0), grid),
            [0.1, 0.3, 0.5, 0.7, 0.9],
            atol=0.015,
        )
    if regularization != "PAC":
        theta, offset = model.sample(20000, random_state=2, return_offset=True)
        assert_allclose(x @ theta + offset, draws)


def test_offset_coordinate_sets_the_ball_dimension():
    X, y = _quartic_data(40)
    with_floor = POPSEllipseRegression(random_state=0).fit(X, y)
    assert with_floor._ball_dim == X.shape[1] + 1
    without = POPSEllipseRegression(delta=0.0, rho_schedule=(1e-1, 1e-2)).fit(X, y)
    assert without._ball_dim == X.shape[1]
    _, offset = without.sample(10, random_state=0, return_offset=True)
    assert np.all(offset == 0.0)


def test_preprocessor_is_fitted_on_pilot_units_only():
    from sklearn.decomposition import PCA

    rng = np.random.RandomState(0)
    raw = rng.randn(60, 8)
    y = raw[:, 0] - 0.5 * raw[:, 1] + 0.1 * np.sin(3 * raw[:, 2])
    model = POPSEllipseRegression(
        regularization="PAC",
        preprocessor=PCA(n_components=4),
        y_bounds=(-10, 10),
        n_bound_samples=300,
        random_state=0,
    ).fit(raw, y)
    assert model.n_features_in_ == 8
    assert model._n_coef == 4
    pilots = []
    for _, kernel, _ in model.components_:
        pre = kernel.preprocessor
        assert pre is not model.preprocessor
        pilots.append(pre.mean_)
    # Each fold's PCA saw only its own pilot half: the two halves, and not
    # the whole sample. The two pilot halves partition the sample.
    assert not np.allclose(pilots[0], pilots[1])
    assert not np.allclose(pilots[0], raw.mean(axis=0))
    assert_allclose(0.5 * (pilots[0] + pilots[1]), raw.mean(axis=0))
    lo, hi = model.predict_interval(raw[:5])
    assert np.all(lo < hi)
    assert model.sample_predictions(raw[:5], 7).shape == (5, 7)
    with pytest.raises(ValueError, match="sample_predictions"):
        model.sample(3)


# --- regularization='PAC' ------------------------------------------------------


def _pac(**kwargs):
    kwargs.setdefault("y_bounds", Y_BOUNDS)
    kwargs.setdefault("random_state", 0)
    kwargs.setdefault("n_bound_samples", 500)
    return POPSEllipseRegression(regularization="PAC", **kwargs)


def test_pac_requires_known_output_bounds():
    X, y = _quartic_data(20)
    with pytest.raises(ValueError, match="requires y_bounds"):
        POPSEllipseRegression(regularization="PAC").fit(X, y)
    with pytest.raises(ValueError, match="outside y_bounds"):
        _pac(y_bounds=(-0.5, 0.5)).fit(X, y)
    with pytest.raises(ValueError, match="sample_weight"):
        _pac().fit(X, y, sample_weight=np.ones(y.size))


def test_pac_bound_decomposition():
    from popsregression._ellipse_regression import _bernoulli_kl

    X, y = _quartic_data(80)
    model = _pac(pac_inequality="linear", pac_log_scale_std=1.0).fit(X, y)
    cert = model.certificate_
    assert model.certificate_status_ == "pac_bound"
    assert len(cert.folds) == 2
    xi = model.failure_probability / 2
    n_lambda = 11
    for fold in cert.folds:
        assert fold.n_units == 40
        assert_allclose(
            fold.raw,
            fold.empirical + fold.monte_carlo + fold.complexity + fold.concentration,
        )
        assert_allclose(fold.complexity, (fold.kl + np.log(n_lambda / xi)) / fold.lam)
        range_ = cert.loss_upper - cert.loss_lower
        assert_allclose(fold.concentration, fold.lam * range_**2 / (8 * fold.n_units))
        assert fold.kl >= 0
        assert fold.mixture_empirical <= fold.empirical + 1e-12
        assert cert.loss_lower <= fold.empirical <= cert.loss_upper
        assert fold.moment_constant == 1.0
        # Stored-mixture step: kl(q_raw || q_stored) = log(1 / xi_mix) / K.
        assert fold.n_stored == model.n_hyper_samples
        assert fold.stored_mixture >= fold.raw
        q = (min(fold.raw, cert.loss_upper) - cert.loss_lower) / range_
        p = (fold.stored_mixture - cert.loss_lower) / range_
        if p < 1.0 - 1e-9:
            assert_allclose(
                _bernoulli_kl(q, p), np.log(2 / 0.01) / fold.n_stored, rtol=1e-6
            )
    assert_allclose(cert.continuous_bound, np.mean([f.raw for f in cert.folds]))
    assert_allclose(cert.raw_bound, np.mean([f.stored_mixture for f in cert.folds]))
    assert cert.raw_bound >= cert.continuous_bound
    assert_allclose(cert.failure_probability, 0.07)
    assert cert.n_units == 80
    assert model.bound_ == cert.raw_bound
    assert cert.is_nonvacuous
    assert cert.capped_bound == min(cert.raw_bound, cert.trivial_bound)


def test_maurer_constant_is_exact_and_below_two_sqrt_n():
    from popsregression._ellipse_regression import _maurer_xi

    assert_allclose(_maurer_xi(1), 2.0)
    assert_allclose(_maurer_xi(2), 2.5)
    for n in (1, 2, 3, 5, 8, 20, 100, 1000):
        k = np.arange(n + 1)
        direct = np.sum(binom.pmf(k, n, k / n))
        assert_allclose(_maurer_xi(n), direct, rtol=1e-10)
        assert np.sqrt(n) - 1e-12 <= _maurer_xi(n) <= 2.0 * np.sqrt(n) + 1e-12


def test_pac_kl_inequality_and_prior_grid():
    from popsregression._ellipse_regression import _bernoulli_kl, _maurer_xi

    X, y = _quartic_data(80)
    grid = (0.25, 1.0)
    model = _pac(pac_log_scale_std=grid).fit(X, y)
    cert = model.certificate_
    R = cert.loss_upper - cert.loss_lower
    xi = model.failure_probability / 2
    for fold in cert.folds:
        assert fold.inequality == "kl"
        assert fold.prior_std in grid
        assert fold.concentration == 0.0
        q = (fold.empirical + fold.monte_carlo - cert.loss_lower) / R
        p = (fold.raw - cert.loss_lower) / R
        xi_n = _maurer_xi(fold.n_units)
        assert_allclose(fold.moment_constant, xi_n)
        budget = (fold.kl + np.log(len(grid) * xi_n / xi)) / fold.n_units
        assert p >= q
        assert_allclose(_bernoulli_kl(q, p), budget, rtol=1e-6)
    # The linear form at the same data is valid too, and typically looser.
    linear = _pac(pac_log_scale_std=grid, pac_inequality="linear").fit(X, y)
    assert linear.certificate_.raw_bound > cert.loss_lower


def test_pac_loss_range_uses_the_width_floor():
    X, y = _quartic_data(40)
    model = _pac(min_half_width=0.2).fit(X, y)
    cert = model.certificate_
    assert cert.min_half_width == 0.2
    for _, kernel, draws in model.components_:
        _, half = kernel.pushforward(kernel.design(X), draws)
        assert np.all(half >= 0.2 - 1e-12)
    assert_allclose(cert.trivial_bound, np.log(8.0 / 0.02))


def test_pac_single_fold_and_lambda_grid():
    X, y = _quartic_data(60)
    model = _pac(
        cross_fit=False,
        lambda_fractions=[0.25],
        pac_inequality="linear",
        pac_log_scale_std=1.0,
    ).fit(X, y)
    (fold,) = model.certificate_.folds
    assert_allclose(fold.lam, 0.25 * 30)
    assert_allclose(fold.complexity, (fold.kl + np.log(1 / 0.05)) / fold.lam)
    assert len(model.components_) == 1


def test_pac_is_deterministic():
    X, y = _quartic_data(40)
    a = _pac().fit(X, y)
    b = _pac().fit(X, y)
    assert a.bound_ == b.bound_
    assert_allclose(a.predict(X), b.predict(X))


def test_pac_groups_count_independent_units():
    X, y = _quartic_data(40)
    groups = np.repeat(np.arange(20), 2)
    model = _pac().fit(X, y, groups=groups)
    assert [f.n_units for f in model.certificate_.folds] == [10, 10]
    # Duplicating every record inside its unit must not increase N_1.
    Xd = np.repeat(X, 2, axis=0)
    yd = np.repeat(y, 2)
    dup = _pac().fit(Xd, yd, groups=np.repeat(np.arange(40), 2))
    assert [f.n_units for f in dup.certificate_.folds] == [20, 20]
    rows = _pac().fit(Xd, yd)
    assert [f.n_units for f in rows.certificate_.folds] == [40, 40]


def test_pac_pilot_kernel_is_frozen_before_certification():
    X, y = _quartic_data(40)
    model = _pac(cross_fit=False).fit(X, y)
    _, kernel, _ = model.components_[0]
    mean, var = model.hyperposteriors_[0]
    # The center is frozen by default: its shifts keep zero mean and variance.
    n_dim = kernel.n_dim
    assert np.all(mean[:n_dim] == 0.0)
    assert np.all(var[:n_dim] == 0.0)
    assert np.all(var[n_dim:] > 0.0)
    assert np.all(var[n_dim:] <= max(model.pac_log_scale_std) ** 2 + 1e-12)


def test_axis_kernel_reproduces_the_pilot_ellipsoid_and_gradients():
    X, y = _quartic_data(40)
    model = _pac(cross_fit=False, optimize_center=True).fit(X, y)
    _, kernel, _ = model.components_[0]
    engine = kernel.engine
    design = kernel.design(X)
    mean, half = kernel.pushforward(design, kernel.psi_hat[None, :])
    ref_mean, ref_max, _ = engine.predict(X, return_bounds=True)
    assert_allclose(mean[0], ref_mean, atol=1e-8)
    assert_allclose(half[0], ref_max - ref_mean, rtol=1e-6, atol=1e-8)
    rng = np.random.RandomState(0)
    psi = 0.3 * rng.randn(2 * kernel.n_dim)
    w = rng.uniform(0.5, 1.5, y.size)
    prec = rng.uniform(0.1, 1.0, psi.size)
    for rho in (1e-1, 1e-3):
        _, grad = kernel.objective(psi, design, y, w, rho, prec)
        eps = 1e-6
        fd = np.array(
            [
                (
                    kernel.objective(psi + eps * e, design, y, w, rho, prec)[0]
                    - kernel.objective(psi - eps * e, design, y, w, rho, prec)[0]
                )
                / (2 * eps)
                for e in np.eye(psi.size)
            ]
        )
        assert_allclose(grad, fd, rtol=1e-4, atol=1e-6)
        hd = kernel.hess_diag(psi, design, y, w, rho)
        fd2 = np.array(
            [
                (
                    kernel.objective(psi + eps * e, design, y, w, rho, 0 * prec)[1]
                    - kernel.objective(psi - eps * e, design, y, w, rho, 0 * prec)[1]
                )[i]
                / (2 * eps)
                for i, e in enumerate(np.eye(psi.size))
            ]
        )
        assert_allclose(hd, fd2, rtol=1e-3, atol=1e-5)


def test_pac_bound_holds_on_an_exactly_summable_population():
    """Integration audit: the bound should exceed the exact population risk.

    Inputs are uniform on a finite grid and outputs deterministic, so the
    population floor-contaminated risk of the hierarchical predictive is an
    exact finite sum (up to a Monte Carlo average over the Gaussian
    hyperposterior, evaluated with many draws). With the default 6% failure
    budget no violation is expected in a handful of draws; this is a sanity
    audit, not a measurement of the failure rate.
    """
    grid = np.linspace(-1.0, 1.0, 41)
    X_pop = np.vander(grid, 5, increasing=True)
    y_pop = np.sin(3.0 * grid) + 0.5 * grid
    beta, R = 0.02, Y_BOUNDS[1] - Y_BOUNDS[0]
    violations = 0
    n_trials = 6
    for seed in range(n_trials):
        rng = np.random.RandomState(100 + seed)
        idx = rng.randint(grid.size, size=40)
        model = _pac(random_state=seed).fit(X_pop[idx], y_pop[idx])
        risks = []
        for (weight, kernel, _), (mean, var) in zip(
            model.components_, model.hyperposteriors_
        ):
            draws = mean + np.sqrt(var) * rng.randn(2000, mean.size)
            mu, half = kernel.pushforward(kernel.design(X_pop), draws)
            log_ball = projected_ball_logpdf(
                y_pop[None, :] - mu, half, kernel.ball_dim
            )
            loss = -np.logaddexp(np.log1p(-beta) + log_ball, np.log(beta / R))
            risks.append(weight * loss.mean())  # E_{pi_H} G(Psi), exact over x
        population_H = float(np.sum(risks))
        violations += population_H > model.certificate_.continuous_bound
        # The returned stored mixture, exactly over the population grid.
        log_p = model.predict_logpdf(X_pop, y_pop)
        stored = -np.logaddexp(np.log1p(-beta) + log_p, np.log(beta / R)).mean()
        violations += stored > model.bound_
    assert violations <= binom.ppf(0.999, 2 * n_trials, 0.07)
