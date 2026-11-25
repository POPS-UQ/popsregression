"""Tests of the PACm and PAC^2_T objectives against their sources."""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from comparisons.low_noise_objectives import (
    _M_OFFSET,
    LowNoiseObjective,
    _GaussianFamily,
    taylor_weight,
)
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.optimize import check_grad


def _data(n, seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, n)
    X = np.column_stack([np.ones(n), x, x**2])
    return X, np.sin(2.5 * x)


def _prepared(objective, n_samples=4, fit_sigma=False, sigma=0.5, n_groups=3):
    X, y = _data(20, seed=1)
    model = LowNoiseObjective(
        objective,
        sigma=sigma,
        n_samples=n_samples,
        n_groups=n_groups,
        fit_sigma=fit_sigma,
        max_iter=1,
    )
    Z, ys = model._standardize_fit(X, y)
    rng = np.random.RandomState(3)
    p = Z.shape[1]
    if objective == "pac2t_ensemble":
        eps = None
        params = [rng.randn(p * n_samples)]
    else:
        model._family = _GaussianFamily(p)
        if objective in ("pac2", "pac2t"):
            eps = rng.randn(p, 2, n_groups * n_samples)
        elif objective == "elbo" or n_samples == 1:
            eps = None
        else:
            eps = rng.randn(p, n_groups, n_samples)
        params = [
            np.concatenate(
                [
                    rng.randn(p),
                    -1 + 0.3 * rng.randn(p),
                    0.2 * rng.randn(p * (p - 1) // 2),
                ]
            )
        ]
    if fit_sigma:
        params.append(np.array([np.log(sigma)]))
    return model, Z, ys, eps, np.concatenate(params)


@pytest.mark.parametrize(
    "objective", ["pacm", "elbo", "pac2", "pac2t", "pac2t_ensemble"]
)
@pytest.mark.parametrize("fit_sigma", [False, True])
def test_analytic_gradients(objective, fit_sigma):
    model, Z, ys, eps, params = _prepared(objective, fit_sigma=fit_sigma)
    err = check_grad(
        lambda v: model._objective(v, Z, ys, eps)[0],
        lambda v: model._objective(v, Z, ys, eps)[1],
        params,
        epsilon=1e-6,
    )
    scale = max(1.0, np.linalg.norm(model._objective(params, Z, ys, eps)[1]))
    assert err / scale < 1e-5


def test_pacm_matches_the_multisample_identity():
    """Eq. 13 of Morningstar et al.: -log mean_j p_j, in the r_* form of
    the paper's Appendix (pacm-small-noise), averaged over the draw groups."""
    model, Z, ys, eps, params = _prepared("pacm", n_samples=6, n_groups=3)
    value, _ = model._objective(params, Z, ys, eps)
    mu, L = model._family.unpack(params)
    sigma = model.sigma * model._sigma_unit
    kl = model._family.kl(mu, L, model.prior_scale**2)[0]
    per_group = []
    for k in range(3):
        Theta = mu[:, None] + L @ eps[:, k, :]
        R = ys[:, None] - Z @ Theta
        r_star2 = (R**2).min(axis=1)
        m = eps.shape[2]
        per_group.append(
            0.5 * np.log(2 * np.pi * sigma**2)
            + r_star2 / (2 * sigma**2)
            + np.log(m)
            - np.log(
                np.sum(np.exp(-(R**2 - r_star2[:, None]) / (2 * sigma**2)), axis=1)
            )
        )
    assert value == pytest.approx(np.mean(per_group) + kl / Z.shape[0], rel=1e-10)

    # m = 1 is the ELBO, evaluated in closed form:
    # E_q[(y - z.theta)^2] = r^2 + ||L^T z||^2.
    model1, Z, ys, eps1, params1 = _prepared("elbo")
    assert eps1 is None
    value1, _ = model1._objective(params1, Z, ys, eps1)
    mu, L = model1._family.unpack(params1)
    r = ys - Z @ mu
    spread = np.sum((Z @ L) ** 2, axis=1)
    expected = np.mean(
        0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * (r * r + spread) / sigma**2
    )
    kl = model1._family.kl(mu, L, model1.prior_scale**2)[0]
    assert value1 == pytest.approx(expected + kl / Z.shape[0], rel=1e-10)
    # PACm with m = 1 takes the same closed-form path.
    modelp, Z, ys, epsp, paramsp = _prepared("pacm", n_samples=1)
    assert epsp is None
    assert modelp._objective(paramsp, Z, ys, None)[0] == pytest.approx(value1)
    # The sample-average approximation of E_q[-ln p] converges to the
    # closed form as the number of draws grows.
    rng = np.random.RandomState(0)
    approx = []
    for K in (4, 4096):
        e = rng.randn(Z.shape[1], K)
        Theta = mu[:, None] + L @ e
        a, _ = model1._loglik(Z, ys, Theta, sigma)
        approx.append(-a.mean())
    assert abs(approx[1] - expected) < abs(approx[0] - expected)
    assert approx[1] == pytest.approx(expected, rel=0.05)


def test_pac2t_variance_term_matches_source():
    """Eq. 5 and the Appendix C stable form of Masegosa (2020): V = (p^2 -
    p p') / exp(2M) and the Taylor weight h(alpha); h -> 1/2 as alpha -> 0-,
    and pac2 (Eq. 4, weight 1 / (2 max p^2)) uses h = 1/2, as the published
    code does (0.5 * var with hmax = 1)."""
    model, Z, ys, eps, params = _prepared("pac2t", n_samples=5, n_groups=2)
    value, _ = model._objective(params, Z, ys, eps)
    mu, L = model._family.unpack(params)
    Theta = mu[:, None] + L @ eps.reshape(Z.shape[1], -1)
    a, _ = model._loglik(Z, ys, Theta, model.sigma * model._sigma_unit)
    A, B = a[:, : eps.shape[2]], a[:, eps.shape[2] :]
    M = np.maximum(A, B) + _M_OFFSET
    p_a, p_b, p_max = np.exp(A), np.exp(B), np.exp(M)
    V = (p_a**2 - p_a * p_b) / p_max**2
    alpha = np.log(np.exp(A - M) + np.exp(B - M)) - np.log(2.0)
    h = alpha / (1 - np.exp(alpha)) ** 2 + 1 / (np.exp(alpha) * (1 - np.exp(alpha)))
    kl = model._family.kl(mu, L, model.prior_scale**2)[0]
    assert value == pytest.approx(np.mean(-A - h * V) + kl / Z.shape[0], rel=1e-10)
    assert np.all(alpha <= -_M_OFFSET + 1e-12)
    assert taylor_weight(-1e-4) == pytest.approx(0.5, abs=1e-3)
    assert taylor_weight(-2.0) > 0
    h0, dh0 = taylor_weight(-0.7, derivative=True)
    fd = (taylor_weight(-0.7 + 1e-6) - taylor_weight(-0.7 - 1e-6)) / 2e-6
    assert dh0 == pytest.approx(fd, rel=1e-6)
    model2, Z, ys, eps, params = _prepared("pac2", n_samples=5, n_groups=2)
    value2, _ = model2._objective(params, Z, ys, eps)
    assert value2 == pytest.approx(np.mean(-A - 0.5 * V) + kl / Z.shape[0], rel=1e-10)
    # The variance correction only lowers the first-order objective, and the
    # tighter weight h >= 1/2 lowers it at least as much.
    assert value2 <= np.mean(-A) + kl / Z.shape[0] + 1e-12
    assert value <= value2 + 1e-12
    assert np.all(h >= 0.5 - 1e-12)


def test_ensemble_with_one_particle_is_map():
    model, Z, ys, eps, params = _prepared("pac2t_ensemble", n_samples=1)
    value, _ = model._objective(params, Z, ys, eps)
    theta = params[: Z.shape[1]]
    a, _ = model._loglik(Z, ys, theta[:, None], model.sigma * model._sigma_unit)
    prior = 0.5 * theta @ theta / model.prior_scale**2 / Z.shape[0]
    assert value == pytest.approx(-a.mean() + prior, rel=1e-10)


@pytest.mark.parametrize("objective", ["pacm", "pac2t", "pac2t_ensemble"])
@pytest.mark.parametrize("include_sigma", [False, True])
def test_fit_predict_and_bookkeeping(objective, include_sigma):
    X, y = _data(40, seed=2)
    model = LowNoiseObjective(objective, sigma=0.3, n_samples=8).fit(X, y)
    assert model.converged_ or model.n_iter_ > 0
    assert model.n_nonfinite_ == 0 and model.n_evaluations_ > 0
    assert model.fit_time_ > 0 and np.isfinite(model.objective_value_)
    # sigma is in units of the training-target standard deviation.
    assert model.sigma_ == pytest.approx(0.3 * y.std())
    discrete = objective == "pac2t_ensemble" and not include_sigma
    if discrete:
        with pytest.raises(ValueError, match="no density"):
            model.predict_logpdf(X, y)
        lo, hi = model.predict_interval(X[:5], level=0.5)
        loc, _ = model._components(X[:5])
        assert_allclose(lo, np.quantile(loc, 0.25, axis=1))
        assert_allclose(hi, np.quantile(loc, 0.75, axis=1))
        return
    log_p = model.predict_logpdf(X, y, include_sigma)
    assert np.all(np.isfinite(log_p))
    x = X[:1]
    mean, std = model.predict(x, return_std=True, include_sigma=include_sigma)
    total = quad(
        lambda t: np.exp(model.predict_logpdf(x, np.array([t]), include_sigma)[0]),
        mean[0] - 12 * std[0],
        mean[0] + 12 * std[0],
        limit=400,
    )[0]
    assert total == pytest.approx(1.0, abs=1e-6)
    lo, hi = model.predict_interval(X[:5], level=0.9, include_sigma=include_sigma)
    assert_allclose(model.predict_cdf(X[:5], lo, include_sigma), 0.05, atol=1e-8)
    assert_allclose(model.predict_cdf(X[:5], hi, include_sigma), 0.95, atol=1e-8)
    if not include_sigma and objective != "pac2t_ensemble":
        # Parameter-only: the predictive variance is x^T Sigma x alone.
        Z = model._transform(x)
        expected = model.y_scale_**2 * np.sum((Z @ model.L_) ** 2)
        assert std[0] ** 2 == pytest.approx(expected)
        _, std_full = model.predict(x, return_std=True, include_sigma=True)
        assert std_full[0] > std[0]


def test_elbo_does_not_inflate_the_posterior():
    """Closed-form m = 1: the posterior covariance stays bounded by the
    likelihood, unlike a single fixed reparameterized draw would allow."""
    X, y = _data(40, seed=5)
    model = LowNoiseObjective("elbo", sigma=1.0).fit(X, y)
    assert model.converged_
    assert np.all(np.diag(model.L_) < 1.0)


def test_fitted_sigma_stays_positive_on_deterministic_data():
    X, y = _data(40, seed=3)
    model = LowNoiseObjective("pacm", sigma=0.3, n_samples=8, fit_sigma=True).fit(X, y)
    assert model.sigma_ > 0 and np.isfinite(model.sigma_)
    assert model.sigma_ != pytest.approx(0.3 * y.std())


def test_narrow_likelihood_raises_the_objective():
    """The r_*^2 / (2 sigma^2) term: at fixed m the optimized objective
    grows as the likelihood narrows on misspecified deterministic data."""
    X, y = _data(40, seed=4)
    values = [
        LowNoiseObjective("pacm", sigma=s, n_samples=4).fit(X, y).objective_value_
        for s in (0.3, 0.03, 0.003)
    ]
    assert values[0] < values[1] < values[2]


def test_invalid_objective():
    with pytest.raises(ValueError, match="objective"):
        LowNoiseObjective("nope")


@pytest.mark.parametrize("objective", ["pacm", "pac2t", "elbo"])
def test_diagnostics_are_recorded(objective):
    X, y = _data(30, seed=6)
    model = LowNoiseObjective(objective, sigma=0.3, n_samples=4, max_iter=5000)
    model.fit(X, y)
    assert model.max_iter_ == 5000 and model.max_fun_ == 10000
    assert model.n_fev_ >= model.n_iter_ > 0
    assert model.n_fev_ == model.n_evaluations_
    assert model.termination_status_ in (0, 1, 2)
    assert model.termination_ in (
        "converged",
        "max_iter",
        "max_fun",
        "line_search",
        "other",
    )
    assert model.converged_ == (model.termination_ == "converged")
    assert isinstance(model.termination_message_, str)
    # The recorded split adds up to the returned objective.
    total = model.objective_data_ + model.objective_kl_
    assert total == pytest.approx(model.objective_value_, rel=1e-10)
    assert model.objective_kl_ > 0
    if objective == "pac2t":
        assert model.objective_variance_ > 0
    else:
        assert model.objective_variance_ == 0.0


def test_iteration_and_evaluation_limits_are_distinguished():
    X, y = _data(30, seed=6)
    by_iter = LowNoiseObjective("pacm", sigma=0.01, n_samples=4, max_iter=3).fit(X, y)
    assert by_iter.termination_ == "max_iter" and not by_iter.converged_
    assert by_iter.n_iter_ == 3
    by_fun = LowNoiseObjective(
        "pacm", sigma=0.01, n_samples=4, max_iter=1000, max_fun=3
    ).fit(X, y)
    assert by_fun.termination_ == "max_fun" and not by_fun.converged_
    # The unconverged iterate is kept and still predicts.
    assert np.all(np.isfinite(by_fun.params_))
    assert np.all(np.isfinite(by_fun.predict(X)))


@pytest.mark.parametrize("objective", ["pacm", "pac2t"])
def test_fresh_draws_reproduce_the_training_objective(objective):
    """With many groups the objective on fresh draws matches the one
    optimized on the fixed draws (the fixed draws are not special)."""
    X, y = _data(40, seed=7)
    model = LowNoiseObjective(
        objective, sigma=0.3, n_samples=8, n_groups=512, random_state=0
    ).fit(X, y)
    data, kl, total = model.evaluate_objective(X, y, n_groups=4096, random_state=99)
    assert kl == pytest.approx(model.objective_kl_, rel=1e-12)
    assert total == pytest.approx(data + kl)
    assert total == pytest.approx(model.objective_value_, rel=0.02)
    # Same draws as the fit reproduce it exactly.
    model2 = LowNoiseObjective(objective, sigma=0.3, n_samples=8, n_groups=4)
    model2.fit(X, y)
    same = model2._objective(
        model2.params_, *model2._standardize_fit(X, y), model2._eps
    )
    assert same[0] == pytest.approx(model2.objective_value_, rel=1e-12)


def test_pacm_m1_fresh_draws_match_closed_form():
    X, y = _data(40, seed=8)
    model = LowNoiseObjective("pacm", sigma=0.3, n_samples=1).fit(X, y)
    _, _, total = model.evaluate_objective(X, y, n_groups=20000, random_state=1)
    assert total == pytest.approx(model.objective_value_, rel=0.01)


def test_fixed_m_objective_grows_like_inverse_variance():
    """At fixed m on data no linear model fits exactly, the optimized PACm
    objective grows like C_m / sigma^2 once sigma is small: shrinking sigma
    by 10 multiplies it by about 100 (the log-sigma term is negligible).
    A larger m lowers C_m but does not make it zero."""
    X, y = _data(20, seed=9)
    assert np.min(np.abs(y - X @ np.linalg.lstsq(X, y, rcond=None)[0])) > 0
    coefficients = []
    for m in (1, 4, 16):
        sigmas = (1e-4, 1e-5)
        values = [
            LowNoiseObjective("pacm", sigma=s, n_samples=m, n_groups=8)
            .fit(X, y)
            .objective_value_
            for s in sigmas
        ]
        slope = np.log10(values[1] / values[0])
        assert 1.9 < slope < 2.1, (m, values)
        coefficients.append(values[1] * sigmas[1] ** 2)
    assert coefficients[0] > coefficients[1] > coefficients[2] > 0


def test_termination_reasons_from_scipy_messages():
    from types import SimpleNamespace

    from comparisons.low_noise_objectives import _termination_reason

    def res(success, message):
        return SimpleNamespace(success=success, message=message)

    assert _termination_reason(res(True, "CONVERGENCE: ...")) == "converged"
    cases = {
        "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT": "max_iter",
        "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT": "max_fun",
        "ABNORMAL: ": "line_search",
        "ABNORMAL_TERMINATION_IN_LNSRCH": "line_search",
        "something else": "other",
    }
    for message, reason in cases.items():
        assert _termination_reason(res(False, message)) == reason
