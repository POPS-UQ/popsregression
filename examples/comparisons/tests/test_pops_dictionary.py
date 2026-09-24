"""Tests for the pilot-split finite-dictionary PAC certificate."""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np
import pytest
from comparisons.pops_dictionary import (
    PACCertificateResult,
    POPSPACCertificate,
    WeightRuleCertificate,
    categorical_kl,
    categorical_pac_bound,
    gibbs_weights,
)
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.stats import beta as beta_dist
from scipy.stats import binom

from popsregression import POPSEllipseRegression
from popsregression._projected_ball import (
    floor_contaminated_logpdf,
    floor_contaminated_loss_bounds,
    log_norm_constant,
    projected_ball_logpdf,
)

Y_BOUNDS = (-2.0, 2.0)


def _features(x):
    """Quadratic feature map with a constant column (index 0)."""
    return np.column_stack([np.ones_like(x), x, x**2])


def _truth(x):
    return np.sin(3.0 * x) + 0.3 * x


def _draw(rng, n):
    x = rng.uniform(-1.0, 1.0, n)
    return _features(x), _truth(x)


def _certifier(**kwargs):
    options = dict(
        y_bounds=Y_BOUNDS,
        min_half_width=0.05,
        constant_feature=0,
        candidate_scales=np.geomspace(0.5, 4.0, 9),
        lambdas=np.geomspace(0.5, 64.0, 8),
        random_state=0,
    )
    options.update(kwargs)
    return POPSPACCertificate(**options)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.RandomState(0)
    X0, y0 = _draw(rng, 40)
    X1, y1 = _draw(rng, 60)
    certifier = _certifier().fit_pilot(X0, y0)
    result = certifier.certify(X1, y1)
    return certifier, result, (X1, y1)


# --- 1. Density normalization ---


@pytest.mark.parametrize("n_dim", [1, 3, 8])
@pytest.mark.parametrize("mean, half_width", [(0.0, 0.7), (1.8, 0.5), (-0.5, 3.0)])
def test_floor_contaminated_density_normalizes(n_dim, mean, half_width):
    """The contaminated density integrates to one on R, including the
    cases where the ellipsoid support sticks out of the output interval."""
    beta = 0.1

    def pdf(y):
        return np.exp(
            floor_contaminated_logpdf(
                y - mean,
                half_width,
                n_dim,
                beta=beta,
                y=y,
                y_bounds=Y_BOUNDS,
                strict=False,
            )
        )

    lo = min(Y_BOUNDS[0], mean - half_width) - 0.5
    hi = max(Y_BOUNDS[1], mean + half_width) + 0.5
    points = sorted({Y_BOUNDS[0], Y_BOUNDS[1], mean - half_width, mean + half_width})
    total = quad(pdf, lo, hi, points=points, limit=400)[0]
    assert total == pytest.approx(1.0, abs=1e-7)


# --- 2. Outside-support behavior ---


def test_outside_support_exact_is_minus_inf_but_contaminated_is_finite():
    beta, half_width, n_dim = 0.05, 0.3, 4
    exact = projected_ball_logpdf(1.0, half_width, n_dim)
    assert np.isneginf(exact)
    contaminated = floor_contaminated_logpdf(
        1.0, half_width, n_dim, beta=beta, y=1.0, y_bounds=Y_BOUNDS
    )
    width = Y_BOUNDS[1] - Y_BOUNDS[0]
    assert contaminated == pytest.approx(np.log(beta / width))
    assert -contaminated == pytest.approx(np.log(width / beta))


# --- 3. Output-domain violation ---


def test_output_outside_bounds_raises_and_no_certificate_is_issued():
    rng = np.random.RandomState(1)
    X0, y0 = _draw(rng, 40)
    X1, y1 = _draw(rng, 30)
    certifier = _certifier().fit_pilot(X0, y0)
    y_bad = y1.copy()
    y_bad[3] = Y_BOUNDS[1] + 1e-3
    with pytest.raises(ValueError, match="outside the declared output interval"):
        certifier.certify(X1, y_bad)
    assert not hasattr(certifier, "result_")
    # Pilot outputs outside the declared support contradict the protocol.
    y0_bad = y0.copy()
    y0_bad[0] = Y_BOUNDS[0] - 1.0
    with pytest.raises(ValueError, match="contradicting"):
        _certifier().fit_pilot(X0, y0_bad)


# --- 4. Loss bounds and the analytic half-width floor ---


def test_loss_bounds_hold_for_random_inputs():
    rng = np.random.RandomState(2)
    n_dim, beta, a_min = 5, 0.03, 0.2
    lower, upper = floor_contaminated_loss_bounds(
        n_dim, beta=beta, y_bounds=Y_BOUNDS, min_half_width=a_min
    )
    assert lower == pytest.approx(
        -np.log(
            (1 - beta) * np.exp(log_norm_constant(n_dim)) / a_min
            + beta / (Y_BOUNDS[1] - Y_BOUNDS[0])
        )
    )
    y = rng.uniform(*Y_BOUNDS, 2000)
    mean = rng.uniform(-3, 3, 2000)
    half = a_min * np.exp(rng.uniform(0, 3, 2000))
    loss = -floor_contaminated_logpdf(
        y - mean, half, n_dim, beta=beta, y=y, y_bounds=Y_BOUNDS
    )
    assert np.all(loss >= lower - 1e-12) and np.all(loss <= upper + 1e-12)
    # The lower bound is attained at the center of the narrowest candidate.
    tight = -floor_contaminated_logpdf(
        0.0, a_min, n_dim, beta=beta, y=0.0, y_bounds=Y_BOUNDS
    )
    assert tight == pytest.approx(lower)


@pytest.mark.parametrize("constant_feature", [0, "intercept"])
def test_analytic_half_width_floor(constant_feature):
    rng = np.random.RandomState(3)
    X0, y0 = _draw(rng, 40)
    certifier = _certifier(constant_feature=constant_feature, min_half_width=0.3)
    certifier.fit_pilot(X0, y0)
    # Far outside the pilot inputs the floor still holds analytically.
    X_far = _features(rng.uniform(-50.0, 50.0, 500))
    _, half = certifier.candidate_half_widths(X_far)
    assert np.all(half >= 0.3 - 1e-12)
    # It is exactly the requested value where the pilot width vanishes.
    scale = certifier.pilot_.candidate_scales
    _, s_pilot, s_floor = certifier._geometry(X_far)
    assert_allclose(s_floor, 0.3**2)
    assert_allclose(half**2, scale[None, :] ** 2 * s_pilot[:, None] + 0.3**2)
    assert "min_half_width" in certifier.pilot_.floor_construction
    if constant_feature == 0:
        X_bad = X_far.copy()
        X_bad[0, 0] = 2.0
        with pytest.raises(ValueError, match="outside the declared domain"):
            certifier.candidate_half_widths(X_bad)
        with pytest.raises(ValueError, match="not a nonzero constant"):
            _certifier(constant_feature=1).fit_pilot(X0, y0)


# --- 5. Exact categorical algebra ---


def test_categorical_algebra_matches_numpy(fitted):
    certifier, result, _ = fitted
    q0 = result.prior_weights
    g = result.candidate_empirical
    lam = result.selected_lambda
    logits = np.log(q0) - lam * g
    q = np.exp(logits) / np.exp(logits).sum()
    assert_allclose(result.posterior_weights, q, rtol=1e-12)
    assert result.empirical == pytest.approx(float(q @ g), rel=1e-12)
    assert result.kl == pytest.approx(float(np.sum(q * np.log(q / q0))), rel=1e-10)
    # g_hat is the plain mean of the row losses (no groups).
    assert_allclose(g, -result.row_log_density.mean(axis=0), rtol=1e-12)
    assert categorical_kl([1.0, 0.0], [0.5, 0.5]) == pytest.approx(np.log(2.0))
    assert categorical_kl([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert_allclose(gibbs_weights(q0, g, lam), q, rtol=1e-12)


# --- 6. Temperature correction ---


def test_temperature_union_correction():
    kw = dict(n_units=20, loss_range=3.0, failure_probability=0.05)
    q, g, q0 = [0.25, 0.75], [1.0, 0.5], [0.5, 0.5]
    many = categorical_pac_bound(q, g, q0, 2.0, n_temperatures=8, **kw)
    single = categorical_pac_bound(q, g, q0, 2.0, n_temperatures=1, **kw)
    kl = categorical_kl(q, q0)
    assert many["complexity_penalty"] == pytest.approx((kl + np.log(8 / 0.05)) / 2.0)
    assert single["complexity_penalty"] == pytest.approx((kl + np.log(1 / 0.05)) / 2.0)
    assert many["concentration_penalty"] == pytest.approx(2.0 * 9.0 / (8 * 20))
    assert many["raw_bound"] == pytest.approx(
        many["empirical"] + many["complexity_penalty"] + many["concentration_penalty"]
    )

    rng = np.random.RandomState(4)
    X0, y0 = _draw(rng, 40)
    X1, y1 = _draw(rng, 50)
    grid = _certifier().fit_pilot(X0, y0).certify(X1, y1)
    one = _certifier(lambdas=[grid.selected_lambda]).fit_pilot(X0, y0).certify(X1, y1)
    n_grid = grid.protocol["n_lambdas"]
    assert n_grid == 8
    assert grid.complexity_penalty == pytest.approx(
        (grid.kl + np.log(n_grid / 0.05)) / grid.selected_lambda
    )
    assert one.complexity_penalty == pytest.approx(
        (one.kl + np.log(1 / 0.05)) / one.selected_lambda
    )
    assert_allclose(one.posterior_weights, grid.posterior_weights)
    assert one.raw_bound < grid.raw_bound


# --- 7 & 8. No certification-sample refit, no candidate leakage ---


def test_pilot_state_is_immutable_across_certify():
    rng = np.random.RandomState(5)
    X0, y0 = _draw(rng, 40)
    certifier = _certifier().fit_pilot(X0, y0)
    pilot = certifier.pilot_
    digest = pilot.digest()
    snapshot = copy.deepcopy(pilot)
    for array in (pilot.whiten, pilot.center, pilot.U, pilot.candidate_scales):
        assert not array.flags.writeable

    for seed in (10, 11):
        X1, y1 = _draw(np.random.RandomState(seed), 70)
        result = certifier.certify(X1, y1)
        assert result.protocol["pilot_digest"] == digest
    assert pilot.digest() == digest
    assert certifier.pilot_ is pilot
    for name in pilot.__dataclass_fields__:
        before, after = getattr(snapshot, name), getattr(pilot, name)
        if isinstance(after, np.ndarray):
            assert after.dtype == before.dtype and after.shape == before.shape
            assert after.tobytes() == before.tobytes()
        else:
            assert after == before
    with pytest.raises(Exception):
        pilot.center[0] = 0.0
    with pytest.raises(Exception):
        pilot.candidate_scales = pilot.candidate_scales * 2


# --- 9. Permutation invariance ---


def test_permutation_invariance(fitted):
    certifier, result, (X1, y1) = fitted
    perm = np.random.RandomState(6).permutation(X1.shape[0])
    permuted = certifier.certify(X1[perm], y1[perm])
    for rule in result.rules:
        a, b = result.rules[rule], permuted.rules[rule]
        assert a.raw_bound == pytest.approx(b.raw_bound, rel=1e-10)
        assert_allclose(a.weights, b.weights, atol=1e-10)
        assert a.selected_lambda == b.selected_lambda
    assert_allclose(result.unit_losses[perm], permuted.unit_losses, rtol=1e-12)


# --- 10. Independent-unit accounting ---


def test_duplicated_records_are_rejected_and_groups_do_not_inflate_n1():
    rng = np.random.RandomState(7)
    X0, y0 = _draw(rng, 40)
    X1, y1 = _draw(rng, 30)
    certifier = _certifier().fit_pilot(X0, y0, sample_ids=np.arange(40))
    base = certifier.certify(X1, y1, sample_ids=100 + np.arange(30))

    # Duplicating records with their identifiers is rejected outright.
    X_dup, y_dup = np.vstack([X1, X1]), np.concatenate([y1, y1])
    ids_dup = np.concatenate([100 + np.arange(30)] * 2)
    with pytest.raises(ValueError, match="Repeated sample_ids"):
        certifier.certify(X_dup, y_dup, sample_ids=ids_dup)
    # Overlap with the pilot is rejected.
    with pytest.raises(ValueError, match="overlap with the pilot"):
        certifier.certify(X1, y1, sample_ids=np.arange(30))
    # Duplicates inside one group (fresh record ids, same case) leave N1 and
    # the bound unchanged: a resampled record is not a new independent unit.
    groups = np.concatenate([np.arange(30)] * 2)
    dup = certifier.certify(
        X_dup, y_dup, sample_ids=200 + np.arange(60), group_ids=groups
    )
    assert dup.n_independent_units == base.n_independent_units == 30
    assert dup.n_rows == 60
    assert dup.raw_bound == pytest.approx(base.raw_bound, rel=1e-12)
    assert_allclose(dup.unit_losses, base.unit_losses, rtol=1e-12)
    assert "N1 counts groups" in " ".join(dup.independence_assumptions["verified"])

    # Identical values from distinct genuine records are not duplicates.
    same = certifier.certify(X_dup, y_dup, sample_ids=200 + np.arange(60))
    assert same.n_independent_units == 60
    assert not base.independence_assumptions["group_ids_provided"]
    assert base.independence_assumptions["sample_ids_provided"]


def test_grouped_case_average_matches_direct_calculation():
    rng = np.random.RandomState(8)
    X0, y0 = _draw(rng, 40)
    certifier = _certifier().fit_pilot(X0, y0, group_ids=np.arange(40))
    n_cases, per_case = 12, 3
    X1, y1 = _draw(rng, n_cases * per_case)
    groups = np.repeat(1000 + np.arange(n_cases), per_case)
    with pytest.raises(ValueError, match="group_ids overlap"):
        certifier.certify(X1, y1, group_ids=np.repeat(np.arange(n_cases), per_case))
    result = certifier.certify(X1, y1, group_ids=groups)
    assert result.n_independent_units == n_cases
    assert result.n_rows == n_cases * per_case

    loss = -certifier.candidate_log_density(X1, y1)
    case_loss = loss.reshape(n_cases, per_case, -1).mean(axis=1)
    assert_allclose(result.unit_losses, case_loss, rtol=1e-12)
    g_hat = case_loss.mean(axis=0)
    assert_allclose(result.candidate_empirical, g_hat, rtol=1e-12)
    pilot = certifier.pilot_
    lam = result.selected_lambda
    q = gibbs_weights(pilot.prior_weights, g_hat, lam)
    expected = (
        q @ g_hat
        + (categorical_kl(q, pilot.prior_weights) + np.log(8 / 0.05)) / lam
        + lam * result.loss_range**2 / (8 * n_cases)
    )
    assert result.raw_bound == pytest.approx(expected, rel=1e-12)
    # The mixture objective uses the same case averaging.
    log_mix = np.log(np.exp(-loss) @ q)
    mix_case = -log_mix.reshape(n_cases, per_case).mean(axis=1)
    assert result.mixture_empirical == pytest.approx(mix_case.mean(), rel=1e-12)
    assert_allclose(result.rules["gibbs"].unit_mixture_losses, mix_case, rtol=1e-12)


# --- 11. Vacuity reporting ---


def test_vacuous_raw_bound_is_preserved():
    rng = np.random.RandomState(9)
    X0, y0 = _draw(rng, 40)
    X1, y1 = _draw(rng, 3)
    result = _certifier().fit_pilot(X0, y0).certify(X1, y1)
    assert result.raw_bound > result.loss_upper
    assert result.capped_bound == result.loss_upper
    assert not result.is_nonvacuous
    assert np.isfinite(result.raw_bound)
    assert result.certified  # valid, merely vacuous


# --- 12. Statistical theorem audit ---


def test_theorem_audit_on_finite_support_population():
    """Exact population risk of the certified mixture on a finite-support
    population, over repeated independent certification draws.

    The bound is conditional on the pilot; the violation frequency over
    the certification draws is compared with the nominal xi through a
    predeclared binomial test. Integration test, not a proof.
    """
    n_draws, n_cert, xi = 300, 25, 0.1
    support_x = np.array([-1.0, -0.6, -0.2, 0.15, 0.5, 0.9])
    prob = np.array([0.1, 0.2, 0.25, 0.2, 0.15, 0.1])
    X_support = _features(support_x)
    y_support = _truth(support_x)

    rng = np.random.RandomState(12)
    idx0 = rng.choice(support_x.size, size=40, p=prob)
    certifier = _certifier(failure_probability=xi).fit_pilot(
        X_support[idx0], y_support[idx0]
    )
    # Exact population risk of every candidate and of any mixture.
    log_p_support = certifier.candidate_log_density(X_support, y_support)

    violations = 0
    slack = 0.0
    for _ in range(n_draws):
        idx1 = rng.choice(support_x.size, size=n_cert, p=prob)
        result = certifier.certify(X_support[idx1], y_support[idx1])
        for rule in result.rules.values():
            q = rule.weights
            population_candidate = -(prob @ log_p_support)
            population_mixture = -float(prob @ np.log(np.exp(log_p_support) @ q))
            # First inequality of the corollary (Jensen) holds exactly.
            assert population_mixture <= float(q @ population_candidate) + 1e-12
            slack = max(slack, population_mixture - rule.raw_bound)
        # Audit the reported (Gibbs) certificate.
        q = result.posterior_weights
        population_mixture = -float(prob @ np.log(np.exp(log_p_support) @ q))
        violations += population_mixture > result.raw_bound
    # Predeclared one-sided binomial test at level 1e-3.
    assert violations <= binom.ppf(0.999, n_draws, xi)
    lower, upper = (
        beta_dist.ppf(0.0005, violations, n_draws - violations + 1),
        beta_dist.ppf(0.9995, violations + 1, n_draws - violations),
    )
    assert np.nan_to_num(lower) <= xi
    assert upper <= 1.0


# --- 14 (certificate side), 15. Jensen term ---


def test_jensen_gap_is_nonnegative_and_reported(fitted):
    _, result, _ = fitted
    for rule in result.rules.values():
        assert rule.mixture_empirical <= rule.empirical + 1e-12
        assert rule.jensen_gap == pytest.approx(rule.empirical - rule.mixture_empirical)
        # The certificate uses the average candidate loss, never the
        # mixture loss.
        assert rule.raw_bound == pytest.approx(
            rule.empirical + rule.complexity_penalty + rule.concentration_penalty
        )
    # Genuine mixtures have a strictly positive gap on this dictionary.
    assert result.rules["fixed_prior"].jensen_gap > 1e-3
    assert result.rules["gibbs"].jensen_gap > 0.0
    single = result.rules["single_best"]
    assert single.jensen_gap == pytest.approx(0.0, abs=1e-12)
    assert single.weights.sum() == 1.0 and np.count_nonzero(single.weights) == 1
    assert single.kl == pytest.approx(np.log(result.prior_weights.size))


# --- 16. Arbitrary-weight certification ---


def test_arbitrary_weights_share_the_certificate(fitted):
    _, result, _ = fitted
    lambdas = result.protocol["lambdas"]
    n_grid = lambdas.size
    gibbs = result.rules["gibbs"]
    for i, lam in enumerate(lambdas):
        gibbs_parts = result.bound_for_weights(gibbs.weights_by_lambda[i], lam)
        assert gibbs_parts["raw_bound"] == pytest.approx(gibbs.bounds_by_lambda[i])
        for name, rule in result.rules.items():
            parts = result.bound_for_weights(rule.weights_by_lambda[i], lam)
            assert parts["raw_bound"] == pytest.approx(rule.bounds_by_lambda[i])
            assert parts["complexity_penalty"] == pytest.approx(
                (parts["kl"] + np.log(n_grid / 0.05)) / lam
            )
            # Gibbs minimizes the right side at every fixed temperature.
            assert gibbs.bounds_by_lambda[i] <= rule.bounds_by_lambda[i] + 1e-10
    # An arbitrary simplex vector is certified on the same event.
    q = np.full(result.prior_weights.size, 1.0)
    q[::2] += 1.0
    q /= q.sum()
    parts = result.bound_for_weights(q)
    assert parts["temperature"] in lambdas
    assert parts["raw_bound"] >= result.raw_bound - 1e-10
    assert parts["jensen_gap"] >= -1e-12
    with pytest.raises(ValueError, match="predeclared grid"):
        result.bound_for_weights(q, 0.123456)
    assert result.rules["stacking"].solver_status["success"]
    assert isinstance(result.rules["stacking"], WeightRuleCertificate)
    assert isinstance(result, PACCertificateResult)


# --- 17. Fluctuation safeguards ---


def test_concentration_term_never_vanishes_for_deterministic_data():
    for strength in (0.0, 0.05, 0.5):
        rng = np.random.RandomState(13)
        x0, x1 = rng.uniform(-1, 1, 40), rng.uniform(-1, 1, 200)
        y0 = 0.3 * x0 + strength * np.sin(3 * x0)
        y1 = 0.3 * x1 + strength * np.sin(3 * x1)
        result = _certifier().fit_pilot(_features(x0), y0).certify(_features(x1), y1)
        lam = result.selected_lambda
        for rule in result.rules.values():
            assert rule.concentration_penalty > 0.0
            # Hoeffding only: lambda R^2 / (8 N1), no variance substitution.
            assert rule.concentration_penalty == pytest.approx(
                rule.selected_lambda * result.loss_range**2 / (8 * 200)
            )
        assert result.concentration_penalty == pytest.approx(
            lam * result.loss_range**2 / (8 * 200)
        )


def test_boundary_outside_support_and_rare_region_losses():
    """A predeclared rare region carries the misspecification; a draw
    missing it and a draw containing it are both certified, and the floor
    contributions are recorded rather than hidden."""
    rng = np.random.RandomState(14)

    def rare_truth(x):
        return 0.5 * x + 1.4 * np.exp(-((x - 0.9) ** 2) / 0.002)

    x0 = rng.uniform(-1, 0.8, 40)  # the pilot never sees the region
    certifier = _certifier(min_half_width=0.02).fit_pilot(_features(x0), rare_truth(x0))
    x_easy = rng.uniform(-1, 0.8, 100)
    x_hard = np.concatenate([x_easy[:-5], np.full(5, 0.9)])
    easy = certifier.certify(_features(x_easy), rare_truth(x_easy))
    hard = certifier.certify(_features(x_hard), rare_truth(x_hard))

    n_out_easy = easy.loss_summary["unmodified_row_loss"]["n_nonfinite"]
    n_out_hard = hard.loss_summary["unmodified_row_loss"]["n_nonfinite"]
    assert np.all(n_out_hard >= n_out_easy)
    assert n_out_hard.max() >= 5
    assert np.all(hard.fraction_outside_support >= easy.fraction_outside_support)
    assert np.all(hard.floor_fraction >= easy.floor_fraction)
    assert hard.rules["gibbs"].floor_fraction > easy.rules["gibbs"].floor_fraction
    # Every contaminated loss is finite and inside the analytic interval.
    for res in (easy, hard):
        assert np.all(np.isfinite(res.unit_losses))
        assert np.all(res.unit_losses >= res.loss_lower - 1e-12)
        assert np.all(res.unit_losses <= res.loss_upper + 1e-12)
        assert res.certified
    # The unmodified loss records its non-finite values instead of a cap.
    assert np.isneginf(hard.row_log_density_exact).any()
    assert np.all(np.isfinite(hard.row_log_density))
    # Near-boundary losses are large but bounded by the floor.
    pilot = certifier.pilot_
    mean, half = certifier.candidate_half_widths(_features(np.array([0.0])))
    y_edge = mean + half[:, 0] * (1 - 1e-9)
    loss_edge = -floor_contaminated_logpdf(
        y_edge - mean,
        half[:, 0],
        pilot.ball_dim,
        beta=pilot.beta,
        y=y_edge,
        y_bounds=pilot.y_bounds,
    )
    assert loss_edge <= pilot.loss_upper + 1e-12
    assert loss_edge > -floor_contaminated_logpdf(
        0.0,
        half[0, 0],
        pilot.ball_dim,
        beta=pilot.beta,
        y=mean[0],
        y_bounds=pilot.y_bounds,
    )


# --- 18. Predictive semantics ---


def test_predictive_moments_and_quantiles_include_the_floor(fitted):
    certifier, result, _ = fitted
    pilot = certifier.pilot_
    X = _features(np.array([-0.7, 0.1, 0.8]))
    q = result.posterior_weights
    mean_c, std_c = certifier.predict(X, return_std=True)
    mean_u, std_u = certifier.predict(X, return_std=True, contaminated=False)
    center, half = certifier.candidate_half_widths(X)
    beta = pilot.beta
    c0 = 0.5 * sum(Y_BOUNDS)
    # Uncontaminated moments are those of the parameter mixture.
    assert_allclose(mean_u, center)
    assert_allclose(std_u**2, (half**2 / (pilot.ball_dim + 2)) @ q)
    # The floor shifts the mean toward the interval center and widens.
    assert_allclose(mean_c, (1 - beta) * center + beta * c0)
    assert np.all(std_c > std_u)

    # Quadrature of the mixture density is one, and its mean matches.
    for i in range(X.shape[0]):
        Xi = X[i : i + 1]
        lo = min(Y_BOUNDS[0], center[i] - half[i].max()) - 0.1
        hi = max(Y_BOUNDS[1], center[i] + half[i].max()) + 0.1

        def density(y, Xi=Xi):
            log_p = floor_contaminated_logpdf(
                y - center[i],
                half[i],
                pilot.ball_dim,
                beta=beta,
                y=y,
                y_bounds=Y_BOUNDS,
                strict=False,
            )
            return float(np.exp(log_p) @ q)

        pts = sorted(set([*Y_BOUNDS, *(center[i] - half[i]), *(center[i] + half[i])]))
        total = quad(density, lo, hi, points=pts, limit=500)[0]
        first = quad(lambda y: y * density(y), lo, hi, points=pts, limit=500)[0]
        assert total == pytest.approx(1.0, abs=1e-6)
        assert first == pytest.approx(mean_c[i], abs=1e-6)
        # predict_logpdf is the log of that density inside the interval.
        y_probe = np.array([center[i]])
        assert certifier.predict_logpdf(Xi, y_probe)[0] == pytest.approx(
            np.log(density(y_probe[0]))
        )

    # Exact central quantiles: CDF at the interval ends equals the levels,
    # and they are not the Gaussian mean +/- 2 std proxy.
    level = 0.9545
    lower, upper = certifier.predict_interval(X, level=level)
    assert_allclose(certifier.predict_cdf(X, lower), 0.5 * (1 - level), atol=1e-8)
    assert_allclose(certifier.predict_cdf(X, upper), 0.5 * (1 + level), atol=1e-8)
    assert not np.allclose(lower, mean_c - 2 * std_c, atol=1e-3)
    assert not np.allclose(upper, mean_c + 2 * std_c, atol=1e-3)


@pytest.mark.parametrize("constant_feature", [0, "intercept"])
def test_parameter_draws_define_whole_fields(constant_feature):
    rng = np.random.RandomState(15)
    X0, y0 = _draw(rng, 40)
    X1, y1 = _draw(rng, 60)
    certifier = _certifier(constant_feature=constant_feature).fit_pilot(X0, y0)
    result = certifier.certify(X1, y1)
    theta, chosen = certifier.sample_parameters(4000, random_state=0)
    X = _features(np.linspace(-1, 1, 25))
    if constant_feature == "intercept":
        assert theta.shape == (X.shape[1] + 1, 4000)
        field = X @ theta[:-1] + theta[-1]
    else:
        assert theta.shape == (X.shape[1], 4000)
        field = X @ theta
    center, half = certifier.candidate_half_widths(X)
    # One draw is one parameter vector: its field lies inside the support
    # of the drawn candidate at every input simultaneously.
    resid = np.abs(field - center[:, None])
    assert np.all(resid <= half[:, chosen] * (1 + 1e-9))
    # The pushforward variance across draws matches the parameter mixture.
    _, std_u = certifier.predict(X, return_std=True, contaminated=False)
    assert_allclose(field.var(axis=1), std_u**2, rtol=0.15)
    # The candidate frequencies follow the Gibbs weights.
    freq = np.bincount(chosen, minlength=result.posterior_weights.size) / 4000
    assert_allclose(freq, result.posterior_weights, atol=0.03)


# --- One-sample estimator never certifies ---


def test_one_sample_estimator_is_diagnostic_only():
    rng = np.random.RandomState(16)
    X, y = _draw(rng, 60)
    model = POPSEllipseRegression(regularization="empirical-bayes", random_state=0)
    model.fit(X, y)
    assert model.certificate_status_ == "diagnostic_empirical_bayes"
    assert not hasattr(model, "bound_")
    assert not hasattr(model, "certificate_")


def test_certify_requires_pilot_and_validates_protocol():
    with pytest.raises(ValueError, match="fit_pilot"):
        _certifier().certify(np.ones((2, 3)), np.zeros(2))
    with pytest.raises(ValueError, match="y_bounds is required"):
        POPSPACCertificate().fit_pilot(np.ones((2, 3)), np.zeros(2))
    with pytest.raises(ValueError, match="beta"):
        _certifier(beta=1.5).fit_pilot(np.ones((2, 3)), np.zeros(2))
    with pytest.raises(ValueError, match="pilot_options"):
        _certifier(pilot_options={"pac_bayes": True}).fit_pilot(
            np.ones((2, 3)), np.zeros(2)
        )
    with pytest.raises(ValueError, match="prior_weights"):
        _certifier(prior_weights=np.zeros(9)).fit_pilot(np.ones((2, 3)), np.zeros(2))


def test_summary_is_flat_and_complete(fitted):
    _, result, _ = fitted
    summary = result.summary()
    assert summary["certified"] is True
    assert summary["target"] == "floor_contaminated_predictive_log_risk"
    for rule in ("gibbs", "fixed_prior", "single_best", "stacking"):
        assert summary[f"{rule}_raw_bound"] == result.rules[rule].raw_bound
    assert all(np.ndim(v) == 0 for v in summary.values())
    assert_array_equal(result.prior_weights, np.full(9, 1 / 9))


# --- PAC Gibbs against predictive stacking ---


def test_gibbs_against_stacking():
    """Stacking minimizes the mixture log loss; Gibbs minimizes the bound.

    On the same frozen dictionary and certification sample: (i) the stacking
    weights attain the smallest empirical mixture log loss of every rule and
    of random simplex vectors; (ii) at every temperature the Gibbs bound is
    no larger than the stacking bound; (iii) the stacking certificate uses
    the average candidate loss (its Jensen gap is not dropped); (iv) both
    predictive mixtures are proper densities that score the same held-out
    sample comparably.
    """
    rng = np.random.RandomState(17)
    x0, x1, x2 = (rng.uniform(-1, 1, n) for n in (40, 120, 400))
    truth = lambda x: np.sin(3 * x) + 0.3 * x + 0.4 * np.exp(-((x + 0.6) ** 2) / 0.02)
    certifier = _certifier().fit_pilot(_features(x0), truth(x0))
    result = certifier.certify(_features(x1), truth(x1))
    stacking, gibbs = result.rules["stacking"], result.rules["gibbs"]
    assert stacking.solver_status["success"]

    # (i) stacking is the mixture-loss minimizer on D1.
    for rule in result.rules.values():
        assert stacking.mixture_empirical <= rule.mixture_empirical + 1e-9
    for q in rng.dirichlet(np.ones(result.prior_weights.size), size=50):
        parts = result.bound_for_weights(q, stacking.selected_lambda)
        assert stacking.mixture_empirical <= parts["mixture_empirical"] + 1e-9
    # (ii) Gibbs is the bound minimizer at every temperature.
    assert np.all(gibbs.bounds_by_lambda <= stacking.bounds_by_lambda + 1e-10)
    assert gibbs.raw_bound <= stacking.raw_bound + 1e-10
    # (iii) the stacking certificate keeps the Jensen term.
    assert stacking.empirical >= stacking.mixture_empirical - 1e-12
    assert stacking.raw_bound == pytest.approx(
        stacking.empirical
        + stacking.complexity_penalty
        + stacking.concentration_penalty
    )
    # (iv) both mixtures are proper densities scoring an independent sample.
    X2, y2 = _features(x2), truth(x2)
    for rule in (stacking, gibbs):
        log_p = certifier.predict_logpdf(X2, y2, weights=rule.weights)
        assert np.all(np.isfinite(log_p))
        assert -log_p.mean() <= result.loss_upper
        assert -log_p.mean() >= result.loss_lower
    # The two rules genuinely differ on this dictionary.
    assert not np.allclose(stacking.weights, gibbs.weights, atol=1e-3)


def test_uncontaminated_interval_is_the_parameter_mixture_quantile(fitted):
    certifier, result, _ = fitted
    pilot = certifier.pilot_
    X = _features(np.array([-0.5, 0.3]))
    q = result.posterior_weights
    center, half = certifier.candidate_half_widths(X)
    y = center + 0.2
    # The uncontaminated CDF is the weighted projected-ball CDF alone.
    from popsregression._projected_ball import projected_ball_cdf

    expected = projected_ball_cdf((y - center)[:, None], half, pilot.ball_dim) @ q
    assert_allclose(certifier.predict_cdf(X, y, contaminated=False), expected)
    lo, hi = certifier.predict_interval(X, level=0.999, contaminated=False)
    assert_allclose(certifier.predict_cdf(X, lo, contaminated=False), 5e-4, atol=1e-8)
    assert_allclose(certifier.predict_cdf(X, hi, contaminated=False), 0.9995, atol=1e-8)
    # It stays inside the parameter mixture's support.
    a_max = half[:, q > 0].max(axis=1)
    assert np.all(lo >= center - a_max - 1e-9) and np.all(hi <= center + a_max + 1e-9)
