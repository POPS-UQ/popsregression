"""Common problems, methods and parameter-only metrics for the paper study.

Every method is wrapped as a :class:`Predictive` exposing exact central
intervals and, when the method has one, a parameter-only predictive log
density. No reported interval, density or score includes a residual-noise
(aleatoric) term: Bayesian ridge uses ``sigma_`` only, Bayesian stacking and
PVI use their parameter-only predictives, and the POPS hypercube is
represented by its parameter draws (it has no density, so no log loss).

Metrics (all on an independent test set, see :func:`evaluate`):

- ``coverage_<level>`` / ``width_<level>``: exact central intervals at the
  common levels 95.45% and 99.9%, identical in meaning across methods;
- ``interval_score``: Gneiting--Raftery interval score of the 95.45%
  interval (proper, finite for every method, lower is better);
- ``nll``: mean parameter-only negative log density (``inf`` if any test
  target has zero density, as happens outside a compact support);
- ``zero_density``: fraction of test targets with zero parameter-only
  density (uncovered by the support);
- ``nll_floor``: mean negative log of ``(1 - b) p + b / R_y`` with a common
  ``b = FLOOR_BETA = 0.01`` on the problem's declared output interval. This
  is a comparison score only; the loss certified by
  ``POPSEllipseRegression(regularization='PAC')`` uses its own
  ``floor_weight = 0.02``;
- ``rmse``: of the predictive mean;
- calibration areas (see :mod:`.calibration`): ``pp_area_abs`` /
  ``pp_area_signed`` of the pooled absolute-error P-P curve (the ACE figure
  metric), and ``pit_area_abs`` / ``pit_area_signed`` of the
  input-conditional PIT curve. Burgers test points are weighted so every
  simulator case counts equally.

Every method sees the same features and the same feature capacity. For ACE
the raw 267 descriptors are projected on 35 PCA modes plus a constant
column (P = 36); the projection is learned from training descriptors only,
and it is refitted wherever a method holds data out: on each pilot split
inside ``Ellipse+PAC`` (so the PAC certification units never influence the
features) and inside every validation fold of Bayesian stacking. The other
methods fit it once on their whole training subset, which is the same
total training budget. The comparison methods rescale features and target
but never center them, so none of them gains an implicit intercept.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import BayesianRidge

from popsregression import POPSEllipseRegression, POPSRegression

from . import calibration
from .bayesian_stacking import BayesianStacking
from .low_noise_objectives import LowNoiseObjective
from .pvi import PredictiveVI

EXAMPLES = Path(__file__).resolve().parents[1]
LEVELS = (0.9545, 0.999)
FLOOR_BETA = 0.01
N_DRAWS = 4096
# Predictive draws per test input for the pooled-error calibration curve.
N_CALIBRATION_DRAWS = 256

METHODS = (
    "Bayesian ridge",
    "POPS hypercube",
    "POPS ellipse",
    "Ellipse+EB",
    "Ellipse+PAC",
    "Bayesian stacking",
    "PVI",
    "PACm",
    "PAC2-T",
)
POPS_METHODS = METHODS[:5]
# Supplementary variants, run alongside METHODS in the repeated-split study.
EXTRA_METHODS = (
    "PVI (lamb=1)",
    "POPS ellipse (free center)",
    "Ellipse+EB (free center)",
    "Ellipse+PAC (free center)",
)
ELLIPSE_METHODS = {
    "POPS ellipse": None,
    "Ellipse+EB": "empirical-bayes",
    "Ellipse+PAC": "PAC",
}
# Published PVI (Lai, Linero and Yao; github.com/lll6924/pvi): code defaults
# (mean-field Gaussian, lamb = 0, learning rate 1e-3), except the published
# 'rmsprop' option with 20000 iterations, since plain SGD at that learning
# rate does not converge within any practical budget, and s = 16 draws.
PVI_SETTINGS = dict(
    sigma=1e-2,
    s=16,
    lamb=0.0,
    optimizer="rmsprop",
    iterations=20000,
    learning_rate=1e-3,
)
# PACm (Morningstar et al. 2022) and PAC^2_T (Masegosa 2020), variational,
# full-rank Gaussian; likelihood width 1% of the target standard deviation
# (the near-deterministic regime; the width sweep is in the low-noise study).
LOW_NOISE_SETTINGS = {
    "PACm": dict(objective="pacm", sigma=1e-2, n_samples=16),
    "PAC2-T": dict(objective="pac2t", sigma=1e-2, n_samples=16),
}


def _low_noise_model(method, problem, seed, X=None):
    settings = dict(LOW_NOISE_SETTINGS[method])
    objective = settings.pop("objective")
    X = problem.design()[0] if X is None else X
    return LowNoiseObjective(objective, random_state=seed, **settings).fit(
        X, problem.y_train
    )


def _ellipse_model(method, problem, seed):
    """Fit an ellipse-family method; '(free center)' optimizes the center."""
    base = method.replace(" (free center)", "")
    regularization = ELLIPSE_METHODS[base]
    kwargs = {
        "regularization": regularization,
        "random_state": seed,
        "optimize_center": method.endswith("(free center)"),
        "preprocessor": problem.preprocessor,
    }
    fit_kwargs = {}
    if regularization == "PAC":
        kwargs["y_bounds"] = problem.y_bounds
        fit_kwargs["groups"] = problem.groups
    return (
        POPSEllipseRegression(**kwargs).fit(
            problem.X_train, problem.y_train, **fit_kwargs
        ),
        regularization,
    )


def _pvi_model(method, problem, seed, X=None):
    settings = dict(PVI_SETTINGS)
    if method == "PVI (lamb=1)":
        settings["lamb"] = 1.0
    X = problem.design()[0] if X is None else X
    return PredictiveVI(random_state=seed, **settings).fit(X, problem.y_train)


# ---------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    """A training draw and its fixed test set.

    ``X_train`` and ``X_test`` are the raw inputs. When ``preprocessor`` is
    set (ACE), every method sees ``preprocessor``-transformed features, and
    the transformer is learned from training inputs only (see the module
    docstring for where it is refitted).
    """

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_bounds: tuple
    stacking_components: list
    groups: np.ndarray = None
    test_groups: np.ndarray = None
    meta: dict = field(default_factory=dict)
    preprocessor: object = None

    def design(self, X_eval=None):
        """Model features of the training and evaluation inputs.

        The preprocessor (if any) is fitted on the whole training subset.
        Returns ``(X_train, X_eval, fitted_preprocessor)``.
        """
        X_eval = self.X_test if X_eval is None else X_eval
        if self.preprocessor is None:
            return self.X_train, X_eval, None
        pre = clone(self.preprocessor).fit(self.X_train, self.y_train)
        return pre.transform(self.X_train), pre.transform(X_eval), pre

    @property
    def n_params(self):
        if self.preprocessor is not None:
            return int(self.preprocessor.n_output_features)
        return self.X_train.shape[1]

    @property
    def n_units(self):
        if self.groups is None:
            return self.y_train.size
        return np.unique(self.groups).size


def quartic_target(x):
    return (x**3 + 0.01 * x**4) * 0.1 + np.sin(x) * x * 10.0


def quartic_features(x):
    return np.vander(np.asarray(x, dtype=float), 5, increasing=True)


# Range of the target on [-10, 10] is [-144.4, 136.9]; the declared interval
# is known from the simulator (the analytic engine), not from samples.
QUARTIC_Y_BOUNDS = (-160.0, 160.0)


def quartic_problem(n_train, seed, n_test=2000, test_seed=2024):
    """Oscillatory engine, quartic surrogate (P = 5), x ~ U[-10, 10]."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-10.0, 10.0, n_train)
    xt = np.random.RandomState(test_seed).uniform(-10.0, 10.0, n_test)
    return Problem(
        name="quartic",
        X_train=quartic_features(x),
        y_train=quartic_target(x),
        X_test=quartic_features(xt),
        y_test=quartic_target(xt),
        y_bounds=QUARTIC_Y_BOUNDS,
        stacking_components=[np.arange(d + 1) for d in range(1, 5)],
        meta={"x_train": x, "x_test": xt},
    )


_BURGERS_CACHE = {}

# |u| <= max|u_0| = 1.3 by the maximum principle for viscous Burgers with
# u_0 = A sin x, A <= 1.3; the offline mean field obeys the same bound, so the
# target u - mean_field lies in [-2.6, 2.6].
BURGERS_Y_BOUNDS = (-2.6, 2.6)
# POD feature columns are mode-major: [m0 (1, nu, A, t), m1 (1, nu, A, t)].
BURGERS_COMPONENTS = [
    np.array([0, 4]),
    np.array([0, 1, 2, 3]),
    np.array([0, 2, 4, 6]),
    np.arange(8),
]


def _burgers_setup():
    if "setup" not in _BURGERS_CACHE:
        import example_burgers_pod as pod

        mean_field, modes, _, _ = pod.build_pod_basis(rank=2)
        rng = np.random.default_rng(pod.TEST_SEED)
        test_cases = _draw_burgers_cases(rng, 120)
        X_test, y_test = pod.simulate_cases(
            test_cases,
            modes,
            mean_field,
            points_per_case=16,
            random_x=True,
            seed=pod.TEST_SEED,
        )
        _BURGERS_CACHE["setup"] = (pod, mean_field, modes, test_cases, X_test, y_test)
    return _BURGERS_CACHE["setup"]


def _draw_burgers_cases(rng, n):
    import example_burgers as burgers

    return np.column_stack(
        [
            rng.uniform(*burgers.NU_RANGE, n),
            rng.uniform(*burgers.AMP_RANGE, n),
            rng.uniform(*burgers.T_RANGE, n),
        ]
    )


def burgers_problem(n_cases, seed, points_per_case=3):
    """Rank-2 POD Burgers emulator (P = 8); cases are the independent units."""
    pod, mean_field, modes, test_cases, X_test, y_test = _burgers_setup()
    rng = np.random.default_rng(10_000 + seed)
    cases = _draw_burgers_cases(rng, n_cases)
    X, y = pod.simulate_cases(
        cases,
        modes,
        mean_field,
        points_per_case=points_per_case,
        random_x=True,
        seed=20_000 + seed,
    )
    return Problem(
        name="burgers",
        X_train=X,
        y_train=y,
        X_test=X_test,
        y_test=y_test,
        y_bounds=BURGERS_Y_BOUNDS,
        stacking_components=BURGERS_COMPONENTS,
        groups=np.repeat(np.arange(n_cases), points_per_case),
        test_groups=np.repeat(np.arange(len(test_cases)), 16),
        meta={
            "cases": cases,
            "test_cases": test_cases,
            "modes": modes,
            "mean_field": mean_field,
        },
    )


# Declared, not proven: the bundled Cu energies span [-3.70, -2.70] (per atom);
# the interval below adds 0.5 on each side as a stated population assumption.
ACE_Y_BOUNDS = (-4.2, -2.2)
ACE_RANK = 35


def _ace_data():
    if "ace" not in _BURGERS_CACHE:
        with np.load(EXAMPLES / "ace_linear_uq_energies.npz") as archive:
            _BURGERS_CACHE["ace"] = {k: archive[k] for k in archive.files}
    return _BURGERS_CACHE["ace"]


def ace_pca_basis(A, rank=ACE_RANK):
    centered = A - A.mean(axis=0, keepdims=True)
    evals, evecs = np.linalg.eigh(centered.T @ centered / A.shape[0])
    return evecs[:, np.argsort(evals)[::-1][:rank]]


class PCAWithConstant(TransformerMixin, BaseEstimator):
    """Project on the leading ``rank`` PCA modes and append a constant column.

    ``fit`` learns the modes from the descriptors it is given (never from
    targets). With a fixed ``basis`` (shape ``(n_features, rank)``) nothing
    is learned: ``fit`` only records it.
    """

    def __init__(self, rank=ACE_RANK, basis=None):
        self.rank = rank
        self.basis = basis

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.components_ = (
            ace_pca_basis(X, self.rank)
            if self.basis is None
            else np.asarray(self.basis, dtype=float)
        )
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([X @ self.components_, np.ones((X.shape[0], 1))])

    @property
    def n_output_features(self):
        return int(self.rank) + 1


def ace_problem(ratio, seed, basis="subset"):
    """Linear ACE energies: 267 raw descriptors, projected by every method on
    35 PCA modes plus a constant column, so all fit the same affine model
    (P = 36).

    ``basis='subset'`` (default) learns the PCA modes from training
    descriptors only, refitted inside every held-out split (see the module
    docstring); this is the construction that keeps the PAC certification
    units out of the features. ``'pool'`` fixes the modes from the
    unlabelled descriptors of all 700 training structures, as in the
    workshop version; those include the certification units' descriptors,
    so it does NOT satisfy the pilot-only condition of the PAC bound.
    ``ratio`` is the number of training structures per PCA mode.
    """
    data = _ace_data()
    A, y = data["A_train_E"], data["y_train_E"]
    n = min(int(round(ratio * ACE_RANK)), y.size)
    idx = np.random.RandomState(seed).choice(y.size, n, replace=False)
    fixed = None if basis == "subset" else ace_pca_basis(A)
    constant = [ACE_RANK]
    return Problem(
        name="ace",
        X_train=A[idx],
        y_train=y[idx],
        X_test=data["A_test_E"],
        y_test=data["y_test_E"],
        y_bounds=ACE_Y_BOUNDS,
        stacking_components=[
            np.concatenate([np.arange(k), constant]) for k in (5, 10, 20, ACE_RANK)
        ],
        meta={"basis": basis, "indices": idx},
        preprocessor=PCAWithConstant(ACE_RANK, basis=fixed),
    )


# ---------------------------------------------------------------------------
# predictive wrappers
# ---------------------------------------------------------------------------


@dataclass
class Predictive:
    """Parameter-only predictive of one fitted method on a fixed input set.

    ``logpdf(y)`` and ``cdf(y)`` evaluate the predictive of each input at
    its own target (``None`` when the method has no density); ``draws(n,
    rng)`` returns ``(n_points, n)`` predictive draws (joint over inputs:
    each column is one parameter draw).
    """

    name: str
    mean: np.ndarray
    intervals: dict
    logpdf: object = None  # callable y -> log density, or None
    fit_time: float = np.nan
    info: dict = field(default_factory=dict)
    cdf: object = None  # callable y -> F(y)
    draws: object = None  # callable (n, rng) -> (n_points, n)


def _gaussian(name, loc, scale, fit_time, info=None):
    intervals = {}
    for level in LEVELS:
        z = norm.ppf(0.5 * (1 + level))
        intervals[level] = (loc - z * scale, loc + z * scale)
    return Predictive(
        name,
        loc,
        intervals,
        lambda y: norm.logpdf(y, loc, scale),
        fit_time,
        info or {},
        cdf=lambda y: norm.cdf(y, loc, scale),
        draws=lambda n, rng: loc[:, None] + scale[:, None] * rng.randn(loc.size, n),
    )


def _optimizer_info(model):
    """Convergence record of a PACm / PAC^2-T / PVI fit (all that exist)."""
    keys = (
        "converged_",
        "n_nonfinite_",
        "n_iter_",
        "n_fev_",
        "termination_status_",
        "termination_message_",
        "max_iter",
        "max_fun",
        "n_skipped_",
    )
    return {k.rstrip("_"): getattr(model, k) for k in keys if hasattr(model, k)}


def fit_predictive(method, problem, seed, X_eval=None):
    """Fit ``method`` on ``problem`` and return its predictive at ``X_eval``."""
    y = problem.y_train
    X_raw_eval = problem.X_test if X_eval is None else X_eval
    start = time.perf_counter()
    # Fixed-design methods: the preprocessor (ACE PCA) fitted on the whole
    # training subset. Ellipse-family and stacking handle it themselves.
    X, X_eval, _ = problem.design(X_raw_eval)
    if method == "Bayesian ridge":
        model = BayesianRidge(fit_intercept=False).fit(X, y)
        loc = X_eval @ model.coef_
        scale = np.sqrt(np.einsum("ij,jk,ik->i", X_eval, model.sigma_, X_eval))
        return _gaussian(method, loc, scale, time.perf_counter() - start)

    if method == "POPS hypercube":
        model = POPSRegression(
            minimum_relative_error=0.0,
            resample_density=N_DRAWS / y.size,
            random_state=seed,
        ).fit(X, y)
        mean = model.predict(X_eval)
        draws = mean[:, None] + X_eval @ model.posterior_samples_
        intervals = {
            level: tuple(
                np.quantile(draws, [0.5 * (1 - level), 0.5 * (1 + level)], axis=1)
            )
            for level in LEVELS
        }
        return Predictive(
            method,
            mean,
            intervals,
            None,
            time.perf_counter() - start,
            {"n_draws": draws.shape[1]},
            cdf=lambda yy: np.mean(draws <= np.asarray(yy)[:, None], axis=1),
            draws=lambda n, rng: draws[:, rng.randint(draws.shape[1], size=n)],
        )

    if method.replace(" (free center)", "") in ELLIPSE_METHODS:
        model, regularization = _ellipse_model(method, problem, seed)
        fit_time = time.perf_counter() - start
        Xe = X_raw_eval
        mean = model.predict(Xe)
        intervals = {level: model.predict_interval(Xe, level) for level in LEVELS}
        info = {"model": model}
        if regularization == "PAC":
            cert = model.certificate_
            info.update(
                bound=cert.raw_bound,
                continuous_bound=cert.continuous_bound,
                trivial_bound=cert.trivial_bound,
                n_units=cert.n_units,
                failure_probability=cert.failure_probability,
            )
            for key in (
                "lam",
                "empirical",
                "monte_carlo",
                "kl",
                "complexity",
                "concentration",
                "mixture_empirical",
                "outside_support_fraction",
                "moment_constant",
                "n_stored",
            ):
                info[key] = float(np.mean([getattr(f, key) for f in cert.folds]))
            info["n_cert_units"] = int(sum(f.n_units for f in cert.folds))
        return Predictive(
            method,
            mean,
            intervals,
            lambda yy: model.predict_logpdf(Xe, yy),
            fit_time,
            info,
            cdf=lambda yy: model.predict_cdf(Xe, yy),
            draws=lambda n, rng: model.sample_predictions(Xe, n, random_state=rng),
        )

    if method == "Bayesian stacking":
        model = BayesianStacking(
            problem.stacking_components,
            cv="loo",
            random_state=seed,
            preprocessor=problem.preprocessor,
        )
        model.fit(problem.X_train, y, group_ids=problem.groups)
        fit_time = time.perf_counter() - start
        Xe = X_raw_eval
        mean = model.predict(Xe)
        intervals = {level: model.predict_interval(Xe, level) for level in LEVELS}

        def stacking_draws(n, rng):
            coef, intercept, _, _ = model.sample_parameters(n, random_state=rng)
            return model.transform(Xe) @ coef + intercept[None, :]

        return Predictive(
            method,
            mean,
            intervals,
            lambda yy: model.predict_logpdf(Xe, yy),
            fit_time,
            {"weights": model.weights_},
            cdf=lambda yy: model.predict_cdf(Xe, yy),
            draws=stacking_draws,
        )

    if method in ("PVI", "PVI (lamb=1)"):
        model = _pvi_model(method, problem, seed, X)
        loc, scale = model.predict(X_eval, return_std=True)
        return _gaussian(
            method,
            loc,
            scale,
            time.perf_counter() - start,
            {"pvi_lam": model.lamb, **_optimizer_info(model)},
        )

    if method in LOW_NOISE_SETTINGS:
        model = _low_noise_model(method, problem, seed, X)
        loc, scale = model.predict(X_eval, return_std=True)
        return _gaussian(
            method, loc, scale, time.perf_counter() - start, _optimizer_info(model)
        )

    raise ValueError(f"unknown method {method!r}")


def calibration_record(pred, y, weights=None, n_draws=N_CALIBRATION_DRAWS, seed=0):
    """PIT values, pooled-error P-P curve and both calibration areas.

    Returns a dict with ``pit`` (``F_i(y_i)``), the P-P curve ``(pp_u,
    pp_C)`` and the four areas (see :mod:`.calibration`).
    """
    out = {}
    if pred.cdf is not None:
        pit = np.asarray(pred.cdf(y), dtype=float)
        _, _, s_pit, a_pit = calibration.pit_curve(pit, weights)
        out.update(pit=pit, pit_area_signed=s_pit, pit_area_abs=a_pit)
    if pred.draws is not None:
        rng = np.random.RandomState(seed)
        errors = pred.draws(n_draws, rng) - pred.mean[:, None]
        u, C, s_pp, a_pp = calibration.error_pp_curve(y - pred.mean, errors, weights)
        out.update(pp_u=u, pp_C=C, pp_area_signed=s_pp, pp_area_abs=a_pp)
    return out


def evaluate(pred, y, y_bounds, floor_beta=FLOOR_BETA, weights=None, record=None):
    """Parameter-only test metrics of a :class:`Predictive` (see module doc).

    ``weights`` weight the test points (every Burgers case equally) in the
    calibration areas; the other metrics are plain test-point averages, as
    before. If ``record`` is a dict it receives the per-point arrays
    (``pit``, ``logpdf``, the P-P curve) for export.
    """
    out = {
        "fit_time": pred.fit_time,
        "rmse": float(np.sqrt(np.mean((pred.mean - y) ** 2))),
    }
    for level in LEVELS:
        lo, hi = pred.intervals[level]
        tag = f"{100 * level:g}"
        out[f"coverage_{tag}"] = float(np.mean((y >= lo) & (y <= hi)))
        out[f"width_{tag}"] = float(np.mean(hi - lo))
    lo, hi = pred.intervals[LEVELS[0]]
    alpha = 1.0 - LEVELS[0]
    score = (
        (hi - lo)
        + (2 / alpha) * (lo - y) * (y < lo)
        + (2 / alpha) * (y - hi) * (y > hi)
    )
    out["interval_score"] = float(np.mean(score))
    logp = None
    if pred.logpdf is None:
        out.update(nll=np.nan, zero_density=np.nan, nll_floor=np.nan)
    else:
        logp = np.asarray(pred.logpdf(y), dtype=float)
        out["nll"] = float(-np.mean(logp))
        out["zero_density"] = float(np.mean(np.isneginf(logp)))
        width = y_bounds[1] - y_bounds[0]
        floored = np.logaddexp(np.log1p(-floor_beta) + logp, np.log(floor_beta / width))
        out["nll_floor"] = float(-np.mean(floored))
    cal = calibration_record(pred, y, weights)
    for key in ("pp_area_signed", "pp_area_abs", "pit_area_signed", "pit_area_abs"):
        out[key] = cal.get(key, np.nan)
    if record is not None:
        record.update(
            mean=pred.mean,
            logpdf=logp,
            **{f"lower_{100 * lv:g}": pred.intervals[lv][0] for lv in LEVELS},
            **{f"upper_{100 * lv:g}": pred.intervals[lv][1] for lv in LEVELS},
            **{k: cal[k] for k in ("pit", "pp_u", "pp_C") if k in cal},
        )
    return out


def parameter_draws(method, problem, seed, n_draws=2000, info=None):
    """Coefficient draws ``(coef (P, n), offset (n,))`` of a fitted method.

    Each draw is one complete surrogate function ``x -> F(x) @ coef + offset``
    and is reused for every input of a propagated field. For the POPS
    ellipse family the offset is the draw's output-offset coordinate (the
    width floor, ``|offset| <= delta``), so propagated fields come from the
    same parameter distribution as the scored intervals. No residual-noise
    term is added for any method. Problems with a preprocessor (ACE) are
    not supported here. If ``info`` is a dict it receives the optimizer
    record of iterative methods (``converged``, iteration counts, ...).
    """
    info = {} if info is None else info
    if problem.preprocessor is not None:
        raise ValueError("parameter_draws needs a fixed design (no preprocessor).")
    X, y = problem.X_train, problem.y_train
    rng = np.random.RandomState(seed)
    zeros = np.zeros(n_draws)
    if method == "Bayesian ridge":
        model = BayesianRidge(fit_intercept=False).fit(X, y)
        L = np.linalg.cholesky(model.sigma_ + 1e-12 * np.eye(X.shape[1]))
        return model.coef_[:, None] + L @ rng.randn(X.shape[1], n_draws), zeros
    if method == "POPS hypercube":
        model = POPSRegression(
            minimum_relative_error=0.0,
            resample_density=n_draws / y.size,
            random_state=seed,
        ).fit(X, y)
        return model.coef_[:, None] + model.posterior_samples_, np.zeros(
            model.posterior_samples_.shape[1]
        )
    if method.replace(" (free center)", "") in ELLIPSE_METHODS:
        model, _ = _ellipse_model(method, problem, seed)
        return model.sample(n_draws, random_state=seed, return_offset=True)
    if method == "Bayesian stacking":
        model = BayesianStacking(
            problem.stacking_components, cv="loo", random_state=seed
        ).fit(X, y, group_ids=problem.groups)
        coef, intercept, _, _ = model.sample_parameters(n_draws, random_state=seed)
        return coef, intercept
    if method in ("PVI", "PVI (lamb=1)"):
        model = _pvi_model(method, problem, seed)
        info.update(_optimizer_info(model))
        return model.sample_parameters(n_draws, random_state=seed)
    if method in LOW_NOISE_SETTINGS:
        model = _low_noise_model(method, problem, seed)
        info.update(_optimizer_info(model))
        theta = model.mu_[:, None] + model.L_ @ rng.randn(model.mu_.size, n_draws)
        return model.y_scale_ * theta / model.x_scale_[:, None], zeros
    raise ValueError(f"unknown method {method!r}")
