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
  ``b = FLOOR_BETA`` on the problem's declared output interval, the bounded
  loss certified by ``POPSEllipseRegression(regularization='PAC')``;
- ``rmse``: of the predictive mean.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge

from popsregression import POPSEllipseRegression, POPSRegression

from .bayesian_stacking import BayesianStacking
from .pvi import PredictiveVI

EXAMPLES = Path(__file__).resolve().parents[1]
LEVELS = (0.9545, 0.999)
FLOOR_BETA = 0.01
N_DRAWS = 4096

METHODS = (
    "Bayesian ridge",
    "POPS hypercube",
    "POPS ellipse",
    "Ellipse+EB",
    "Ellipse+PAC",
    "Bayesian stacking",
    "PVI",
)
POPS_METHODS = METHODS[:5]
# Supplementary variants, run alongside METHODS in the repeated-split study.
EXTRA_METHODS = ("PVI (no KL)",)


# ---------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    """A training draw and its fixed test set."""

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

    @property
    def n_params(self):
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


def ace_problem(ratio, seed, basis="subset"):
    """Linear ACE energies, 267 features projected onto 35 PCA modes.

    ``basis='subset'`` builds the PCA basis from the training subset only
    (no information from the other training structures); ``'pool'`` uses the
    unlabelled descriptors of all 700 training structures, as in the
    workshop version.
    """
    data = _ace_data()
    A, y = data["A_train_E"], data["y_train_E"]
    n = min(int(round(ratio * ACE_RANK)), y.size)
    idx = np.random.RandomState(seed).choice(y.size, n, replace=False)
    V = ace_pca_basis(A[idx] if basis == "subset" else A)
    return Problem(
        name="ace",
        X_train=A[idx] @ V,
        y_train=y[idx],
        X_test=data["A_test_E"] @ V,
        y_test=data["y_test_E"],
        y_bounds=ACE_Y_BOUNDS,
        stacking_components=[np.arange(k) for k in (5, 10, 20, ACE_RANK)],
        meta={"basis": basis, "indices": idx},
    )


# ---------------------------------------------------------------------------
# predictive wrappers
# ---------------------------------------------------------------------------


@dataclass
class Predictive:
    """Parameter-only predictive of one fitted method on a fixed input set."""

    name: str
    mean: np.ndarray
    intervals: dict
    logpdf: object = None  # callable y -> log density, or None
    fit_time: float = np.nan
    info: dict = field(default_factory=dict)


def _gaussian(name, loc, scale, fit_time, info=None):
    intervals = {}
    for level in LEVELS:
        z = norm.ppf(0.5 * (1 + level))
        intervals[level] = (loc - z * scale, loc + z * scale)
    return Predictive(
        name, loc, intervals, lambda y: norm.logpdf(y, loc, scale), fit_time, info or {}
    )


def fit_predictive(method, problem, seed, X_eval=None):
    """Fit ``method`` on ``problem`` and return its predictive at ``X_eval``."""
    X, y = problem.X_train, problem.y_train
    X_eval = problem.X_test if X_eval is None else X_eval
    start = time.perf_counter()
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
        )

    if method in ("POPS ellipse", "Ellipse+EB", "Ellipse+PAC"):
        regularization = {
            "POPS ellipse": None,
            "Ellipse+EB": "empirical-bayes",
            "Ellipse+PAC": "PAC",
        }[method]
        kwargs = {"regularization": regularization, "random_state": seed}
        fit_kwargs = {}
        if regularization == "PAC":
            kwargs["y_bounds"] = problem.y_bounds
            fit_kwargs["groups"] = problem.groups
        model = POPSEllipseRegression(**kwargs).fit(X, y, **fit_kwargs)
        fit_time = time.perf_counter() - start
        mean = model.predict(X_eval)
        intervals = {level: model.predict_interval(X_eval, level) for level in LEVELS}
        info = {"model": model}
        if regularization == "PAC":
            cert = model.certificate_
            info.update(bound=cert.raw_bound, trivial_bound=cert.trivial_bound)
            for key in (
                "lam",
                "empirical",
                "monte_carlo",
                "kl",
                "complexity",
                "concentration",
                "mixture_empirical",
                "outside_support_fraction",
            ):
                info[key] = float(np.mean([getattr(f, key) for f in cert.folds]))
        return Predictive(
            method,
            mean,
            intervals,
            lambda yy: model.predict_logpdf(X_eval, yy),
            fit_time,
            info,
        )

    if method == "Bayesian stacking":
        cv = "loo"
        model = BayesianStacking(problem.stacking_components, cv=cv, random_state=seed)
        model.fit(X, y, group_ids=problem.groups)
        fit_time = time.perf_counter() - start
        mean = model.predict(X_eval)
        intervals = {level: model.predict_interval(X_eval, level) for level in LEVELS}
        return Predictive(
            method,
            mean,
            intervals,
            lambda yy: model.predict_logpdf(X_eval, yy),
            fit_time,
            {"weights": model.weights_},
        )

    if method in ("PVI", "PVI (no KL)"):
        lam = "cv" if method == "PVI" else 0.0
        model = PredictiveVI(lam=lam, random_state=seed).fit(X, y)
        loc, scale = model.predict(X_eval, return_std=True)
        return _gaussian(
            method,
            loc,
            scale,
            time.perf_counter() - start,
            {"pvi_lam": model.lam_, "converged": model.converged_},
        )

    raise ValueError(f"unknown method {method!r}")


def evaluate(pred, y, y_bounds, floor_beta=FLOOR_BETA):
    """Parameter-only test metrics of a :class:`Predictive` (see module doc)."""
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
    if pred.logpdf is None:
        out.update(nll=np.nan, zero_density=np.nan, nll_floor=np.nan)
    else:
        logp = np.asarray(pred.logpdf(y), dtype=float)
        out["nll"] = float(-np.mean(logp))
        out["zero_density"] = float(np.mean(np.isneginf(logp)))
        width = y_bounds[1] - y_bounds[0]
        floored = np.logaddexp(np.log1p(-floor_beta) + logp, np.log(floor_beta / width))
        out["nll_floor"] = float(-np.mean(floored))
    return out


def parameter_draws(method, problem, seed, n_draws=2000):
    """Coefficient draws ``(coef (P, n), offset (n,))`` of a fitted method.

    Each draw is one complete surrogate function ``x -> F(x) @ coef + offset``
    and is reused for every input of a propagated field. No residual-noise
    term is added for any method.
    """
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
    if method in ("POPS ellipse", "Ellipse+EB", "Ellipse+PAC"):
        regularization = {
            "POPS ellipse": None,
            "Ellipse+EB": "empirical-bayes",
            "Ellipse+PAC": "PAC",
        }[method]
        kwargs = {"y_bounds": problem.y_bounds} if regularization == "PAC" else {}
        model = POPSEllipseRegression(
            regularization=regularization, random_state=seed, **kwargs
        )
        fit_kwargs = {"groups": problem.groups} if regularization == "PAC" else {}
        model.fit(X, y, **fit_kwargs)
        return model.sample(n_draws, random_state=seed), zeros
    if method == "Bayesian stacking":
        model = BayesianStacking(
            problem.stacking_components, cv="loo", random_state=seed
        ).fit(X, y, group_ids=problem.groups)
        coef, intercept, _, _ = model.sample_parameters(n_draws, random_state=seed)
        return coef, intercept
    if method in ("PVI", "PVI (no KL)"):
        lam = "cv" if method == "PVI" else 0.0
        model = PredictiveVI(lam=lam, random_state=seed).fit(X, y)
        return model.sample_parameters(n_draws, random_state=seed)
    raise ValueError(f"unknown method {method!r}")
