"""Low-noise study: PACm, PAC^2-T and PVI against sigma-free references.

The finite-noise objectives fit a Gaussian working likelihood
``N(y; x.theta, sigma^2)``. Here its width ``sigma`` (in units of the
training-target standard deviation) is swept down to ``3e-4`` on two
problems with fixed budgets of independent units and paired training draws
(the same seed gives the same training draw for every method and width):

- the quartic surrogate of the deterministic oscillatory engine (P = 5),
  N = 10 and 50 training points;
- the rank-2 POD Burgers emulator (P = 8), 8 and 24 simulator cases with 3
  points each (the cases are the independent units, and every test case is
  weighted equally in every score and calibration area).

Methods, at several Monte Carlo sample counts:

- PACm-Bayes (Morningstar et al. 2022), ``m = 1, 16, 64``, full-rank
  Gaussian, KL weight 1, L-BFGS on a fixed sample average;
- PAC^2-T variational (Masegosa 2020), ``S = 1, 16, 64`` pairs per group
  (``S`` only changes the number of Monte Carlo pairs, not the objective;
  ``S = 1`` is the published one-pair-per-step convention);
- predictive variational inference as published (Lai, Linero and Yao;
  ``comparisons.pvi``, the ``harness.PVI_SETTINGS`` with only ``sigma`` and
  ``s`` varied), ``s = 1, 16``: the same Monte Carlo log score with fresh
  draws at every step and stochastic optimization.

The POPS Ellipse, Ellipse+EB, Ellipse+PAC and Bayesian stacking do not use
``sigma`` and are drawn as horizontal references.

For fixed ``m`` (or ``s``) the finite-noise training objectives grow like
``sigma^{-2}`` once ``sigma`` is below the typical nearest residual; a
larger finite ``m`` only moves the onset to smaller ``sigma``. A diverging
objective is not the same as collapsed predictions, so both are recorded:
the training objective (split into data and KL terms, on the fixed training
draws), the same objective re-estimated on fresh independent draws at the
fitted parameters, the optimizer's termination reason, iteration and
evaluation counts, and the parameter-only predictive scores. Every
prediction, interval and score uses the parameter-only predictive (no
residual-noise term is added). Failed or unconverged fits are kept.

Writes ``generated/low_noise_objectives.csv`` and ``.md`` and the figure
``low_noise_objectives.png``.
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from comparisons import calibration, harness, labels  # noqa: E402
from comparisons.low_noise_objectives import LowNoiseObjective  # noqa: E402
from comparisons.pvi import PredictiveVI  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
SIGMAS = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)
PROBLEMS = {"quartic": (10, 50), "burgers": (8, 24)}
PROBLEM_TITLES = {"quartic": "Quartic (P = 5)", "burgers": "Burgers POD (P = 8)"}
UNIT_NAMES = {"quartic": "points", "burgers": "cases"}
CONFIGS = (
    ("pacm", 1),
    ("pacm", 16),
    ("pacm", 64),
    ("pac2t", 1),
    ("pac2t", 16),
    ("pac2t", 64),
    ("pvi", 1),
    ("pvi", 16),
)
REFERENCES = ("POPS ellipse", "Ellipse+EB", "Ellipse+PAC", "Bayesian stacking")
SEED_OFFSET = 500  # training draw of repeat r: seed 500 + r (as before)
FRESH_SEED = 70_000  # fresh Monte Carlo draws of the objective re-evaluation
FRESH_GROUPS = 256
N_DRAWS = 256  # predictive draws per test input for the pooled P-P curve
PVI_TAIL = 200  # PVI training objective: mean over the last steps
TERMINATIONS = ("converged", "max_iter", "max_fun", "line_search", "other", "error")


def config_name(objective, n_samples, tex=False):
    """Reader-facing name of a finite-noise configuration."""
    if objective == "pacm":
        name = f"{labels.display('PACm')} (m={n_samples})"
    elif objective == "pac2t":
        name = f"{labels.display('PAC2-T')} (S={n_samples})"
    else:
        name = f"{labels.display('PVI')} (s={n_samples})"
    return name if tex else name.replace("$", "")


def make_problem(name, n_units, repeat):
    seed = SEED_OFFSET + repeat
    if name == "quartic":
        return harness.quartic_problem(n_units, seed)
    return harness.burgers_problem(n_units, seed)


# ---------------------------------------------------------------------------
# parameter-only predictive scores (test inputs weighted by problem.test_groups)
# ---------------------------------------------------------------------------


def _weights(problem):
    w = calibration.group_weights(problem.test_groups)
    n = problem.y_test.size
    return np.full(n, 1.0 / n) if w is None else w / w.sum()


def _scores(y, w, y_bounds, mean, intervals, logp, pit, draws):
    """Weighted scores of one parameter-only predictive on the test set."""
    out = {"rmse": float(np.sqrt(w @ (mean - y) ** 2))}
    for level in harness.LEVELS:
        lo, hi = intervals[level]
        tag = f"{100 * level:g}"
        out[f"coverage_{tag}"] = float(w @ ((y >= lo) & (y <= hi)))
        out[f"width_{tag}"] = float(w @ (hi - lo))
    lo, hi = intervals[harness.LEVELS[0]]
    alpha = 1.0 - harness.LEVELS[0]
    score = (hi - lo) + (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    out["interval_score"] = float(w @ score)
    if logp is None:
        out.update(nll=np.nan, nll_floor=np.nan, zero_density=np.nan)
    else:
        logp = np.asarray(logp, dtype=float)
        with np.errstate(invalid="ignore"):
            out["nll"] = float(-(w @ logp))
        out["zero_density"] = float(w @ np.isneginf(logp))
        beta = harness.FLOOR_BETA
        width = y_bounds[1] - y_bounds[0]
        floored = np.logaddexp(np.log1p(-beta) + logp, np.log(beta / width))
        out["nll_floor"] = float(-(w @ floored))
    finite = np.all(np.isfinite(mean)) and np.all(np.isfinite(draws))
    if finite:
        _, _, s_pp, a_pp = calibration.error_pp_curve(
            y - mean, draws - mean[:, None], w
        )
        out.update(pp_area_signed=s_pp, pp_area_abs=a_pp)
    if pit is not None and np.all(np.isfinite(pit)):
        _, _, s_pit, a_pit = calibration.pit_curve(pit, w)
        out.update(pit_area_signed=s_pit, pit_area_abs=a_pit)
    return out


def _gaussian_scores(problem, loc, scale, seed):
    """Scores of the parameter-only Gaussian ``N(loc, scale^2)``; a zero
    scale is a point mass (zero density off the point)."""
    y, w = problem.y_test, _weights(problem)
    loc, scale = np.asarray(loc, dtype=float), np.asarray(scale, dtype=float)
    safe = np.where(scale > 0, scale, 1.0)
    intervals = {}
    for level in harness.LEVELS:
        z = norm.ppf(0.5 * (1 + level))
        intervals[level] = (loc - z * scale, loc + z * scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(
            scale > 0,
            norm.logpdf(y, loc, safe),
            np.where(y == loc, np.inf, -np.inf),
        )
        pit = np.where(scale > 0, norm.cdf(y, loc, safe), (y >= loc).astype(float))
    rng = np.random.RandomState(seed)
    draws = loc[:, None] + scale[:, None] * rng.randn(y.size, N_DRAWS)
    return _scores(y, w, problem.y_bounds, loc, intervals, logp, pit, draws)


def _grid_cdf(pred, n_grid=2049):
    """Numerical CDF of a predictive known only through its log density."""
    lo, hi = pred.intervals[max(harness.LEVELS)]
    span = np.maximum(hi - lo, 1e-12)
    a, b = lo - 2.0 * span, hi + 2.0 * span
    t = np.linspace(0.0, 1.0, n_grid)
    grid = a[:, None] + (b - a)[:, None] * t[None, :]
    dens = np.column_stack([np.exp(pred.logpdf(grid[:, k])) for k in range(n_grid)])
    steps = 0.5 * (dens[:, 1:] + dens[:, :-1]) * np.diff(grid, axis=1)
    cdf = np.concatenate([np.zeros((grid.shape[0], 1)), np.cumsum(steps, axis=1)], 1)
    cdf /= np.where(cdf[:, -1:] > 0, cdf[:, -1:], 1.0)
    return grid, cdf


def _reference_scores(pred, problem, seed):
    y, w = problem.y_test, _weights(problem)
    model = pred.info.get("model")
    logp = None if pred.logpdf is None else pred.logpdf(y)
    grid = None
    if model is not None and hasattr(model, "predict_cdf"):
        pit = model.predict_cdf(problem.X_test, y)
    else:
        grid, cdf = _grid_cdf(pred)
        pit = np.array([np.interp(v, g, c) for v, g, c in zip(y, grid, cdf)])
    if model is not None and hasattr(model, "sample_predictions"):
        draws = model.sample_predictions(problem.X_test, N_DRAWS, random_state=seed)
    else:
        if grid is None:
            grid, cdf = _grid_cdf(pred)
        u = np.random.RandomState(seed).uniform(size=(y.size, N_DRAWS))
        draws = np.array([np.interp(q, c, g) for q, g, c in zip(u, grid, cdf)])
    return _scores(y, w, problem.y_bounds, pred.mean, pred.intervals, logp, pit, draws)


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------


def _reference_rows(problem, base, repeat):
    rows = []
    log_std = np.log(problem.y_train.std())
    for method in REFERENCES:
        row = {**base, "method": method, "sigma": np.nan, "n_samples": np.nan}
        try:
            pred = harness.fit_predictive(method, problem, repeat)
            model = pred.info.get("model")
            # The POPS training objective has no sigma (per datum, std units).
            train = getattr(model, "objective_", np.nan) - log_std
            row.update(
                converged=True,
                termination="closed_form",
                train_objective=train,
                fit_time=pred.fit_time,
                **_reference_scores(pred, problem, 90_000 + repeat),
            )
        except Exception as exc:  # keep the failure as a row
            row.update(converged=False, termination="error", error=_error(exc))
        rows.append(row)
    return rows


def _error(exc):
    return f"{type(exc).__name__}: {exc}"[:400]


def _fit_row(objective, n_samples, sigma, problem, base, repeat):
    X, y = problem.X_train, problem.y_train
    row = {**base, "method": objective, "sigma": sigma, "n_samples": n_samples}
    try:
        start = time.perf_counter()
        n = y.size
        if objective == "pvi":
            settings = dict(harness.PVI_SETTINGS, sigma=sigma, s=n_samples)
            model = PredictiveVI(random_state=repeat, **settings).fit(X, y)
            tail = max(1, min(PVI_TAIL, len(model.score_values_)))
            # The published loop maximizes; as a per-datum quantity to
            # minimize, averaged over the last steps (their own fresh draws).
            train_data = -np.mean(model.score_values_[-tail:]) / n
            train_kl = -model.lamb * np.mean(model.kl_values_[-tail:]) / n
            row.update(
                converged=model.converged_,
                termination=model.termination_,
                termination_status=np.nan,
                termination_message=f"fixed budget of {model.iterations} steps",
                n_iter=model.n_iter_,
                n_fev=model.n_fev_,
                max_iter=model.iterations,
                max_fun=np.nan,
                n_nonfinite=model.n_skipped_,
                train_variance=np.nan,
            )
        else:
            model = LowNoiseObjective(
                objective, sigma=sigma, n_samples=n_samples, random_state=repeat
            ).fit(X, y)
            train_data, train_kl = model.objective_data_, model.objective_kl_
            row.update(
                converged=model.converged_,
                termination=model.termination_,
                termination_status=model.termination_status_,
                termination_message=model.termination_message_,
                n_iter=model.n_iter_,
                n_fev=model.n_fev_,
                max_iter=model.max_iter_,
                max_fun=model.max_fun_,
                n_nonfinite=model.n_nonfinite_,
                train_variance=model.objective_variance_,
            )
        fit_time = time.perf_counter() - start
        # Objectives are per datum in RMS-scaled target units; the data term
        # (a log density) is shifted to target-std units, the KL is unitless.
        shift = np.log(model.y_scale_) - np.log(y.std())
        fresh_data, fresh_kl, _ = model.evaluate_objective(
            X, y, n_groups=FRESH_GROUPS, random_state=FRESH_SEED + repeat
        )
        row.update(
            train_objective=train_data + shift + train_kl,
            train_data=train_data + shift,
            train_kl=train_kl,
            fresh_objective=fresh_data + shift + fresh_kl,
            fresh_data=fresh_data + shift,
            fresh_kl=fresh_kl,
            fit_time=fit_time,
        )
        loc, scale = model.predict(problem.X_test, return_std=True)
        row.update(_gaussian_scores(problem, loc, scale, 90_000 + repeat))
    except Exception as exc:  # keep the failure as a row
        row.update(
            converged=False,
            termination="error",
            error=_error(exc),
            traceback=traceback.format_exc(limit=3)[-600:],
        )
    return row


def run_task(task):
    """One (problem, units, repeat, config) cell: every width of one config,
    or the four sigma-free references."""
    name, n_units, repeat, config, sigmas = task
    warnings.filterwarnings("ignore")
    base = dict(problem=name, N=n_units, repeat=repeat)
    try:
        problem = make_problem(name, n_units, repeat)
    except Exception as exc:
        return [{**base, "method": str(config), "termination": "error",
                 "converged": False, "error": _error(exc)}]  # fmt: skip
    if config == "references":
        return _reference_rows(problem, base, repeat)
    objective, n_samples = config
    return [
        _fit_row(objective, n_samples, sigma, problem, base, repeat) for sigma in sigmas
    ]


def run(problems, repeats, workers, sigmas):
    tasks = []
    # Slowest configurations first, for load balance.
    order = sorted(CONFIGS, key=lambda c: (c[0] == "pvi", -c[1]))
    for config in [*order, "references"]:
        for name, sizes in problems.items():
            for n_units in sizes:
                for r in range(repeats):
                    tasks.append((name, n_units, r, config, sigmas))
    rows = []
    OUT.mkdir(exist_ok=True)
    partial = OUT / "low_noise_objectives.partial.csv"
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for k, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if k % 10 == 0 or k == len(tasks):
                pd.DataFrame(rows).to_csv(partial, index=False)
                errors = sum(r.get("termination") == "error" for r in rows)
                print(
                    (
                        f"[{time.perf_counter() - start:7.0f} s]"
                        f" {k}/{len(tasks)} tasks, {len(rows)} rows, {errors} errors"
                    ),
                    flush=True,
                )
    frame = pd.DataFrame(rows).sort_values(
        ["problem", "N", "method", "n_samples", "sigma", "repeat"],
        na_position="first",
        kind="stable",
    )
    frame.to_csv(OUT / "low_noise_objectives.csv", index=False)
    partial.unlink(missing_ok=True)
    return frame


# ---------------------------------------------------------------------------
# summary tables
# ---------------------------------------------------------------------------


def _median(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.median(v)) if v.size else np.nan


def _fmt(value, fmt="{:.3g}"):
    if value is None or np.isnan(value):
        return "--"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return fmt.format(value)


def _select(frame, problem, n_units, method, n_samples=None, sigma=None):
    s = frame[
        (frame.problem == problem) & (frame.N == n_units) & (frame.method == method)
    ]
    if n_samples is not None:
        s = s[s.n_samples == n_samples]
    if sigma is not None:
        s = s[np.isclose(s.sigma, sigma)]
    return s


def _reason_counts(s):
    """Fits per termination reason; PVI's fixed budget, closed-form fits and
    unknown reasons count as 'other'."""
    reasons = s.termination.where(s.termination.isin(TERMINATIONS), "other")
    return reasons.value_counts()


def _terminations(s):
    counts = _reason_counts(s)
    return "/".join(str(int(counts.get(t, 0))) for t in TERMINATIONS)


def _slope(sigmas, values):
    sig, val = np.asarray(sigmas, dtype=float), np.asarray(values, dtype=float)
    if sig.size < 2 or np.any(~np.isfinite(val)) or np.any(val <= 0):
        return np.nan
    return float(np.polyfit(np.log(sig), np.log(val), 1)[0])


def summarize(frame):
    frame = frame.copy()
    for col in ("error", "termination"):
        if col not in frame:
            frame[col] = np.nan
    repeats = frame.repeat.nunique()
    sigmas = sorted(frame.sigma.dropna().unique(), reverse=True)
    pac2 = labels.display("PAC2-T")
    keys = ["POPS", "EB", "PAC", "PACm", pac2, "PVI", "RMSE", "IS", "NLL"]
    abbreviations = labels.abbreviation_note(
        [k for k in keys if k in labels.ABBREVIATIONS]
    ).replace("$", "")
    pac2 = pac2.replace("$", "")
    lines = [
        f"# Low-noise study: PACm, {pac2} and PVI against sigma-free references\n",
        (
            f"Medians over {repeats} paired training draws (the same draw for every "
            "method and width) at fixed budgets of independent units: quartic "
            "surrogate (P = 5) with N training points, and Burgers POD emulator "
            "(P = 8) with N simulator cases of 3 points each, every test case "
            "weighted equally. `sigma` is the width of the Gaussian working "
            "likelihood in units of the training-target standard deviation; the "
            "references (POPS Ellipse, POPS Ellipse+EB, POPS Ellipse+PAC, Bayesian "
            "stacking) do not use it. Every score is of the parameter-only "
            "predictive (no residual-noise term). "
            + abbreviations
            + "\n"
        ),
        (
            "Objective columns are per datum, in nats with the target in "
            "standard-deviation units: `train` is the minimized objective on the "
            "fixed training draws (for PVI, the mean of its last "
            f"{PVI_TAIL} stochastic steps), split into its data term and its "
            "KL term (KL(q || prior) / n; 0 for PVI with the published lamb = 0); "
            "`fresh` is the same objective at the fitted parameters re-estimated "
            f"on {FRESH_GROUPS} fresh independent groups of draws (m-tuples for "
            f"PACm, S pairs for {pac2}, s draws for PVI). A training objective "
            "that grows like sigma^-2 is a diverging objective; it is not the "
            "same as a collapsed predictive, which the coverage columns show.\n"
        ),
        (
            "`conv.` is the fraction of fits whose optimizer reported convergence "
            "(for PVI, which runs a fixed budget of 20000 steps with no "
            "convergence test, the fraction with finite final parameters). "
            "`stops` counts the fits by termination reason, in the order "
            "converged / L-BFGS iteration limit (maxiter) / L-BFGS "
            "function-evaluation limit (maxfun) / failed line search (scipy's "
            "'ABNORMAL' stop) / other "
            "(including PVI's fixed budget and closed-form fits) / error. For "
            f"PACm and {pac2} maxiter = 20000 and maxfun = 40000 are separate "
            "limits; `iter.` and `f-evals` are the median L-BFGS iteration and "
            "objective-and-gradient evaluation counts actually used, `non-fin.` "
            "the mean number of non-finite objective evaluations per fit.\n"
        ),
        (
            "Score columns: RMSE; `cov.` the weighted fraction of test targets "
            "inside the central 95.45% and 99.9% intervals; IS; `raw NLL` the mean "
            "NLL (inf if a test target has zero density) and `floored NLL` the "
            f"same with a floor b = {harness.FLOOR_BETA} spread uniformly over the "
            "declared output interval; `zero-dens.` the fraction of test targets "
            "with zero density. Calibration areas "
            "(A_abs = int |C(u) - u| du, A_s = int (C(u) - u) du): `P-P` compares "
            f"the pooled distribution of test |errors| with that of {N_DRAWS} "
            "parameter-only predictive draws per test input (positive A_s: net "
            "over-confidence); `PIT` is the input-conditional curve of "
            "u_i = F_i(y_i).\n"
        ),
    ]
    lines += _growth_table(frame, sigmas)
    lines += _stops_table(frame)
    for name in [p for p in PROBLEMS if p in set(frame.problem)]:
        for n_units in sorted(frame[frame.problem == name].N.unique()):
            lines += _detail_tables(frame, name, n_units, sigmas)
    (OUT / "low_noise_objectives.md").write_text("\n".join(lines) + "\n")


def _growth_table(frame, sigmas):
    smallest = sigmas[-3:]
    lines = [
        "\n## Objective growth\n",
        (
            f"Median training objective at the largest (sigma = {sigmas[0]:g}) and "
            f"smallest (sigma = {sigmas[-1]:g}) widths, their ratio, and the fitted "
            "log-log slope d log(objective) / d log(sigma) of the median training "
            "and fresh objectives over the three smallest widths "
            f"({', '.join(f'{s:g}' for s in smallest)}): about -2 where the "
            "sigma^-2 divergence has set in (-- if a median is not positive). The "
            "last columns give the median coverage of the 95.45% interval at "
            "sigma = 0.01 and at the smallest width.\n"
        ),
        (
            "| problem | N | method | train, sigma = "
            f"{sigmas[0]:g} | train, sigma = {sigmas[-1]:g} | ratio | slope (train) | "
            "slope (fresh) | cov. 95.45%, sigma = 0.01 | cov. 95.45%, sigma = "
            f"{sigmas[-1]:g} |"
        ),
        "|---" * 10 + "|",
    ]
    for name in [p for p in PROBLEMS if p in set(frame.problem)]:
        for n_units in sorted(frame[frame.problem == name].N.unique()):
            for objective, n_samples in CONFIGS:
                s = _select(frame, name, n_units, objective, n_samples)
                if s.empty:
                    continue
                med = s.groupby("sigma").train_objective.apply(_median)
                fresh = s.groupby("sigma").fresh_objective.apply(_median)
                cov = s.groupby("sigma")["coverage_95.45"].apply(_median)
                first, last = med.get(sigmas[0], np.nan), med.get(sigmas[-1], np.nan)
                ratio = last / first if first and np.isfinite(first) else np.nan
                lines.append(
                    f"| {name} | {n_units} | {config_name(objective, n_samples)} | "
                    f"{_fmt(first)} | {_fmt(last)} | {_fmt(ratio)} | "
                    f"{_fmt(_slope(smallest, med.reindex(smallest)), '{:.2f}')} | "
                    f"{_fmt(_slope(smallest, fresh.reindex(smallest)), '{:.2f}')} | "
                    f"{_fmt(cov.get(0.01, np.nan), '{:.3f}')} | "
                    f"{_fmt(cov.get(sigmas[-1], np.nan), '{:.3f}')} |"
                )
    return lines


def _stops_table(frame):
    lines = [
        "\n## Optimizer termination (all widths and budgets)\n",
        (
            "Number of fits by termination reason, and the largest iteration "
            "and function-evaluation counts used, against the limits.\n"
        ),
        (
            "| problem | method | fits | converged | maxiter reached | maxfun reached "
            "| line search failed | other | error | max iter. used | max f-evals used "
            "| maxiter | maxfun |"
        ),
        "|---" * 13 + "|",
    ]
    for name in [p for p in PROBLEMS if p in set(frame.problem)]:
        for objective, n_samples in CONFIGS:
            s = frame[
                (frame.problem == name)
                & (frame.method == objective)
                & (frame.n_samples == n_samples)
            ]
            if s.empty:
                continue
            counts = _reason_counts(s)
            cells = " | ".join(str(int(counts.get(t, 0))) for t in TERMINATIONS)
            lines.append(
                f"| {name} | {config_name(objective, n_samples)} | {len(s)} | "
                f"{cells} | {_fmt(s.n_iter.max(), '{:.0f}')} | "
                f"{_fmt(s.n_fev.max(), '{:.0f}')} | "
                f"{_fmt(s.max_iter.max(), '{:.0f}')} | "
                f"{_fmt(s.max_fun.max(), '{:.0f}')} |"
            )
    return lines


def _detail_tables(frame, name, n_units, sigmas):
    unit = UNIT_NAMES[name]
    head_a = (
        "| method | sigma | conv. | stops | iter. | f-evals | non-fin. | train | "
        "train data | train KL | fresh | fresh data |"
    )
    head_b = (
        "| method | sigma | RMSE | cov. 95.45% | cov. 99.9% | width 95.45% | IS | "
        "raw NLL | floored NLL | zero-dens. | P-P A_abs | P-P A_s | PIT A_abs | "
        "PIT A_s |"
    )
    rows_a, rows_b = [], []

    def add(label, sigma_text, s):
        m = {c: _median(s[c]) for c in s.select_dtypes("number").columns}
        rows_a.append(
            f"| {label} | {sigma_text} | {_fmt(s.converged.mean(), '{:.2f}')} | "
            f"{_terminations(s)} | {_fmt(m.get('n_iter', np.nan), '{:.0f}')} | "
            f"{_fmt(m.get('n_fev', np.nan), '{:.0f}')} | "
            f"{_fmt(s.n_nonfinite.mean() if 'n_nonfinite' in s else np.nan, '{:.1f}')}"
            f" | {_fmt(m.get('train_objective', np.nan))} | "
            f"{_fmt(m.get('train_data', np.nan))} | "
            f"{_fmt(m.get('train_kl', np.nan))} | "
            f"{_fmt(m.get('fresh_objective', np.nan))} | "
            f"{_fmt(m.get('fresh_data', np.nan))} |"
        )
        rows_b.append(
            f"| {label} | {sigma_text} | {_fmt(m.get('rmse', np.nan))} | "
            f"{_fmt(m.get('coverage_95.45', np.nan), '{:.3f}')} | "
            f"{_fmt(m.get('coverage_99.9', np.nan), '{:.3f}')} | "
            f"{_fmt(m.get('width_95.45', np.nan))} | "
            f"{_fmt(m.get('interval_score', np.nan))} | "
            f"{_fmt(m.get('nll', np.nan))} | "
            f"{_fmt(m.get('nll_floor', np.nan), '{:.2f}')} | "
            f"{_fmt(m.get('zero_density', np.nan), '{:.2f}')} | "
            f"{_fmt(m.get('pp_area_abs', np.nan), '{:.3f}')} | "
            f"{_fmt(m.get('pp_area_signed', np.nan), '{:+.3f}')} | "
            f"{_fmt(m.get('pit_area_abs', np.nan), '{:.3f}')} | "
            f"{_fmt(m.get('pit_area_signed', np.nan), '{:+.3f}')} |"
        )

    for method in REFERENCES:
        s = _select(frame, name, n_units, method)
        if not s.empty:
            add(labels.display(method), "none", s)
    for objective, n_samples in CONFIGS:
        for sigma in sigmas:
            s = _select(frame, name, n_units, objective, n_samples, sigma)
            if not s.empty:
                add(config_name(objective, n_samples), f"{sigma:g}", s)
    title = f"\n## {PROBLEM_TITLES[name]}, N = {n_units} {unit}\n"
    return [
        title,
        "Objective and optimizer:\n",
        head_a,
        "|---" * 12 + "|",
        *rows_a,
        "\nParameter-only predictive scores:\n",
        head_b,
        "|---" * 14 + "|",
        *rows_b,
    ]


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

COLORS = {
    ("pacm", 1): "#9ecae1",
    ("pacm", 16): "#3182bd",
    ("pacm", 64): "#08306b",
    ("pac2t", 1): "#fdae6b",
    ("pac2t", 16): "#e6550d",
    ("pac2t", 64): "#7f2704",
    ("pvi", 1): "#74c476",
    ("pvi", 16): "#00441b",
}
STYLES = {"pacm": "-", "pac2t": "--", "pvi": "-."}
REF_STYLES = {
    "POPS ellipse": ("k", "-"),
    "Ellipse+EB": ("0.45", "--"),
    "Ellipse+PAC": ("#9e6b00", ":"),
    "Bayesian stacking": ("#c51b8a", "-."),
}


def plot(frame, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cells = [
        (name, n)
        for name in PROBLEMS
        if name in set(frame.problem)
        for n in sorted(frame[frame.problem == name].N.unique())
    ]
    keys = (
        ("train_objective", "Training objective [nats/datum]", "sym"),
        ("fresh_objective", "Objective on fresh draws [nats/datum]", "sym"),
        ("coverage_95.45", "Coverage of the 95.45% interval", "linear"),
        ("nll_floor", "Floored NLL [nats]", "linear"),
    )
    fig, axes = plt.subplots(
        len(cells), 4, figsize=(14, 2.75 * len(cells)), squeeze=False
    )
    sigmas = np.array(sorted(frame.sigma.dropna().unique(), reverse=True))
    for r, (name, n_units) in enumerate(cells):
        sub = frame[(frame.problem == name) & (frame.N == n_units)]
        for c, (key, title, scale) in enumerate(keys):
            ax = axes[r, c]
            for objective, n_samples in CONFIGS:
                s = sub[(sub.method == objective) & (sub.n_samples == n_samples)]
                if s.empty or key not in s:
                    continue
                stats = np.array(
                    [
                        calibration.central_interval(s[np.isclose(s.sigma, g)][key])
                        for g in sigmas
                    ]
                )
                conv = np.array(
                    [s[np.isclose(s.sigma, g)].converged.mean() for g in sigmas]
                )
                color = COLORS[(objective, n_samples)]
                ax.fill_between(
                    sigmas, stats[:, 1], stats[:, 2], color=color, alpha=0.12, lw=0
                )
                # PVI (no convergence test) sits below the L-BFGS methods.
                z = 1.5 if objective == "pvi" else 2.0
                ax.plot(
                    sigmas,
                    stats[:, 0],
                    ls=STYLES[objective],
                    color=color,
                    lw=1.4,
                    zorder=z,
                )
                full = conv >= 1.0
                ax.plot(
                    sigmas[full], stats[full, 0], "o", color=color, ms=3.5, zorder=z
                )
                # Hollow markers (drawn on top): not every repeat converged.
                ax.plot(
                    sigmas[~full],
                    stats[~full, 0],
                    "o",
                    mfc="white",
                    mec=color,
                    ms=4.5,
                    mew=1.1,
                    zorder=6,
                )
            for method in REFERENCES:
                s = sub[sub.method == method]
                if key not in s:
                    continue
                value = _median(s[key])
                if np.isfinite(value):
                    color, ls = REF_STYLES[method]
                    ax.axhline(value, color=color, ls=ls, lw=1.0)
            ax.set_xscale("log")
            ax.invert_xaxis()
            if scale == "sym":
                ax.set_yscale("symlog", linthresh=1.0)
                ref = sigmas[-1]
                s0 = sub[
                    np.isclose(sub.sigma, ref)
                    & (sub.method == "pacm")
                    & (sub.n_samples == 1)
                ]
                anchor = _median(s0.train_objective) if not s0.empty else np.nan
                if np.isfinite(anchor) and anchor > 0:
                    ax.plot(
                        sigmas,
                        anchor * (ref / sigmas) ** 2,
                        color="k",
                        ls=(0, (1, 2)),
                        lw=0.9,
                    )
                ax.set_ylim(bottom=-1.0)
            if key.startswith("coverage"):
                ax.axhline(0.9545, color="k", lw=0.6, ls=":")
                ax.set_ylim(-0.02, 1.02)
            if r == 0:
                ax.set_title(title, fontsize=9)
            if r == len(cells) - 1:
                ax.set_xlabel(
                    r"likelihood width $\sigma$ (target-std units)", fontsize=8
                )
            if c == 0:
                ax.set_ylabel(
                    f"{PROBLEM_TITLES[name]}\nN = {n_units} {UNIT_NAMES[name]}",
                    fontsize=8,
                )
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, lw=0.5)
    handles = [
        Line2D(
            [],
            [],
            color=COLORS[cfg],
            ls=STYLES[cfg[0]],
            marker="o",
            ms=3.5,
            label=config_name(*cfg, tex=True),
        )  # fmt: skip
        for cfg in CONFIGS
    ]
    handles += [
        Line2D([], [], color=col, ls=ls, label=f"{labels.display(m)} (no $\\sigma$)")
        for m, (col, ls) in REF_STYLES.items()
    ]
    handles += [
        Line2D([], [], color="k", ls=(0, (1, 2)), label=r"$\propto\sigma^{-2}$"),
        Line2D(
            [], [], ls="none", marker="o", mfc="white", mec="k",
            label="not every repeat converged",
        ),  # fmt: skip
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=0.6)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def normalize_terminations(frame):
    """Label scipy's "ABNORMAL" L-BFGS-B stop (a line search that could not
    lower the objective) as 'line_search' in rows written by older code."""
    if "termination_message" not in frame:
        return frame, False
    message = frame.termination_message.fillna("").astype(str).str.upper()
    fix = (frame.termination == "other") & message.str.contains("ABNORMAL")
    frame.loc[fix, "termination"] = "line_search"
    return frame, bool(fix.any())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--problems", default="quartic,burgers")
    parser.add_argument("--quartic-sizes", default="10,50")
    parser.add_argument("--burgers-sizes", default="8,24")
    parser.add_argument("--sigmas", default=",".join(f"{s:g}" for s in SIGMAS))
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--output-dir", default=None, help="tables and figure go here (tests)"
    )
    args = parser.parse_args()
    global OUT
    figure = HERE / "low_noise_objectives.png"
    if args.output_dir:
        OUT = Path(args.output_dir)
        figure = OUT / "low_noise_objectives.png"
    if args.plot_only:
        frame = pd.read_csv(OUT / "low_noise_objectives.csv")
        frame, changed = normalize_terminations(frame)
        if changed:
            frame.to_csv(OUT / "low_noise_objectives.csv", index=False)
    else:
        sizes = {
            "quartic": tuple(int(s) for s in args.quartic_sizes.split(",")),
            "burgers": tuple(int(s) for s in args.burgers_sizes.split(",")),
        }
        problems = {p: sizes[p] for p in args.problems.split(",")}
        sigmas = tuple(float(s) for s in args.sigmas.split(","))
        print(f"python {sys.version.split()[0]}, {args.workers} workers", flush=True)
        frame = run(problems, args.repeats, args.workers, sigmas)
    summarize(frame)
    plot(frame, figure)


if __name__ == "__main__":
    main()
