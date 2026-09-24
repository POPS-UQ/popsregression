"""Low-noise study: PACm, PAC^2_T and PVI against the zero-noise POPS ellipse.

On the quartic surrogate of the deterministic oscillatory engine (P = 5), the
Gaussian working likelihood of the finite-noise objectives is given a
decreasing width ``sigma`` (in units of the training-target standard
deviation), at several Monte Carlo sample, pair or particle counts, over
paired training draws:

- PACm-Bayes (Morningstar et al. 2022) with ``m`` samples. With KL weight 1
  this is also predictive variational inference (Lai, Linero and Yao) with the
  Monte Carlo estimate ``log (1/m) sum_j p(y | theta_j)`` of its log score.
- PAC^2_T (Masegosa 2020), variational (``S`` pairs) and ensemble (``E``
  particles).
- PVI with the exact Gaussian predictive integral (``m -> infinity``) and
  KL weight 1.

The POPS ellipse, Ellipse+EB and Ellipse+PAC use the analytic zero-noise
pushforward and do not depend on ``sigma``; Bayesian stacking of degree 1-4
regressions is a further ``sigma``-free reference. The likelihood width is a
fitting device only: every score and interval uses the parameter-only
predictive (the particle ensemble has no density and gets no log loss).

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
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from comparisons import harness  # noqa: E402
from comparisons.low_noise_objectives import LowNoiseObjective  # noqa: E402
from comparisons.pvi import PredictiveVI  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
SIGMAS = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003)
CONFIGS = (
    ("pacm", 1),
    ("pacm", 4),
    ("pacm", 16),
    ("pacm", 64),
    ("pac2t", 16),
    ("pac2t_ensemble", 16),
    ("pvi_exact", 0),
)
LABEL = {
    "pacm": "PACm / PVI-MC (m={})",
    "pac2t": "PAC$^2_T$ var. (S={})",
    "pac2t_ensemble": "PAC$^2_T$ ens. (E={})",
    "pvi_exact": "PVI, exact predictive",
}
REFERENCES = ("POPS ellipse", "Ellipse+EB", "Ellipse+PAC", "Bayesian stacking")


def _predictive(name, model, X_test, fit_time, info):
    intervals = {
        level: model.predict_interval(X_test, level=level) for level in harness.LEVELS
    }
    try:
        model.predict_logpdf(X_test[:1], model.predict(X_test[:1]))
        logpdf = lambda y: model.predict_logpdf(X_test, y)  # noqa: E731
    except ValueError:  # discrete particle predictive: no density
        logpdf = None
    return harness.Predictive(
        name, model.predict(X_test), intervals, logpdf, fit_time, info
    )


def run_one(task):
    n_train, repeat = task
    warnings.filterwarnings("ignore")
    problem = harness.quartic_problem(n_train, 500 + repeat)
    X, y = problem.X_train, problem.y_train
    base = dict(N=n_train, repeat=repeat)
    rows = []
    for method in REFERENCES:
        pred = harness.fit_predictive(method, problem, repeat)
        rows.append(
            {
                **base,
                "method": method,
                "sigma": np.nan,
                "n_samples": np.nan,
                "converged": True,
                "n_nonfinite": 0,
                **harness.evaluate(pred, problem.y_test, problem.y_bounds),
            }
        )
    for objective, n_samples in CONFIGS:
        for sigma in SIGMAS:
            start = time.perf_counter()
            if objective == "pvi_exact":
                model = PredictiveVI(sigma=sigma, lam=1.0).fit(X, y)
                info = dict(converged=model.converged_, n_nonfinite=0)
            else:
                model = LowNoiseObjective(
                    objective, sigma=sigma, n_samples=n_samples, random_state=repeat
                ).fit(X, y)
                info = dict(converged=model.converged_, n_nonfinite=model.n_nonfinite_)
            pred = _predictive(
                objective, model, problem.X_test, time.perf_counter() - start, info
            )
            rows.append(
                {
                    **base,
                    "method": objective,
                    "sigma": sigma,
                    "n_samples": n_samples,
                    **info,
                    **harness.evaluate(pred, problem.y_test, problem.y_bounds),
                }
            )
    return rows


def run(train_sizes, repeats, workers):
    tasks = [(n, r) for n in train_sizes for r in range(repeats)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(run_one, tasks, chunksize=1):
            rows.extend(result)
    frame = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    frame.to_csv(OUT / "low_noise_objectives.csv", index=False)
    return frame


def _med(series, fmt):
    v = np.asarray(series, dtype=float)
    v = v[np.isfinite(v)]
    return fmt.format(np.median(v)) if v.size else "--"


def summarize(frame):
    lines = [
        "# Low-noise study (quartic, P = 5)\n",
        (
            f"Medians over {frame.repeat.nunique()} paired training draws, 2000 test "
            "points. `sigma` is the likelihood width in target-std units; `raw NLL` is "
            "the mean parameter-only negative log density (inf if a test target has "
            f"zero density); `floored NLL` uses beta = {harness.FLOOR_BETA} on the "
            "declared output interval; `conv.` the fraction of L-BFGS runs reporting "
            "convergence; `non-finite` the mean number of non-finite objective "
            "evaluations per fit.\n"
        ),
    ]
    for n in sorted(frame.N.unique()):
        sub = frame[frame.N == n]
        lines.append(f"\n## N = {n}\n")
        lines.append(
            "| method | sigma | conv. | non-finite | raw NLL | floored NLL | "
            "cov. 95.45% | interval score | time [s] |"
        )
        lines.append("|---" * 9 + "|")
        for method in REFERENCES:
            s = sub[sub.method == method]
            lines.append(
                f"| {method} | zero-noise | 1.00 | 0 | {_med(s.nll, '{:.3g}')} | "
                f"{_med(s.nll_floor, '{:.2f}')} | "
                f"{_med(s['coverage_95.45'], '{:.3f}')} | "
                f"{_med(s.interval_score, '{:.3g}')} | "
                f"{_med(s.fit_time, '{:.2g}')} |"
            )
        for objective, n_samples in CONFIGS:
            for sigma in SIGMAS:
                s = sub[
                    (sub.method == objective)
                    & (sub.sigma == sigma)
                    & (sub.n_samples == n_samples)
                ]
                name = LABEL[objective].format(n_samples).replace("$", "")
                lines.append(
                    f"| {name} | {sigma:g} | {s.converged.mean():.2f} |"
                    f" {s.n_nonfinite.mean():.1f} | {_med(s.nll, '{:.3g}')} |"
                    f" {_med(s.nll_floor, '{:.2f}')} |"
                    f" {_med(s['coverage_95.45'], '{:.3f}')} |"
                    f" {_med(s.interval_score, '{:.3g}')} |"
                    f" {_med(s.fit_time, '{:.2g}')} |"
                )
    (OUT / "low_noise_objectives.md").write_text("\n".join(lines) + "\n")


def plot(frame, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    sizes = sorted(frame.N.unique())
    fig, axes = plt.subplots(
        len(sizes), 3, figsize=(10.5, 2.7 * len(sizes)), squeeze=False
    )
    styles = {"pacm": "-", "pac2t": "--", "pac2t_ensemble": ":", "pvi_exact": "-"}
    colors = {
        ("pacm", 1): "C0",
        ("pacm", 4): "C1",
        ("pacm", 16): "C2",
        ("pacm", 64): "C3",
        ("pac2t", 16): "C4",
        ("pac2t_ensemble", 16): "C5",
        ("pvi_exact", 0): "C9",
    }
    ref_colors = {
        "POPS ellipse": "C2",
        "Ellipse+EB": "0.35",
        "Ellipse+PAC": "C1",
        "Bayesian stacking": "C3",
    }
    keys = (
        ("nll_floor", "Floored test NLL [nats]", "linear"),
        ("coverage_95.45", "Coverage of 95.45% interval", "linear"),
        ("interval_score", "Interval score (95.45%)", "log"),
    )
    for r, n in enumerate(sizes):
        sub = frame[frame.N == n]
        for c, (key, title, scale) in enumerate(keys):
            ax = axes[r, c]
            for objective, n_samples in CONFIGS:
                s = sub[(sub.method == objective) & (sub.n_samples == n_samples)]
                med = s.groupby("sigma")[key].median().reindex(SIGMAS)
                ax.plot(
                    SIGMAS,
                    med.values,
                    ls=styles[objective],
                    color=colors[(objective, n_samples)],
                    lw=1.4,
                    marker="o",
                    ms=2,
                    label=LABEL[objective].format(n_samples),
                )
            for method in REFERENCES:
                value = sub[sub.method == method][key].median()
                ax.axhline(
                    value,
                    color=ref_colors[method],
                    lw=1.1,
                    ls="-.",
                    label=f"{method} (zero noise)",
                )
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_yscale(scale)
            if scale == "log":
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            if key.startswith("coverage"):
                ax.axhline(0.9545, color="k", lw=0.7, ls=":")
                ax.set_ylim(-0.02, 1.02)
            if r == 0:
                ax.set_title(title, fontsize=9)
            if r == len(sizes) - 1:
                ax.set_xlabel(
                    r"likelihood width $\sigma$ (target-std units)", fontsize=8
                )
            if c == 0:
                ax.set_ylabel(f"N = {n}", fontsize=9)
            ax.tick_params(labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.6)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--train-sizes", default="10,50")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        frame = pd.read_csv(OUT / "low_noise_objectives.csv")
    else:
        sizes = [int(s) for s in args.train_sizes.split(",")]
        frame = run(sizes, args.repeats, args.workers)
    summarize(frame)
    plot(frame, HERE / "low_noise_objectives.png")


if __name__ == "__main__":
    main()
