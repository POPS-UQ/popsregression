"""Runtime and peak memory against P, N and the ellipsoid rank r.

Times ``BayesianRidge``, the POPS hypercube, POPS Ellipse, POPS Ellipse+EB and
POPS Ellipse+PAC on

- the full 267-feature ACE energy design (no PCA projection), and
- a synthetic misspecified linear problem with ``P`` up to 2000 features,
  ``N = 4 P`` rows (and a fixed-``P`` sweep over ``N`` and ``r``).

Peak memory is the ``tracemalloc`` peak of a separate, identical fit (NumPy
array buffers are traced), reported relative to the size of the design
matrix. BLAS threading is left at its default.

Writes ``generated/scaling.csv`` and ``.md`` and ``scaling.png``.
"""

import argparse
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from comparisons import harness
from comparisons.labels import display
from sklearn.linear_model import BayesianRidge

from popsregression import POPSEllipseRegression, POPSRegression

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
METHODS = (
    "Bayesian ridge",
    "POPS hypercube",
    "POPS ellipse",
    "Ellipse+EB",
    "Ellipse+PAC",
)


def synthetic(n, p, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p) / np.sqrt(p)
    w = rng.randn(p)
    u = X @ rng.randn(p)
    y = X @ w + 0.3 * np.sin(3.0 * u)  # smooth, persistent misspecification
    bound = float(np.abs(w).sum() / np.sqrt(p) * 6 + 1.0)
    return X, y, (-bound, bound)


def make(method, rank, y_bounds):
    if method == "Bayesian ridge":
        return BayesianRidge(fit_intercept=False)
    if method == "POPS hypercube":
        return POPSRegression(random_state=0)
    reg = {"POPS ellipse": None, "Ellipse+EB": "empirical-bayes", "Ellipse+PAC": "PAC"}[
        method
    ]
    kwargs = {"y_bounds": y_bounds} if reg == "PAC" else {}
    return POPSEllipseRegression(
        regularization=reg, rank=rank, random_state=0, **kwargs
    )


def measure(method, X, y, rank, y_bounds, memory):
    model = make(method, rank, y_bounds)
    start = time.perf_counter()
    model.fit(X, y)
    seconds = time.perf_counter() - start
    peak = np.nan
    if memory:
        model = make(method, rank, y_bounds)
        tracemalloc.start()
        model.fit(X, y)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return seconds, peak


def run(max_p, memory):
    warnings.filterwarnings("ignore")
    rows = []
    data = harness._ace_data()
    X, y = data["A_train_E"], data["y_train_E"]
    for method in METHODS:
        seconds, peak = measure(method, X, y, 32, harness.ACE_Y_BOUNDS, memory)
        rows.append(
            dict(
                study="ACE full design",
                method=method,
                N=X.shape[0],
                P=X.shape[1],
                rank=32,
                seconds=seconds,
                peak_over_design=peak / X.nbytes,
            )
        )
        print(rows[-1], flush=True)
    for p in [p for p in (50, 100, 250, 500, 1000, 2000) if p <= max_p]:
        X, y, bounds = synthetic(4 * p, p)
        for method in METHODS:
            seconds, peak = measure(method, X, y, 32, bounds, memory and p <= 1000)
            rows.append(
                dict(
                    study="P sweep (N = 4P, r = 32)",
                    method=method,
                    N=4 * p,
                    P=p,
                    rank=32,
                    seconds=seconds,
                    peak_over_design=peak / X.nbytes,
                )
            )
            print(rows[-1], flush=True)
    for n in (500, 1000, 2000, 4000, 8000):
        X, y, bounds = synthetic(n, 250)
        for method in METHODS:
            seconds, _ = measure(method, X, y, 32, bounds, False)
            rows.append(
                dict(
                    study="N sweep (P = 250, r = 32)",
                    method=method,
                    N=n,
                    P=250,
                    rank=32,
                    seconds=seconds,
                    peak_over_design=np.nan,
                )
            )
    for rank in (4, 8, 16, 32, 64, 128):
        X, y, bounds = synthetic(1000, 250)
        for method in METHODS[2:]:
            seconds, _ = measure(method, X, y, rank, bounds, False)
            rows.append(
                dict(
                    study="rank sweep (N = 1000, P = 250)",
                    method=method,
                    N=1000,
                    P=250,
                    rank=rank,
                    seconds=seconds,
                    peak_over_design=np.nan,
                )
            )
    frame = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    frame.to_csv(OUT / "scaling.csv", index=False)
    return frame


def summarize(frame):
    lines = [
        "# Runtime and memory\n",
        (
            "Single fits, default BLAS threading; `peak/design` is the tracemalloc "
            "peak divided by the bytes of the design matrix.\n"
        ),
    ]
    for study, sub in frame.groupby("study", sort=False):
        lines.append(f"\n## {study}\n")
        lines.append("| method | N | P | rank | seconds | peak/design |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r.method} | {r.N} | {r.P} | {r['rank']} | "
                f"{r.seconds:.3g} | {r.peak_over_design:.2f} |"
            )
    (OUT / "scaling.md").write_text("\n".join(lines) + "\n")


def plot(frame, output):
    import matplotlib.pyplot as plt
    from example_repeated_splits import STYLE
    from matplotlib.ticker import (
        FixedFormatter,
        FixedLocator,
        FuncFormatter,
        NullFormatter,
        NullLocator,
    )

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.7))
    for ax, (study, xkey) in zip(
        axes,
        (
            ("P sweep (N = 4P, r = 32)", "P"),
            ("N sweep (P = 250, r = 32)", "N"),
            ("rank sweep (N = 1000, P = 250)", "rank"),
        ),
    ):
        sub = frame[frame.study == study]
        for method in METHODS:
            s = sub[sub.method == method]
            if s.empty:
                continue
            ax.plot(
                s[xkey],
                s.seconds,
                marker="o",
                ms=3,
                label=display(method),
                **STYLE[method],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = sorted(set(sub[xkey]))
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(study, fontsize=9)
        ax.set_xlabel(xkey, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("fit time [s]", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-p", type=int, default=2000)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        frame = pd.read_csv(OUT / "scaling.csv")
    else:
        frame = run(args.max_p, not args.no_memory)
    summarize(frame)
    plot(frame, HERE / "scaling.png")


if __name__ == "__main__":
    main()
