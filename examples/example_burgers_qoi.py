"""Propagation of parameter uncertainty to Burgers quantities of interest.

Each parameter draw of each method defines a complete POD surrogate field
``u(x) = mean_field(x) + F(x; nu, A, t) @ theta``, which is kept for the whole
field before two quantities of interest are evaluated on the 96-point grid:

- the viscous dissipation ``D = nu * int |du/dx|^2 dx``;
- the front steepness ``max_x (-du/dx)``.

(The front location itself is not informative here: with ``u_0 = A sin x``
the solution is odd about ``x = pi``, so the steepest descent is always at
``pi``.)

For held-out simulator cases the reference values come from the numerical
solver. We report the coverage and width of the exact empirical 95.45%
central interval of the propagated draws, over repeated training draws. No
scalar residual variance is turned into spatial noise for any method.

Writes ``generated/burgers_qoi.csv`` and ``.md`` and ``burgers_qoi.png``.
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
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import example_burgers as burgers  # noqa: E402
import example_burgers_pod as pod  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from comparisons import harness  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
CASES = (8, 16, 24, 40, 80)
N_QOI_CASES = 60
N_DRAWS = 2000
LEVEL = 0.9545
QOIS = ("dissipation", "steepness")


def qoi(fields, nu):
    """Dissipation and front steepness of fields with shape (..., n_grid)."""
    x = np.linspace(0.0, 2.0 * np.pi, burgers.N_GRID, endpoint=False)
    dx = x[1] - x[0]
    grad = (np.roll(fields, -1, axis=-1) - np.roll(fields, 1, axis=-1)) / (2 * dx)
    return nu * np.sum(grad * grad, axis=-1) * dx, np.max(-grad, axis=-1)


def reference_qois(cases):
    fields = np.array([burgers.burgers_solution(*case)[1] for case in cases])
    return qoi(fields, cases[:, 0])


def run_one(task):
    n_cases, seed, methods = task
    warnings.filterwarnings("ignore")
    problem = harness.burgers_problem(n_cases, seed)
    modes, mean_field = problem.meta["modes"], problem.meta["mean_field"]
    cases = problem.meta["test_cases"][:N_QOI_CASES]
    true_d, true_front = reference_qois(cases)
    x = np.linspace(0.0, 2.0 * np.pi, burgers.N_GRID, endpoint=False)
    rows = []
    for method in methods:
        coef, offset = harness.parameter_draws(method, problem, seed, N_DRAWS)
        cov = {q: [] for q in QOIS}
        width = {q: [] for q in QOIS}
        err = {q: [] for q in QOIS}
        for (nu, amp, time), d_ref, f_ref in zip(cases, true_d, true_front):
            F = pod.rom_features(
                np.full_like(x, nu),
                np.full_like(x, amp),
                np.full_like(x, time),
                x,
                modes,
            )
            fields = mean_field[None, :] + (F @ coef + offset[None, :]).T
            d, front = qoi(fields, nu)
            for name, values, ref in (
                ("dissipation", d, d_ref),
                ("steepness", front, f_ref),
            ):
                lo, hi = np.quantile(values, [0.5 * (1 - LEVEL), 0.5 * (1 + LEVEL)])
                cov[name].append(lo <= ref <= hi)
                scale = abs(ref)
                width[name].append((hi - lo) / scale)
                err[name].append((np.median(values) - ref) / scale)
        row = {"cases": n_cases, "seed": seed, "method": method}
        for name in QOIS:
            row[f"{name}_coverage"] = float(np.mean(cov[name]))
            row[f"{name}_width"] = float(np.median(width[name]))
            row[f"{name}_error"] = float(np.median(np.abs(err[name])))
        rows.append(row)
    return rows


def summarize(frame):
    lines = [
        "# Burgers quantities of interest\n",
        (
            f"Coverage and median width of the empirical 95.45% interval of {N_DRAWS} "
            f"propagated parameter draws on {N_QOI_CASES} held-out cases (widths "
            "relative to the reference value), medians "
            f"and central 90% intervals over {frame.seed.nunique()} training draws.\n"
        ),
    ]
    for name in QOIS:
        for metric in ("coverage", "width"):
            lines.append(f"\n## {name}: {metric}\n")
            lines.append("| method | " + " | ".join(str(c) for c in CASES) + " |")
            lines.append("|---" * (len(CASES) + 1) + "|")
            for method in harness.METHODS:
                cells = []
                for c in CASES:
                    v = frame[(frame.method == method) & (frame.cases == c)][
                        f"{name}_{metric}"
                    ]
                    lo, med, hi = np.quantile(v, [0.05, 0.5, 0.95])
                    cells.append(f"{med:.3f} [{lo:.3f}, {hi:.3f}]")
                lines.append(f"| {method} | " + " | ".join(cells) + " |")
    (OUT / "burgers_qoi.md").write_text("\n".join(lines) + "\n")


def plot(frame, output):
    import matplotlib.pyplot as plt
    from example_repeated_splits import STYLE

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.5))
    panels = (
        ("dissipation_coverage", "Dissipation: coverage"),
        ("dissipation_width", "Dissipation: rel. width"),
        ("steepness_coverage", "Front steepness: coverage"),
        ("steepness_width", "Front steepness: rel. width"),
    )
    for ax, (key, title) in zip(axes, panels):
        for method in harness.METHODS:
            g = frame[frame.method == method].groupby("cases")[key]
            q = np.array(
                [np.quantile(g.get_group(c), [0.05, 0.5, 0.95]) for c in CASES]
            )
            style = STYLE[method]
            ax.plot(CASES, q[:, 1], marker="o", ms=2.5, label=method, **style)
            ax.fill_between(
                CASES, q[:, 0], q[:, 2], color=style["color"], alpha=0.1, lw=0
            )
        ax.set_xscale("log")
        ax.set_xticks(CASES)
        ax.set_xticklabels([str(c) for c in CASES])
        ax.minorticks_off()
        if "coverage" in key:
            ax.axhline(LEVEL, color="k", lw=0.7, ls=":")
            ax.set_ylim(-0.02, 1.02)
        else:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("simulator cases N", fontsize=8)
        ax.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--methods", default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        frame = pd.read_csv(OUT / "burgers_qoi.csv")
    else:
        methods = tuple(args.methods.split(",")) if args.methods else harness.METHODS
        tasks = [(c, s, methods) for c in CASES for s in range(args.seeds)]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(run_one, tasks, chunksize=1):
                rows.extend(r)
        frame = pd.DataFrame(rows)
        if args.merge:
            old = pd.read_csv(OUT / "burgers_qoi.csv")
            frame = pd.concat(
                [old[~old.method.isin(methods)], frame], ignore_index=True
            )
        OUT.mkdir(exist_ok=True)
        frame.to_csv(OUT / "burgers_qoi.csv", index=False)
    summarize(frame)
    plot(frame, HERE / "burgers_qoi.png")


if __name__ == "__main__":
    main()
