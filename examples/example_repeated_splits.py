"""Repeated-split predictive comparison (paper Fig. ``fig:certificate-reserved``).

Every method is refitted on independent training draws at several data sizes
for the quartic, Burgers and ACE problems, and evaluated on a fixed
independent test set with parameter-only predictives (no residual-noise
term for any method). Outputs, under ``examples/generated/``:

- ``repeated_splits.csv``: one row per (problem, size, seed, method), with
  every metric of :func:`comparisons.harness.evaluate`, fit time, failures
  and the Ellipse+PAC bound decomposition;
- ``repeated_splits_summary.md`` / ``.tex``: medians and central 90%
  intervals over seeds;
- ``repeated_splits.png``: learning curves (medians and central 90% bands).

Usage::

    python example_repeated_splits.py --seeds 30 --workers 16
    python example_repeated_splits.py --plot-only
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
import traceback  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from comparisons import harness  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
SIZES = {
    "quartic": (10, 20, 50, 100, 300),  # training points; P = 5
    "burgers": (8, 16, 24, 40, 80),  # simulator cases (3 points each); P = 8
    "ace": (1.5, 3.0, 6.0, 20.0),  # N / P with P = 35
}
PROBLEM_TITLES = {
    "quartic": "Quartic surrogate ($P=5$)",
    "burgers": "Burgers POD emulator ($P=8$)",
    "ace": "ACE energies ($P=36$)",
}
BOUND_KEYS = (
    "bound",
    "trivial_bound",
    "lam",
    "empirical",
    "monte_carlo",
    "kl",
    "complexity",
    "concentration",
    "mixture_empirical",
    "outside_support_fraction",
)


def make_problem(name, size, seed):
    if name == "quartic":
        return harness.quartic_problem(int(size), seed)
    if name == "burgers":
        return harness.burgers_problem(int(size), seed)
    return harness.ace_problem(float(size), seed)


def run_one(task):
    name, size, seed, methods = task
    warnings.filterwarnings("ignore")
    problem = make_problem(name, size, seed)
    rows = []
    for method in methods:
        row = {
            "problem": name,
            "size": size,
            "seed": seed,
            "method": method,
            "n_rows": problem.y_train.size,
            "n_units": problem.n_units,
            "n_params": problem.n_params,
            "units_per_param": problem.n_units / problem.n_params,
            "failed": False,
        }
        try:
            pred = harness.fit_predictive(method, problem, seed)
            row.update(harness.evaluate(pred, problem.y_test, problem.y_bounds))
            for key in BOUND_KEYS:
                if key in pred.info:
                    row[key] = pred.info[key]
            if method == "Ellipse+PAC":
                # The certified target: floored NLL with the PAC path's beta.
                beta = pred.info["model"].certificate_.beta
                row["pac_test_risk"] = harness.evaluate(
                    pred, problem.y_test, problem.y_bounds, floor_beta=beta
                )["nll_floor"]
            if "pvi_lam" in pred.info:
                row["pvi_lam"] = pred.info["pvi_lam"]
                row["converged"] = pred.info["converged"]
        except Exception:  # record failures instead of dropping them
            row["failed"] = True
            row["error"] = traceback.format_exc(limit=1).strip().splitlines()[-1]
        rows.append(row)
    return rows


def run(seeds, workers, problems, methods):
    tasks = [
        (name, size, seed, methods)
        for name in problems
        for size in SIZES[name]
        for seed in range(seeds)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(run_one, tasks, chunksize=1)):
            rows.extend(result)
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(tasks)} tasks", flush=True)
    frame = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    frame.to_csv(OUT / "repeated_splits.csv", index=False)
    return frame


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

SUMMARY_METRICS = (
    ("coverage_95.45", "cov. 95.45%", "{:.3f}"),
    ("coverage_99.9", "cov. 99.9%", "{:.3f}"),
    ("width_95.45", "width 95.45%", "{:.3g}"),
    ("interval_score", "interval score", "{:.3g}"),
    ("nll_floor", "floored NLL", "{:.2f}"),
    ("zero_density", "zero density", "{:.3f}"),
    ("fit_time", "fit time [s]", "{:.2g}"),
)


def _median_interval(values, fmt):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "--"
    lo, med, hi = np.quantile(values, [0.05, 0.5, 0.95])
    return f"{fmt.format(med)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def summarize(frame):
    lines_md, lines_tex = [], []
    methods = [
        m for m in harness.METHODS + harness.EXTRA_METHODS if m in set(frame.method)
    ]
    n_seeds = frame.seed.nunique()
    lines_md.append(
        "# Repeated-split comparison\n\nMedians and central 90% intervals over "
        f"{n_seeds} independent training draws (fixed test sets). All intervals "
        "are exact central intervals of parameter-only predictives. The floored "
        f"NLL uses beta = {harness.FLOOR_BETA} on each problem's declared output "
        "interval; the POPS hypercube has no density.\n"
    )
    for name in SIZES:
        sub = frame[frame.problem == name]
        if sub.empty:
            continue
        lines_md.append(f"\n## {name}\n")
        for key, label, fmt in SUMMARY_METRICS:
            lines_md.append(f"\n### {label}\n")
            header = "| method | " + " | ".join(f"{s:g}" for s in SIZES[name]) + " |"
            lines_md.append(header)
            lines_md.append("|" + "---|" * (len(SIZES[name]) + 1))
            tex_rows = []
            for method in methods:
                cells = []
                for size in SIZES[name]:
                    cell = sub[(sub.method == method) & (sub["size"] == size)]
                    cells.append(_median_interval(cell[key], fmt))
                lines_md.append(f"| {method} | " + " | ".join(cells) + " |")
                tex_rows.append(
                    f"{method} & "
                    + " & ".join(
                        c.replace("[", "{\\scriptsize[").replace("]", "]}")
                        for c in cells
                    )
                    + r" \\"
                )
            lines_tex.append(
                f"% {name}: {label}\n\\begin{{tabular}}{{l"
                + "c" * len(SIZES[name])
                + "}\n\\toprule\nMethod & "
                + " & ".join(f"{s:g}" for s in SIZES[name])
                + r" \\"
                + "\n\\midrule\n"
                + "\n".join(tex_rows)
                + "\n\\bottomrule\n\\end{tabular}\n"
            )
        failures = sub.groupby("method").failed.sum()
        lines_md.append(
            "\nFailed fits: "
            + ", ".join(f"{m}: {int(failures.get(m, 0))}" for m in methods)
            + "\n"
        )

        pac = sub[sub.method == "Ellipse+PAC"]
        if not pac.empty and "bound" in pac:
            lines_md.append("\n### Ellipse+PAC bound (floor-contaminated log risk)\n")
            lines_md.append(
                "| size | bound | trivial | test floored NLL (PAC beta) "
                "| non-vacuous | lambda/N1 |"
            )
            lines_md.append("|---|---|---|---|---|---|")
            for size in SIZES[name]:
                cell = pac[pac["size"] == size]
                nonvac = np.mean(cell.bound < cell.trivial_bound)
                lines_md.append(
                    f"| {size:g} | {_median_interval(cell.bound, '{:.2f}')} | "
                    f"{cell.trivial_bound.median():.2f} | "
                    f"{_median_interval(cell.pac_test_risk, '{:.2f}')} | "
                    f"{nonvac:.2f} | "
                    f"{_median_interval(cell.lam / (cell.n_units / 2), '{:.3f}')} |"
                )
    (OUT / "repeated_splits_summary.md").write_text("\n".join(lines_md) + "\n")
    (OUT / "repeated_splits_summary.tex").write_text("\n".join(lines_tex))


HEADLINE = (
    ("coverage_95.45", "cov.", "{:.2f}"),
    ("interval_score", "IS", "{:.3g}"),
    ("nll_floor", "NLL$_b$", "{:.2f}"),
)


def headline(frame):
    """Compact main-text table: smallest and largest size of each problem."""
    methods = [m for m in harness.METHODS if m in set(frame.method)]
    md, tex = ["# Headline table (medians over training draws)\n"], []
    for name in SIZES:
        sub = frame[frame.problem == name]
        if sub.empty:
            continue
        sizes = (SIZES[name][0], SIZES[name][-1])
        ratios = [sub[sub["size"] == s].units_per_param.iloc[0] for s in sizes]
        md.append(f"\n## {name}\n")
        md.append(
            "| method | "
            + " | ".join(
                f"{label} ({r:.2g} u/P)" for r in ratios for _, label, _ in HEADLINE
            )
            + " |"
        )
        md.append("|---" * (1 + 2 * len(HEADLINE)) + "|")
        rows = []
        for method in methods:
            cells = []
            for size in sizes:
                cell = sub[(sub.method == method) & (sub["size"] == size)]
                for key, _, fmt in HEADLINE:
                    v = cell[key].to_numpy(dtype=float)
                    v = v[np.isfinite(v)]
                    cells.append(fmt.format(np.median(v)) if v.size else "--")
            md.append(f"| {method} | " + " | ".join(cells) + " |")
            rows.append(f"{method} & " + " & ".join(cells) + r" \\")
        tex.append(
            f"% {name}: units per parameter {ratios[0]:.2g} and {ratios[1]:.2g}\n"
            "\\begin{tabular}{l"
            + "c" * (2 * len(HEADLINE))
            + "}\n\\toprule\n"
            + " & "
            + " & ".join(
                f"\\multicolumn{{{len(HEADLINE)}}}{{c}}{{{r:.2g} units/param}}"
                for r in ratios
            )
            + r" \\"
            + "\n"
            + "Method & "
            + " & ".join(label for _ in ratios for _, label, _ in HEADLINE)
            + r" \\"
            + "\n"
            + "\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )
    (OUT / "repeated_splits_headline.md").write_text("\n".join(md) + "\n")
    (OUT / "repeated_splits_headline.tex").write_text("\n".join(tex))


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

STYLE = {
    "Bayesian ridge": dict(color="C0", ls="-"),
    "POPS hypercube": dict(color="C8", ls="-"),
    "POPS ellipse": dict(color="C2", ls="-"),
    "Ellipse+EB": dict(color="0.35", ls="-"),
    "Ellipse+PAC": dict(color="C1", ls="-", lw=2.2),
    "Bayesian stacking": dict(color="C3", ls="--"),
    "PVI": dict(color="C4", ls="--"),
    "PACm": dict(color="C5", ls="--"),
    "PAC2-T": dict(color="C6", ls="--"),
}


def plot(frame, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (
        FixedFormatter,
        FixedLocator,
        FuncFormatter,
        NullFormatter,
        NullLocator,
    )

    columns = (
        ("coverage_95.45", "Coverage of 95.45% interval", "linear"),
        ("interval_score", "Interval score (95.45%)", "log"),
        ("nll_floor", "Floored test NLL [nats]", "linear"),
    )
    problems = [p for p in SIZES if p in set(frame.problem)]
    fig, axes = plt.subplots(
        len(problems), 3, figsize=(10.5, 2.55 * len(problems)), squeeze=False
    )
    methods = [m for m in harness.METHODS if m in set(frame.method)]
    for r, name in enumerate(problems):
        sub = frame[(frame.problem == name) & ~frame.failed]
        for c, (key, title, scale) in enumerate(columns):
            ax = axes[r, c]
            for method in methods:
                cell = sub[sub.method == method]
                if cell[key].notna().sum() == 0:
                    continue
                g = cell.groupby("units_per_param")[key]
                x = np.array(sorted(g.groups))
                q = np.array(
                    [np.nanquantile(g.get_group(v), [0.05, 0.5, 0.95]) for v in x]
                )
                style = STYLE[method]
                ax.plot(x, q[:, 1], label=method, marker="o", ms=2.5, **style)
                ax.fill_between(
                    x, q[:, 0], q[:, 2], color=style["color"], alpha=0.12, lw=0
                )
            ax.set_xscale("log")
            ax.set_yscale(scale)
            ticks = sorted(set(sub.units_per_param.round(3)))
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([f"{t:.2g}" for t in ticks]))
            ax.xaxis.set_minor_locator(NullLocator())
            if scale == "log":
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            if key.startswith("coverage"):
                ax.axhline(0.9545, color="k", lw=0.8, ls=":")
                ax.set_ylim(-0.02, 1.02)
            if r == 0:
                ax.set_title(title, fontsize=9)
            if r == len(problems) - 1:
                ax.set_xlabel("independent units per parameter", fontsize=8)
            if c == 0:
                ax.set_ylabel(PROBLEM_TITLES[name], fontsize=8)
            ax.tick_params(labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.6, w_pad=0.8)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    parser.add_argument("--problems", default="quartic,burgers,ace")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        frame = pd.read_csv(OUT / "repeated_splits.csv")
    else:
        frame = run(
            args.seeds,
            args.workers,
            args.problems.split(","),
            harness.METHODS + harness.EXTRA_METHODS,
        )
    summarize(frame)
    headline(frame)
    plot(frame, HERE / "repeated_splits.png")


if __name__ == "__main__":
    main()
