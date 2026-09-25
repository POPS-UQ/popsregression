"""Repeated-split predictive comparison (paper Fig. ``fig:certificate-reserved``).

Every method is refitted on independent training draws at several data sizes
for the quartic, Burgers and ACE problems, and evaluated on a fixed
independent test set with parameter-only predictives (no residual-noise
term for any method). Outputs, under ``examples/generated/``:

- ``repeated_splits.csv``: one row per (problem, size, seed, method), with
  every metric of :func:`comparisons.harness.evaluate` (including the
  calibration areas), fit time, optimizer termination records, failures
  and the Ellipse+PAC bound decomposition;
- ``predictions/<problem>_<size>_<seed>.npz``: per-test-point predictions of
  every method (mean, interval ends, log density, PIT value ``F(y)``) and
  its pooled-error P-P curve, for recomputing any metric later;
- ``repeated_splits_summary.md`` / ``.tex``: medians and central 90%
  intervals over seeds;
- ``repeated_splits_calibration.md`` / ``.tex``: calibration areas;
- ``repeated_splits.png``: learning curves (medians and central 90% bands).

Fits whose optimizer did not report convergence are kept and flagged
(``converged`` column; ``*`` in the tables, hollow markers in the figure);
failed fits are kept as rows with ``failed = True``.

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
from comparisons import calibration, harness  # noqa: E402
from comparisons.labels import ABBREVIATIONS, display  # noqa: E402

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
    "continuous_bound",
    "n_cert_units",
    "moment_constant",
    "n_stored",
    "failure_probability",
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


OPTIMIZER_KEYS = (
    "pvi_lam",
    "converged",
    "n_nonfinite",
    "n_iter",
    "n_fev",
    "termination_status",
    "termination_message",
    "max_iter",
    "max_fun",
    "n_skipped",
)
PREDICTIONS = OUT / "predictions"


def run_one(task):
    name, size, seed, methods = task
    warnings.filterwarnings("ignore")
    problem = make_problem(name, size, seed)
    weights = calibration.group_weights(problem.test_groups)
    rows, arrays = [], {"y_test": problem.y_test}
    if problem.test_groups is not None:
        arrays["test_groups"] = problem.test_groups
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
            record = {}
            row.update(
                harness.evaluate(
                    pred,
                    problem.y_test,
                    problem.y_bounds,
                    weights=weights,
                    record=record,
                )
            )
            for key, value in record.items():
                if value is not None:
                    arrays[f"{method}/{key}"] = np.asarray(value, dtype=np.float32)
            for key in BOUND_KEYS:
                if key in pred.info:
                    row[key] = pred.info[key]
            if method.startswith("Ellipse+PAC"):
                # The certified target: floored NLL of the returned (stored
                # finite-mixture) predictive with the PAC path's beta.
                beta = pred.info["model"].certificate_.beta
                row["pac_test_risk"] = harness.evaluate(
                    pred, problem.y_test, problem.y_bounds, floor_beta=beta
                )["nll_floor"]
            for key in OPTIMIZER_KEYS:
                if key in pred.info:
                    row[key] = pred.info[key]
        except Exception:  # record failures instead of dropping them
            row["failed"] = True
            row["error"] = traceback.format_exc(limit=1).strip().splitlines()[-1]
        rows.append(row)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS / f"{name}_{size:g}_{seed}.npz"
    if path.exists():  # keep the saved arrays of methods not rerun here
        with np.load(path) as old:
            rerun = tuple(f"{m}/" for m in methods)
            kept = {k: old[k] for k in old.files if not k.startswith(rerun)}
        arrays = {**kept, **arrays}
    np.savez_compressed(path, **arrays)
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
    ("coverage_95.45", "coverage of the 95.45% central interval", "{:.3f}"),
    ("coverage_99.9", "coverage of the 99.9% central interval", "{:.3f}"),
    ("width_95.45", "mean width of the 95.45% central interval", "{:.3g}"),
    ("interval_score", "interval score (IS) of the 95.45% interval", "{:.3g}"),
    ("nll_floor", "floored negative log predictive density (NLL, b = 0.01)", "{:.2f}"),
    ("zero_density", "fraction of test targets with zero predictive density", "{:.3f}"),
    ("pp_area_abs", "unsigned calibration area A_abs (pooled |error| P-P)", "{:.3f}"),
    ("pp_area_signed", "signed calibration area A_s (pooled |error| P-P)", "{:+.3f}"),
    ("pit_area_abs", "unsigned PIT calibration area (input-conditional)", "{:.3f}"),
    ("fit_time", "fit time [s]", "{:.2g}"),
)
CALIBRATION_METRICS = SUMMARY_METRICS[6:9] + (
    ("pit_area_signed", "signed PIT calibration area (input-conditional)", "{:+.3f}"),
)
# Methods whose fits report optimizer convergence; unconverged cells get '*'.
ITERATIVE = ("PVI", "PVI (lamb=1)", "PACm", "PAC2-T")
CAPTION = (
    "Medians and central 90% intervals over {n} independent training draws "
    "(fixed test sets). Every interval is an exact central interval of the "
    "method's parameter-only predictive (no residual-noise term). "
    "IS: interval score (Gneiting and Raftery) of the 95.45% interval, lower "
    "is better. NLL: negative log predictive density, floored as "
    "-log((1 - b) p + b / R_y) with b = {b} on the declared output interval "
    "of width R_y (a comparison score; the PAC certificate uses its own "
    "beta = 0.02). A_abs = int |C(u) - u| du and A_s = int (C(u) - u) du "
    "for the pooled absolute-error P-P curve C (smaller A_abs is better; "
    "A_s > 0 means net over-confidence, but A_s = 0 does not imply "
    "agreement). The PIT areas use u_i = F_i(y_i) instead and measure "
    "input-conditional calibration. Burgers test points are weighted so "
    "every simulator case counts equally in the areas. The POPS hypercube "
    "has no density (no NLL, no PIT from a density: its PIT uses its "
    "draws). '*' marks cells where at least one fit did not report "
    "optimizer convergence; those iterates are kept, not dropped. "
    "Abbreviations: {abbrev}."
)


def _abbreviations():
    keys = ("POPS", "EB", "PAC", "PACm", "PAC$^2_T$", "PVI", "IS", "NLL")
    return "; ".join(f"{k} = {ABBREVIATIONS[k]}" for k in keys)


def _median_interval(values, fmt):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "--"
    lo, med, hi = np.quantile(values, [0.05, 0.5, 0.95])
    return f"{fmt.format(med)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def _unconverged(cell):
    """True if any fit in ``cell`` reported non-convergence."""
    if "converged" not in cell or cell.method.iloc[0] not in ITERATIVE:
        return False
    flags = cell.converged.dropna()
    return bool(len(flags) and not flags.astype(bool).all())


def summarize(frame, metrics=SUMMARY_METRICS, stem="repeated_splits_summary"):
    lines_md, lines_tex = [], []
    methods = [
        m for m in harness.METHODS + harness.EXTRA_METHODS if m in set(frame.method)
    ]
    n_seeds = frame.seed.nunique()
    caption = CAPTION.format(n=n_seeds, b=harness.FLOOR_BETA, abbrev=_abbreviations())
    lines_md.append(f"# Repeated-split comparison\n\n{caption}\n")
    lines_tex.append("% " + caption + "\n")
    for name in SIZES:
        sub = frame[frame.problem == name]
        if sub.empty:
            continue
        lines_md.append(f"\n## {name}\n")
        for key, label, fmt in metrics:
            if key not in sub:
                continue
            lines_md.append(f"\n### {label}\n")
            header = "| method | " + " | ".join(f"{s:g}" for s in SIZES[name]) + " |"
            lines_md.append(header)
            lines_md.append("|" + "---|" * (len(SIZES[name]) + 1))
            tex_rows = []
            for method in methods:
                cells = []
                for size in SIZES[name]:
                    cell = sub[(sub.method == method) & (sub["size"] == size)]
                    text = _median_interval(cell[key], fmt)
                    if not cell.empty and _unconverged(cell):
                        text += "*"
                    cells.append(text)
                label_m = display(method)
                lines_md.append(f"| {label_m} | " + " | ".join(cells) + " |")
                tex_rows.append(
                    f"{label_m} & "
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
        if metrics is not SUMMARY_METRICS:
            continue
        failures = sub.groupby("method").failed.sum()
        lines_md.append(
            "\nFailed fits: "
            + ", ".join(f"{display(m)}: {int(failures.get(m, 0))}" for m in methods)
            + "\n"
        )
        lines_md.extend(_convergence_lines(sub))
        lines_md.extend(_bound_lines(sub, name))
    (OUT / f"{stem}.md").write_text("\n".join(lines_md) + "\n")
    (OUT / f"{stem}.tex").write_text("\n".join(lines_tex))


def _convergence_lines(sub):
    """Termination counts of the iterative comparators, per size."""
    rows = sub[sub.method.isin(ITERATIVE)]
    if rows.empty or "converged" not in rows:
        return []
    out = [
        "\n### Optimizer termination (iterative comparators)\n",
        (
            "PVI runs a fixed number of stochastic steps (no convergence test; "
            "'converged' only means finite parameters). PACm and PAC$^2_T$ use "
            "L-BFGS-B with separate iteration (maxiter) and function-evaluation "
            "(maxfun) limits; 'converged' is scipy's success flag.\n"
        ),
        (
            "| method | size | converged | median iterations | median evaluations "
            "| max_iter | max_fun | termination messages |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for (method, size), cell in rows.groupby(["method", "size"], sort=False):
        conv = cell.converged.dropna().astype(bool)
        msgs = (
            cell.termination_message.dropna().value_counts().to_dict()
            if "termination_message" in cell
            else {}
        )
        msg = "; ".join(f"{k} ({v})" for k, v in msgs.items()) or "--"

        def med(col):
            v = cell[col].dropna() if col in cell else []
            return f"{np.median(v):.0f}" if len(v) else "--"

        def first(col):
            v = cell[col].dropna() if col in cell else []
            return f"{v.iloc[0]:.0f}" if len(v) else "--"

        out.append(
            f"| {display(method)} | {size:g} | {int(conv.sum())}/{len(conv)} | "
            f"{med('n_iter')} | {med('n_fev')} | {first('max_iter')} | "
            f"{first('max_fun')} | {msg} |"
        )
    return out


def _bound_lines(sub, name):
    pac = sub[sub.method == "Ellipse+PAC"]
    if pac.empty or "bound" not in pac:
        return []
    out = [
        "\n### POPS Ellipse+PAC bound (floor-contaminated log risk, beta = 0.02)\n",
        (
            "'stored' is the bound for the returned predictive (the stored finite "
            "mixture of hyperparameter draws that is also scored); 'continuous' is "
            "the bound for the continuous Gaussian-hyperposterior mixture, which "
            "the stored mixture approximates. 'test risk' is the same "
            "floor-contaminated loss of the stored mixture on the test set. "
            "Failure budget per fit: 0.05 (PAC-Bayes) + 0.01 (Monte Carlo) + 0.01 "
            "(stored mixture) = 0.07. 'units' counts independent units (Burgers: "
            "simulator cases), split in half between pilot and certification in "
            "each of the two folds. The bound constrains log risk only; it "
            "implies no interval coverage.\n"
        ),
        (
            "| size | units | stored bound | continuous bound | trivial | test risk "
            "| non-vacuous | lambda/N1 |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for size in SIZES[name]:
        cell = pac[pac["size"] == size]
        if cell.empty:
            continue
        nonvac = np.mean(cell.bound < cell.trivial_bound)
        cont = (
            _median_interval(cell.continuous_bound, "{:.2f}")
            if "continuous_bound" in cell
            else "--"
        )
        out.append(
            f"| {size:g} | {cell.n_units.median():.0f} | "
            f"{_median_interval(cell.bound, '{:.2f}')} | {cont} | "
            f"{cell.trivial_bound.median():.2f} | "
            f"{_median_interval(cell.pac_test_risk, '{:.2f}')} | "
            f"{nonvac:.2f} | "
            f"{_median_interval(cell.lam / (cell.n_units / 2), '{:.3f}')} |"
        )
    return out


HEADLINE = (
    ("coverage_95.45", "cov.", "{:.2f}"),
    ("interval_score", "IS", "{:.3g}"),
    ("nll_floor", "NLL$_b$", "{:.2f}"),
    ("pp_area_abs", "A$_{abs}$", "{:.3f}"),
)
HEADLINE_CAPTION = (
    "Medians over training draws at the smallest and largest training size "
    "(u/P: independent units per parameter). cov.: coverage of the 95.45% "
    "central interval; IS: interval score of that interval; NLL$_b$: floored "
    "negative log predictive density (b = 0.01); A$_{abs}$: unsigned "
    "calibration area of the pooled absolute-error P-P curve. '*': at least "
    "one fit did not report optimizer convergence (kept). Abbreviations: "
)


def headline(frame):
    """Compact main-text table: smallest and largest size of each problem."""
    methods = [m for m in harness.METHODS if m in set(frame.method)]
    caption = HEADLINE_CAPTION + _abbreviations() + "."
    md = [f"# Headline table (medians over training draws)\n\n{caption}\n"]
    tex = ["% " + caption + "\n"]
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
                flag = "*" if not cell.empty and _unconverged(cell) else ""
                for key, _, fmt in HEADLINE:
                    v = cell[key].to_numpy(dtype=float) if key in cell else []
                    v = np.asarray(v)[np.isfinite(v)] if len(v) else np.array([])
                    cells.append(fmt.format(np.median(v)) + flag if v.size else "--")
            md.append(f"| {display(method)} | " + " | ".join(cells) + " |")
            rows.append(f"{display(method)} & " + " & ".join(cells) + r" \\")
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
        ("interval_score", "Interval score of 95.45% interval", "log"),
        ("nll_floor", "Floored neg. log pred. density [nats]", "linear"),
        ("pp_area_abs", "Unsigned calibration area $A_{abs}$", "linear"),
    )
    columns = tuple(c for c in columns if c[0] in frame)
    problems = [p for p in SIZES if p in set(frame.problem)]
    fig, axes = plt.subplots(
        len(problems),
        len(columns),
        figsize=(3.5 * len(columns), 2.55 * len(problems)),
        squeeze=False,
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
                ax.plot(x, q[:, 1], label=display(method), ms=2.5, **style)
                ax.fill_between(
                    x, q[:, 0], q[:, 2], color=style["color"], alpha=0.12, lw=0
                )
                # Filled markers: every fit converged; hollow: at least one
                # fit did not (kept, not dropped).
                bad = np.array(
                    [_unconverged(cell[cell.units_per_param == v]) for v in x]
                )
                ax.plot(x[~bad], q[~bad, 1], "o", ms=2.8, color=style["color"])
                ax.plot(x[bad], q[bad, 1], "o", ms=4, mfc="white", color=style["color"])
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
    parser.add_argument(
        "--methods", default=None, help="comma-separated subset of methods to run"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="replace only the rows of --methods in the saved CSV",
    )
    args = parser.parse_args()
    if args.plot_only:
        frame = pd.read_csv(OUT / "repeated_splits.csv")
    else:
        methods = harness.METHODS + harness.EXTRA_METHODS
        if args.methods:
            methods = tuple(args.methods.split(","))
        old = pd.read_csv(OUT / "repeated_splits.csv") if args.merge else None
        frame = run(args.seeds, args.workers, args.problems.split(","), methods)
        if old is not None:
            frame = pd.concat(
                [old[~old.method.isin(methods)], frame], ignore_index=True
            )
            frame.to_csv(OUT / "repeated_splits.csv", index=False)
    summarize(frame)
    summarize(frame, CALIBRATION_METRICS, "repeated_splits_calibration")
    headline(frame)
    plot(frame, HERE / "repeated_splits.png")


if __name__ == "__main__":
    main()
