"""Fluctuations of the hyperparameter-level log loss from held-out data.

This study measures, on independent test data, the one-sided moment that the
PAC-Bayes bound requires (see :mod:`comparisons.fluctuations`), always for the
floor-contaminated log loss of the PAC construction (``beta = 0.02`` on the
declared output interval), with Burgers losses averaged within each
simulator case (the independent unit) first:

- the plug-in cumulant generating function ``psi(t)`` of ``G(Psi) -
  l(Psi, Z)`` for ``0 < t <= 1`` of the fitted POPS Ellipse, as the median and
  central 90% band over the training draws, against

  - its linear Jensen bound ``t J(Psi)`` (plug-in),
  - the one-sided Bennett (sub-gamma) bound with the sample variance and
    ``c(Psi) = (G - a) / 3``, where ``a`` is either the protocol-fixed
    domain-wide minimum loss (from the analytic half-width floor, a valid
    constant) or the smallest loss seen on the test inputs (a plug-in, NOT a
    domain bound),
  - the uniform Hoeffding bound ``t^2 (b - a)^2 / 8`` (a population
    guarantee for the floored loss, no data used);

- ``J(Psi) = log E p - E log p``, the exact per-datum moment term at
  ``lambda = N``, for the fitted ellipsoid, for the stored draws of each PAC
  hyperposterior fold and for draws of each PAC hyperprior in the
  predeclared grid (log axis multipliers only: the center is frozen by
  default, exactly as in the fitted construction), with the exponential
  average ``(1 / N_1) log mean exp(N_1 J)`` formed within each fold;
- a stability check of those exponential averages against the number of
  prior draws and of test units;
- the realized generalization gap ``G_test(Psi_hat) - G_hat_train(Psi_hat)``
  of the fitted ellipsoid over repeated training draws.

Everything here is an empirical plug-in estimate from finite test sets,
except the Hoeffding curve and the domain-wide ``a``. No finite sample
establishes a moment condition uniformly over a hyperprior.

Outputs: ``generated/loss_fluctuations.csv``, ``loss_fluctuations.md``,
``loss_fluctuation_stability.csv`` and ``loss_fluctuations.png``.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from comparisons import fluctuations as fl  # noqa: E402
from comparisons import harness  # noqa: E402

from popsregression import POPSEllipseRegression  # noqa: E402
from popsregression._projected_ball import log_norm_constant  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
BETA = 0.02
T_GRID = np.linspace(0.0, 1.0, 41)[1:]
SIZES = {
    "quartic": (10, 20, 50, 100, 300),
    "burgers": (8, 16, 24, 40, 80),
    "ace": (1.5, 3.0, 6.0, 20.0),
}
CURVE_SIZES = {"quartic": (20, 300), "burgers": (8, 80), "ace": (1.5, 20.0)}
N_PRIOR = 512  # hyperprior draws per fold and grid value
STABILITY_SEEDS = 3
STABILITY_DRAWS = (64, 256, 1024, 4096)  # fresh draws in every replicate
STABILITY_FRACTIONS = (0.25, 0.5, 1.0)
STABILITY_REPLICATES = 10
TITLES = {
    "quartic": "Quartic ($P=5$)",
    "burgers": "Burgers ($P=8$)",
    "ace": "ACE ($P=36$)",
}


def make_problem(name, size, seed):
    if name == "quartic":
        return harness.quartic_problem(int(size), seed)
    if name == "burgers":
        return harness.burgers_problem(int(size), seed)
    return harness.ace_problem(float(size), seed)


def unit_average(losses, groups):
    """Average row losses within independent units (last axis)."""
    if groups is None:
        return losses
    _, index = np.unique(groups, return_inverse=True)
    counts = np.bincount(index)
    out = np.zeros(losses.shape[:-1] + (counts.size,))
    np.add.at(out, (..., index), losses)
    return out / counts


def kernel_losses(kernel, psi, X, y, y_bounds):
    """Per-draw, per-row floored losses of one kernel at hyperparameters psi."""
    mean, half = kernel.pushforward(kernel.design(X), psi)
    return fl.floored_losses(mean, half, y, kernel.ball_dim, BETA, y_bounds)


def model_losses(model, X, y, y_bounds):
    """Floored losses of the stored draws, one array per fold (component)."""
    out = []
    for _, kernel, draws in model.components_:
        loss, outside = kernel_losses(kernel, draws, X, y, y_bounds)
        out.append((loss, float(outside.mean())))
    return out


def test_minimum_loss(model, X, y_bounds):
    """Smallest floored loss over the TEST inputs (densest point): a plug-in,
    not a domain-wide bound."""
    half = min(
        kernel.pushforward(kernel.design(X), draws)[1].min()
        for _, kernel, draws in model.components_
    )
    density = (1 - BETA) * np.exp(log_norm_constant(model._ball_dim)) / half
    return -np.log(density + BETA / (y_bounds[1] - y_bounds[0]))


def prior_draws(kernel, model, prior_std, n, rng):
    """Draws of one predeclared PAC hyperprior, in the free coordinates only.

    With the default ``optimize_center=False`` the center shifts are frozen at
    zero and only the log axis multipliers are random, exactly as in the
    fitted construction.
    """
    n_dim = kernel.n_dim
    psi = np.zeros((n, 2 * n_dim))
    psi[:, n_dim:] = prior_std * rng.randn(n, n_dim)
    if model.optimize_center:
        psi[:, :n_dim] = model.pac_center_std * rng.randn(n, n_dim)
    return psi


def run_one(task):
    name, size, seed = task
    warnings.filterwarnings("ignore")
    problem = make_problem(name, size, seed)
    X, y, groups = problem.X_train, problem.y_train, problem.groups
    Xt, yt, gt = problem.X_test, problem.y_test, problem.test_groups
    pre = problem.preprocessor
    base = {
        "problem": name,
        "size": size,
        "seed": seed,
        "n_units": problem.n_units,
        "n_test_units": int(np.unique(gt).size) if gt is not None else yt.size,
        "units_per_param": problem.n_units / problem.n_params,
    }
    rows, curves, stability = [], [], []

    # Bare fitted ellipsoid: data-dependent Psi_hat, fixed given training.
    bare = POPSEllipseRegression(preprocessor=pre, random_state=seed).fit(X, y)
    ((test, outside),) = model_losses(bare, Xt, yt, problem.y_bounds)
    ((train, _),) = model_losses(bare, X, y, problem.y_bounds)
    test_units = unit_average(test, gt)[0]
    G = float(test_units.mean())
    variance = float(test_units.var())
    a_domain = fl.domain_minimum_loss(
        bare._ball_dim, BETA, problem.y_bounds, bare.delta
    )
    a_test = test_minimum_loss(bare, Xt, problem.y_bounds)
    b = float(np.log((problem.y_bounds[1] - problem.y_bounds[0]) / BETA))
    c_domain, tmax_domain = fl.bennett_range(G - a_domain)
    c_test, tmax_test = fl.bennett_range(G - a_test)
    rows.append(
        {
            **base,
            "psi": "fitted ellipse",
            "J": float(fl.jensen_gap(test_units)),
            "var": variance,
            "G_test": G,
            "G_train": float(unit_average(train, groups)[0].mean()),
            "a_domain": a_domain,
            "a_test": a_test,
            "loss_upper": b,
            "c_domain": c_domain,
            "t_max_domain": tmax_domain,
            "c_test": c_test,
            "t_max_test": tmax_test,
            "hoeffding_at_1": float(fl.hoeffding_cgf(1.0, b - a_domain)),
            "outside_support": outside,
        }
    )
    if size in CURVE_SIZES[name]:
        curves.append(
            {
                "problem": name,
                "size": size,
                "seed": seed,
                "t": T_GRID,
                "psi": fl.cgf(test_units, T_GRID),
                "J": float(fl.jensen_gap(test_units)),
                "var": variance,
                "G": G,
                "a_domain": a_domain,
                "a_test": a_test,
                "loss_upper": b,
                "losses": test_units,
            }
        )

    # PAC hyperposterior (stored draws) and the predeclared hyperpriors, one
    # fold at a time: exponential averages are never pooled across folds.
    pac = POPSEllipseRegression(
        regularization="PAC",
        y_bounds=problem.y_bounds,
        preprocessor=pre,
        random_state=seed,
    ).fit(X, y, groups=groups)
    cert = pac.certificate_
    grid = np.atleast_1d(np.asarray(pac.pac_log_scale_std, dtype=float))
    rng = np.random.RandomState(1000 + seed)
    for k, ((_, kernel, draws), fold) in enumerate(zip(pac.components_, cert.folds)):
        loss, outside_post = kernel_losses(kernel, draws, Xt, yt, problem.y_bounds)
        J_post = fl.jensen_gap(unit_average(loss, gt))
        a_pac = fl.domain_minimum_loss(
            kernel.ball_dim, BETA, problem.y_bounds, kernel.delta
        )
        rows.append(
            {
                **base,
                "psi": "PAC hyperposterior",
                "fold": k,
                "prior_std": fold.prior_std,
                "J": float(J_post.mean()),
                "J_softmax_N1": float(fl.soft_max(J_post, fold.n_units)),
                "n_cert_units": fold.n_units,
                "a_domain": a_pac,
                "outside_support": float(outside_post.mean()),
                "bound": pac.bound_,
                "continuous_bound": cert.continuous_bound,
            }
        )
        for prior_std in grid:
            psi = prior_draws(kernel, pac, prior_std, N_PRIOR, rng)
            loss, _ = kernel_losses(kernel, psi, Xt, yt, problem.y_bounds)
            J = fl.jensen_gap(unit_average(loss, gt))
            rows.append(
                {
                    **base,
                    "psi": "PAC hyperprior",
                    "fold": k,
                    "prior_std": prior_std,
                    "selected": bool(prior_std == fold.prior_std),
                    "J": float(J.mean()),
                    "J_max": float(J.max()),
                    "J_softmax_N1": float(fl.soft_max(J, fold.n_units)),
                    "n_cert_units": fold.n_units,
                }
            )
        if size in CURVE_SIZES[name] and seed < STABILITY_SEEDS:
            stability.extend(
                stability_rows(base, kernel, pac, fold, Xt, yt, gt, problem, rng, k)
            )
    return rows, curves, stability


def stability_rows(base, kernel, model, fold, Xt, yt, gt, problem, rng, k):
    """Exponential averages at N_1 versus prior draws and test units.

    For each (number of prior draws M, fraction of test units kept) the
    selected hyperprior's ``(1/N_1) log mean exp(N_1 J)`` is re-estimated on
    independent replications; their spread shows whether the plug-in value
    has stabilized.
    """
    units = np.arange(yt.size) if gt is None else np.unique(gt)
    unit_of = np.arange(yt.size) if gt is None else gt
    rows = []
    for frac in STABILITY_FRACTIONS:
        for m in STABILITY_DRAWS:
            estimates = []
            for _ in range(STABILITY_REPLICATES):
                # Fresh prior draws and a fresh subset of test units for
                # every replicate.
                keep = rng.choice(units, max(2, int(frac * units.size)), False)
                cols = np.isin(unit_of, keep)
                psi = prior_draws(kernel, model, fold.prior_std, m, rng)
                loss, _ = kernel_losses(
                    kernel, psi, Xt[cols], yt[cols], problem.y_bounds
                )
                J = fl.jensen_gap(unit_average(loss, None if gt is None else gt[cols]))
                estimates.append(fl.soft_max(J, fold.n_units))
            rows.append(
                {
                    **base,
                    "fold": k,
                    "prior_std": fold.prior_std,
                    "n_prior_draws": m,
                    "test_unit_fraction": frac,
                    "softmax_median": float(np.median(estimates)),
                    "softmax_spread": float(
                        np.subtract(*np.quantile(estimates, [0.95, 0.05]))
                    ),
                }
            )
    return rows


def run(seeds, workers):
    tasks = [(n, s, seed) for n in SIZES for s in SIZES[n] for seed in range(seeds)]
    rows, curves, stability = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r, c, st in pool.map(run_one, tasks, chunksize=1):
            rows.extend(r)
            curves.extend(c)
            stability.extend(st)
    frame = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    frame.to_csv(OUT / "loss_fluctuations.csv", index=False)
    pd.DataFrame(stability).to_csv(OUT / "loss_fluctuation_stability.csv", index=False)
    np.save(
        OUT / "loss_fluctuation_curves.npy",
        np.array(curves, dtype=object),
        allow_pickle=True,
    )
    return frame, curves


def _mi(values, fmt="{:.3f}"):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "--"
    lo, med, hi = np.quantile(v, [0.05, 0.5, 0.95])
    return f"{fmt.format(med)} [{fmt.format(lo)}, {fmt.format(hi)}]"


BALL_DIM = {"quartic": 6, "burgers": 9, "ace": 37}  # P + 1 (offset coordinate)


def summarize(frame):
    n = frame.seed.nunique()
    lines = [
        "# Loss fluctuations of the hyperparameter-level log loss\n",
        (
            "Floor-contaminated log loss (beta = 0.02), held-out data, medians and "
            f"central 90% intervals over {n} training draws; Burgers losses are "
            "averaged within each simulator case first. All entries are empirical "
            "plug-in estimates from a finite test set, except the domain-wide "
            "minimum loss a and the Hoeffding value, which use no data. "
            "`J = log E p - E log p` is the per-datum moment term at lambda = N. "
            "`c = (G - a) / 3` is the Bennett (sub-gamma) scale and `t < 1/c` "
            "its admissible range: with the domain-wide a (from the analytic "
            "half-width floor delta = 0.001 of the fitted ellipse) it is a valid "
            "constant; with the smallest loss on the test inputs it is only a "
            "plug-in. `Hoeffding at t=1` is (b - a)^2 / 8, the uniform "
            "sub-Gaussian bound of the floored loss (it certifies nothing about "
            "the unfloored loss). PAC columns are per fold: `J softmax` is "
            "`(1/N_1) log mean exp(N_1 J)` over one fold's draws, never pooled "
            "across folds; the prior rows use the predeclared hyperprior grid "
            "in the free coordinates (log axis multipliers; center frozen). "
            "The calibrated fixed-width Gaussian reference is J = "
            f"{fl.GAUSSIAN_CALIBRATED_GAP:.3f}.\n"
        ),
    ]
    for name in SIZES:
        sub = frame[frame.problem == name]
        if sub.empty:
            continue
        ref = fl.calibrated_projected_ball_gap(BALL_DIM[name])
        lines.append(
            f"\n## {name} (calibrated projected-ball reference J = {ref:.3f})\n"
        )
        lines.append("\n### Fitted POPS Ellipse\n")
        lines.append(
            "| size | J | Var(l) | G_test - G_train | outside support "
            "| a domain | c domain | t_max domain | a test (plug-in) | c test "
            "| Hoeffding at t=1 |"
        )
        lines.append("|---" * 11 + "|")
        for size in SIZES[name]:
            f = sub[(sub["size"] == size) & (sub.psi == "fitted ellipse")]
            lines.append(
                f"| {size:g} | {_mi(f.J)} | {_mi(f['var'])} |"
                f" {_mi(f.G_test - f.G_train)} | {_mi(f.outside_support)} |"
                f" {f.a_domain.median():.2f} | {_mi(f.c_domain, '{:.2f}')} |"
                f" {_mi(f.t_max_domain, '{:.3f}')} | {_mi(f.a_test, '{:.2f}')} |"
                f" {_mi(f.c_test, '{:.2f}')} | {f.hoeffding_at_1.median():.1f} |"
            )
        lines.append("\n### POPS Ellipse+PAC, per fold\n")
        lines.append(
            "| size | N_1 | J post. | softmax post. | J prior (selected) "
            "| softmax prior (selected) | softmax prior (worst grid value) "
            "| stored bound |"
        )
        lines.append("|---" * 8 + "|")
        for size in SIZES[name]:
            p = sub[(sub["size"] == size) & (sub.psi == "PAC hyperposterior")]
            q = sub[(sub["size"] == size) & (sub.psi == "PAC hyperprior")]
            sel = q[q.selected.astype(bool)]
            worst = q.groupby(["seed", "fold"]).J_softmax_N1.max()
            lines.append(
                f"| {size:g} | {p.n_cert_units.median():.0f} | {_mi(p.J)} |"
                f" {_mi(p.J_softmax_N1)} | {_mi(sel.J)} | {_mi(sel.J_softmax_N1)} |"
                f" {_mi(worst)} | {_mi(p.bound, '{:.2f}')} |"
            )
    stab_path = OUT / "loss_fluctuation_stability.csv"
    if stab_path.exists():
        st = pd.read_csv(stab_path)
        lines.append(
            "\n## Stability of the prior exponential average at N_1\n\n"
            "Selected hyperprior, per fold, first "
            f"{STABILITY_SEEDS} training draws at the curve sizes. Each cell: "
            "median over draws and folds of the replicate median, and in "
            "parentheses the median 5-95% spread across "
            f"{STABILITY_REPLICATES} replications with independent prior draws "
            "and test-unit subsets. A spread that shrinks with more draws and "
            "units indicates a stable plug-in value; a median that keeps rising "
            "with more prior draws indicates that the exponential average is "
            "dominated by rare draws, and is then not established by the "
            "sample.\n"
        )
        for name in SIZES:
            s2 = st[st.problem == name]
            if s2.empty:
                continue
            for size in sorted(s2["size"].unique()):
                s3 = s2[s2["size"] == size]
                lines.append(f"\n### {name}, size {size:g}\n")
                draws = sorted(s3.n_prior_draws.unique())
                lines.append(
                    "| test units kept | "
                    + " | ".join(f"M = {m}" for m in draws)
                    + " |"
                )
                lines.append("|---" * (1 + len(draws)) + "|")
                for frac in sorted(s3.test_unit_fraction.unique()):
                    cells = []
                    for m in draws:
                        c = s3[
                            (s3.test_unit_fraction == frac) & (s3.n_prior_draws == m)
                        ]
                        cells.append(
                            f"{c.softmax_median.median():.3f} "
                            f"({c.softmax_spread.median():.3f})"
                        )
                    lines.append(f"| {frac:.0%} | " + " | ".join(cells) + " |")
    (OUT / "loss_fluctuations.md").write_text("\n".join(lines) + "\n")


def _band(ax, t, values, **style):
    q = np.nanquantile(np.vstack(values), [0.05, 0.5, 0.95], axis=0)
    ax.plot(t, q[1], **style)
    ax.fill_between(t, q[0], q[2], color=style.get("color", "k"), alpha=0.1, lw=0)


def plot(frame, curves, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8))
    colors = {"quartic": "C0", "burgers": "C2", "ace": "C3"}
    for c, name in enumerate(SIZES):
        ax = axes[0, c]
        recs = [r for r in curves if r["problem"] == name]
        sizes = sorted({r["size"] for r in recs})
        top = 0.0
        for k, size in enumerate(sizes):
            group = [r for r in recs if r["size"] == size]
            t = group[0]["t"]
            ls = "-" if k == 0 else "--"
            tag = "small N" if k == 0 else "large N"
            _band(
                ax,
                t,
                [r["psi"] for r in group],
                color="k",
                ls=ls,
                lw=1.6,
                label=rf"$\hat\psi(t)$, {tag}",
            )
            _band(
                ax,
                t,
                [t * r["J"] for r in group],
                color="C1",
                ls=ls,
                lw=1.0,
                label=r"$t\,\hat J$" if k == 0 else None,
            )
            _band(
                ax,
                t,
                [fl.bennett_cgf(t, r["var"], r["G"] - r["a_domain"]) for r in group],
                color="C2",
                ls=ls,
                lw=1.0,
                label="Bennett, domain-wide $a$" if k == 0 else None,
            )
            _band(
                ax,
                t,
                [fl.bennett_cgf(t, r["var"], r["G"] - r["a_test"]) for r in group],
                color="C9",
                ls=ls,
                lw=1.0,
                label="Bennett, test-set $a$ (plug-in)" if k == 0 else None,
            )
            _band(
                ax,
                t,
                [fl.hoeffding_cgf(t, r["loss_upper"] - r["a_domain"]) for r in group],
                color="C4",
                ls=ls,
                lw=1.0,
                label=r"Hoeffding $t^2R^2/8$" if k == 0 else None,
            )
            top = max(top, np.nanmedian(np.vstack([r["psi"] for r in group]), 0).max())
        ax.set_ylim(0, 3.0 * top)
        ax.set_title(
            TITLES[name] + f", median of {len(recs) // len(sizes)} draws", fontsize=8.5
        )
        ax.set_xlabel(r"$t=\lambda/N$", fontsize=8)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel("one-sided CGF [nats]", fontsize=8)
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")

    ax = axes[1, 0]
    for name in SIZES:
        for psi, ls, marker in (
            ("fitted ellipse", "-", "o"),
            ("PAC hyperposterior", "--", "s"),
            ("PAC hyperprior", ":", "^"),
        ):
            sub = frame[(frame.problem == name) & (frame.psi == psi)]
            if psi == "PAC hyperprior":
                sub = sub[sub.selected.astype(bool)]
            g = sub.groupby("units_per_param").J
            x = np.array(sorted(g.groups))
            q = np.array([np.nanquantile(g.get_group(v), [0.05, 0.5, 0.95]) for v in x])
            ax.plot(x, q[:, 1], color=colors[name], ls=ls, marker=marker, ms=2.5)
            ax.fill_between(x, q[:, 0], q[:, 2], color=colors[name], alpha=0.08, lw=0)
    ax.axhline(fl.GAUSSIAN_CALIBRATED_GAP, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1.0, 2.0, 5.0]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.1", "0.2", "0.5", "1", "2", "5"]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_title(r"$J(\Psi)=\ln\mathbb{E}p-\mathbb{E}\ln p$ (plug-in)", fontsize=9)
    ax.set_xlabel("units per parameter", fontsize=8)
    ax.tick_params(labelsize=7)
    first = ax.legend(
        handles=[Line2D([], [], color=colors[n], label=n) for n in SIZES],
        fontsize=6,
        frameon=False,
        loc="upper right",
    )
    ax.add_artist(first)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color="0.3",
                ls="-",
                marker="o",
                ms=2.5,
                label="fitted POPS Ellipse",
            ),
            Line2D(
                [],
                [],
                color="0.3",
                ls="--",
                marker="s",
                ms=2.5,
                label="PAC hyperposterior (per fold)",
            ),
            Line2D(
                [],
                [],
                color="0.3",
                ls=":",
                marker="^",
                ms=2.5,
                label="PAC hyperprior (selected, per fold)",
            ),
            Line2D([], [], color="k", ls=":", lw=0.8, label="calibrated Gaussian"),
        ],
        fontsize=6,
        frameon=False,
        loc="lower right",
    )

    ax = axes[1, 1]
    for name in SIZES:
        sub = frame[(frame.problem == name) & (frame.psi == "fitted ellipse")]
        g = sub.assign(gap=sub.G_test - sub.G_train).groupby("units_per_param").gap
        x = np.array(sorted(g.groups))
        q = np.array([np.nanquantile(g.get_group(v), [0.05, 0.5, 0.95]) for v in x])
        ax.plot(x, q[:, 1], color=colors[name], marker="o", ms=2.5, label=name)
        ax.fill_between(x, q[:, 0], q[:, 2], color=colors[name], alpha=0.15, lw=0)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_title(
        r"$G_{\rm test}(\hat\Psi)-\hat G_{N}(\hat\Psi)$, fitted POPS Ellipse",
        fontsize=9,
    )
    ax.set_xlabel("units per parameter", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False)

    ax = axes[1, 2]
    for name in SIZES:
        recs = [r for r in curves if r["problem"] == name]
        small = min(r["size"] for r in recs)
        group = [r for r in recs if r["size"] == small]
        grid = np.linspace(0.005, 0.995, 199)
        qs = [np.quantile(r["losses"] - r["G"], grid) for r in group]
        med = np.median(np.vstack(qs), axis=0)
        ax.plot(med, grid, color=colors[name], label=f"{name}, small N")
        ax.axvline(
            np.median([r["a_domain"] - r["G"] for r in group]),
            color=colors[name],
            ls=":",
            lw=0.9,
        )
        ax.axvline(
            np.median([r["a_test"] - r["G"] for r in group]),
            color=colors[name],
            ls="--",
            lw=0.7,
        )
    ax.set_title(
        r"test CDF of $\ell-G$ (dotted: domain $a-G$; dashed: test-set)", fontsize=8
    )
    ax.set_xlabel("centered loss [nats]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        frame = pd.read_csv(OUT / "loss_fluctuations.csv")
        curves = list(np.load(OUT / "loss_fluctuation_curves.npy", allow_pickle=True))
    else:
        frame, curves = run(args.seeds, args.workers)
    summarize(frame)
    plot(frame, curves, HERE / "loss_fluctuations.png")


if __name__ == "__main__":
    main()
