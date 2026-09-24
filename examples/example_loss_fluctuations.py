"""Fluctuations of the hyperparameter-level log loss from held-out data.

This study measures, on independent test data, the one-sided moment that the
PAC-Bayes bound requires (see :mod:`comparisons.fluctuations`):

- the plug-in cumulant generating function ``psi(t)`` of ``G(Psi) - l(Psi, Z)``
  for ``0 < t <= 1``, against its linear Jensen bound ``t J(Psi)``, the
  one-sided Bennett (sub-gamma) bound and the sub-Gaussian proxy
  ``t^2 Var(l) / 2``;
- ``J(Psi) = log E p - E log p``, which is exactly the per-datum moment term
  at ``lambda = N``, for the fitted bare ellipsoid, for draws of the PAC
  hyperposterior and for draws of the (data-independent) PAC hyperprior,
  where ``(1 / N_1) log E_{pi_0H} exp(N_1 J)`` is the exact moment term of
  Theorem 1 at ``lambda = N_1``;
- the realized generalization gap ``G_test(Psi_hat) - G_hat_train(Psi_hat)``
  of the fitted ellipsoid over repeated training draws.

All losses are the floor-contaminated log loss of the PAC construction
(``beta = 0.02`` on the declared output interval), which is finite; the
fraction of test targets outside the compact support (infinite unfloored
loss) is reported alongside.

Outputs: ``generated/loss_fluctuations.csv``, ``generated/loss_fluctuations.md``
and ``loss_fluctuations.png``.
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


def model_losses(model, X, y, groups, y_bounds):
    """Per-draw, per-unit floored losses of every stored component draw."""
    rows, outside = [], []
    for _, kernel, draws in model.components_:
        mean, half = kernel.pushforward(kernel.design(X), draws)
        loss, out = fl.floored_losses(mean, half, y, model._ball_dim, BETA, y_bounds)
        rows.append(unit_average(loss, groups))
        outside.append(out)
    return np.vstack(rows), float(np.mean(np.vstack(outside)))


def minimum_loss(model, X, y_bounds):
    """``a_Psi``: the smallest floored loss over the test inputs (densest point)."""
    half = min(
        kernel.pushforward(kernel.design(X), draws)[1].min()
        for _, kernel, draws in model.components_
    )
    density = (1 - BETA) * np.exp(log_norm_constant(model._ball_dim)) / half
    return -np.log(density + BETA / (y_bounds[1] - y_bounds[0]))


def run_one(task):
    name, size, seed = task
    warnings.filterwarnings("ignore")
    problem = make_problem(name, size, seed)
    X, y, groups = problem.X_train, problem.y_train, problem.groups
    Xt, yt, gt = problem.X_test, problem.y_test, problem.test_groups
    base = {
        "problem": name,
        "size": size,
        "seed": seed,
        "n_units": problem.n_units,
        "units_per_param": problem.n_units / problem.n_params,
    }
    rows, curves = [], []

    # Bare fitted ellipsoid: data-dependent Psi_hat.
    bare = POPSEllipseRegression(random_state=seed).fit(X, y)
    test, outside = model_losses(bare, Xt, yt, None, problem.y_bounds)
    train, _ = model_losses(bare, X, y, groups, problem.y_bounds)
    test_units = unit_average(test, gt)
    G = float(test_units.mean())
    gap_min = G - minimum_loss(bare, Xt, problem.y_bounds)
    variance = float(test_units.var())
    rows.append(
        {
            **base,
            "psi": "fitted ellipse",
            "J": float(fl.jensen_gap(test_units)[0]),
            "var": variance,
            "G_test": G,
            "G_train": float(train.mean()),
            "gap_to_min": gap_min,
            "outside_support": outside,
        }
    )
    if size in CURVE_SIZES[name] and seed == 0:
        curves.append(
            {
                "problem": name,
                "size": size,
                "t": T_GRID,
                "psi": fl.cgf(test_units[0], T_GRID),
                "J": float(fl.jensen_gap(test_units)[0]),
                "var": variance,
                "gap_to_min": gap_min,
                "losses": test_units[0],
            }
        )

    # PAC hyperposterior draws and the data-independent hyperprior.
    pac = POPSEllipseRegression(
        regularization="PAC", y_bounds=problem.y_bounds, random_state=seed
    ).fit(X, y, groups=groups)
    post, outside_post = model_losses(pac, Xt, yt, None, problem.y_bounds)
    J_post = fl.jensen_gap(unit_average(post, gt))
    rows.append(
        {
            **base,
            "psi": "PAC hyperposterior",
            "J": float(J_post.mean()),
            "J_softmax_N1": float(
                np.mean(
                    [fl.soft_max(J_post, f.n_units) for f in pac.certificate_.folds]
                )
            ),
            "outside_support": outside_post,
            "bound": pac.bound_,
        }
    )
    rng = np.random.RandomState(1000 + seed)
    J_prior, softmax_prior = [], []
    for (_, kernel, _), fold in zip(pac.components_, pac.certificate_.folds):
        n_dim = kernel.n_dim
        psi = np.zeros((512, 2 * n_dim))
        psi[:, n_dim:] = fold.prior_std * rng.randn(512, n_dim)
        mean, half = kernel.pushforward(kernel.design(Xt), psi)
        loss, _ = fl.floored_losses(mean, half, yt, n_dim, BETA, problem.y_bounds)
        J = fl.jensen_gap(unit_average(loss, gt))
        J_prior.append(J)
        softmax_prior.append(fl.soft_max(J, fold.n_units))
    rows.append(
        {
            **base,
            "psi": "PAC hyperprior",
            "J": float(np.mean(np.concatenate(J_prior))),
            "J_softmax_N1": float(np.mean(softmax_prior)),
        }
    )
    return rows, curves


def run(seeds, workers):
    tasks = [(n, s, seed) for n in SIZES for s in SIZES[n] for seed in range(seeds)]
    rows, curves = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r, c in pool.map(run_one, tasks, chunksize=1):
            rows.extend(r)
            curves.extend(c)
    frame = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    frame.to_csv(OUT / "loss_fluctuations.csv", index=False)
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


def summarize(frame):
    lines = [
        "# Loss fluctuations of the hyperparameter-level log loss\n",
        (
            "Floor-contaminated log loss (beta = 0.02), held-out data, medians and "
            f"central 90% intervals over {frame.seed.nunique()} training draws. "
            "`J = log E p - E log p` is the exact per-datum moment term at lambda = N; "
            "`J softmax` is `(1/N_1) log mean exp(N_1 J)` over draws. The calibrated "
            f"fixed-width Gaussian reference is J = {fl.GAUSSIAN_CALIBRATED_GAP:.3f}.\n"
        ),
    ]
    for name in SIZES:
        ref = fl.calibrated_projected_ball_gap(
            {"quartic": 5, "burgers": 8, "ace": 36}[name]
        )
        lines.append(
            f"\n## {name} (calibrated projected-ball reference J = {ref:.3f})\n"
        )
        lines.append(
            "| size | J fitted | Var(l) fitted | G_test - G_train | outside support "
            "| J PAC post. | J PAC prior | prior softmax at N_1 | PAC bound |"
        )
        lines.append("|---" * 9 + "|")
        sub = frame[frame.problem == name]
        for size in SIZES[name]:
            f = sub[(sub["size"] == size) & (sub.psi == "fitted ellipse")]
            p = sub[(sub["size"] == size) & (sub.psi == "PAC hyperposterior")]
            q = sub[(sub["size"] == size) & (sub.psi == "PAC hyperprior")]
            lines.append(
                f"| {size:g} | {_mi(f.J)} | {_mi(f['var'])} |"
                f" {_mi(f.G_test - f.G_train)} | {_mi(f.outside_support)} |"
                f" {_mi(p.J)} | {_mi(q.J)} | {_mi(q.J_softmax_N1)} |"
                f" {_mi(p.bound, '{:.2f}')} |"
            )
    (OUT / "loss_fluctuations.md").write_text("\n".join(lines) + "\n")


def plot(frame, curves, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    colors = {"quartic": "C0", "burgers": "C2", "ace": "C3"}
    for c, name in enumerate(SIZES):
        ax = axes[0, c]
        for k, rec in enumerate(r for r in curves if r["problem"] == name):
            ls = "-" if k == 0 else "--"
            t = rec["t"]
            label = f"{'small' if k == 0 else 'large'} N"
            ax.plot(
                t,
                rec["psi"],
                color="k",
                ls=ls,
                lw=1.6,
                label=rf"$\hat\psi(t)$, {label}",
            )
            ax.plot(
                t,
                t * rec["J"],
                color="C1",
                ls=ls,
                lw=1.1,
                label=r"$t\,\hat J$" if k == 0 else None,
            )
            ax.plot(
                t,
                0.5 * rec["var"] * t * t,
                color="C4",
                ls=ls,
                lw=1.1,
                label=r"$t^2\widehat{\rm Var}(\ell)/2$" if k == 0 else None,
            )
            ax.plot(
                t,
                fl.bennett_cgf(t, rec["var"], rec["gap_to_min"]),
                color="C2",
                ls=ls,
                lw=1.1,
                label="Bennett" if k == 0 else None,
            )
        top = max(r["psi"].max() for r in curves if r["problem"] == name)
        ax.set_ylim(0, 2.2 * top)
        ax.set_title(TITLES[name], fontsize=9)
        ax.set_xlabel(r"$t=\lambda/N$", fontsize=8)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel("one-sided CGF [nats]", fontsize=8)
        ax.legend(fontsize=6, frameon=False, loc="upper left")

    ax = axes[1, 0]
    for name in SIZES:
        for psi, ls, marker in (
            ("fitted ellipse", "-", "o"),
            ("PAC hyperposterior", "--", "s"),
            ("PAC hyperprior", ":", "^"),
        ):
            sub = frame[(frame.problem == name) & (frame.psi == psi)]
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
    ax.set_title(r"$J(\Psi)=\ln\mathbb{E}p-\mathbb{E}\ln p$", fontsize=9)
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
                [], [], color="0.3", ls="-", marker="o", ms=2.5, label="fitted ellipse"
            ),
            Line2D(
                [],
                [],
                color="0.3",
                ls="--",
                marker="s",
                ms=2.5,
                label="PAC hyperposterior",
            ),
            Line2D(
                [], [], color="0.3", ls=":", marker="^", ms=2.5, label="PAC hyperprior"
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
        r"$G_{\rm test}(\hat\Psi)-\hat G_{N}(\hat\Psi)$, fitted ellipse", fontsize=9
    )
    ax.set_xlabel("units per parameter", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False)

    ax = axes[1, 2]
    for name in SIZES:
        rec = [r for r in curves if r["problem"] == name][0]
        centered = np.sort(rec["losses"] - rec["losses"].mean())
        prob = (np.arange(centered.size) + 0.5) / centered.size
        ax.plot(centered, prob, color=colors[name], label=f"{name}, small N")
        ax.axvline(-rec["gap_to_min"], color=colors[name], ls=":", lw=0.9)
    ax.set_title(r"test CDF of $\ell-G$ (dotted: $a_\Psi-G$)", fontsize=9)
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
