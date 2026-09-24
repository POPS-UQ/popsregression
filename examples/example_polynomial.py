"""Quartic surrogate of an oscillatory engine (paper Fig. ``fig:demo``).

Rows are training-set sizes; columns are Bayesian ridge, the POPS hypercube,
the bare POPS ellipse, Ellipse+EB (the empirical-Bayes Laplace layer) and
Ellipse+PAC (the pilot-split PAC-Bayes construction). Every panel shows the
same two exact central intervals of the parameter-only predictive, 95.45%
(inner) and 99.9% (outer): Gaussian quantiles of the weight posterior
``sigma_`` for Bayesian ridge (its fitted noise precision is never used),
empirical quantiles of 4096 parameter draws for the hypercube, and exact
mixture quantiles for the ellipse family. Labels report held-out coverage of
both intervals on a dense uniform grid.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from comparisons import harness
from comparisons.plotting import band_panel, coverage_label, legend_handles

SEED = 1042
TRAIN_SIZES = (10, 100)
METHODS = (
    "Bayesian ridge",
    "POPS hypercube",
    "POPS ellipse",
    "Ellipse+EB",
    "Ellipse+PAC",
)
TITLES = (
    "Bayesian ridge",
    "POPS hypercube",
    "POPS ellipse",
    "Ellipse+EB",
    "Ellipse+PAC",
)


def make_problem(rng, n_samples):
    """N uniform inputs on [-10, 10] plus the two endpoints (as in the paper)."""
    x_train = np.sort(np.append(rng.uniform(-1, 1, n_samples), [-1.0, 1.0]) * 10)
    x_dense = np.linspace(-10, 10, 401)
    problem = harness.Problem(
        name="quartic",
        X_train=harness.quartic_features(x_train),
        y_train=harness.quartic_target(x_train),
        X_test=harness.quartic_features(x_dense),
        y_test=harness.quartic_target(x_dense),
        y_bounds=harness.QUARTIC_Y_BOUNDS,
        stacking_components=[np.arange(d + 1) for d in range(1, 5)],
    )
    return problem, x_train, x_dense


def main(output=None):
    rng = np.random.RandomState(SEED)
    fig, axes = plt.subplots(
        len(TRAIN_SIZES), len(METHODS), figsize=(10, 3.3), sharex=True, sharey=True
    )
    for row, n_samples in enumerate(TRAIN_SIZES):
        problem, x_train, x_dense = make_problem(rng, n_samples)
        for col, method in enumerate(METHODS):
            pred = harness.fit_predictive(method, problem, SEED)
            ax = axes[row, col]
            band_panel(ax, x_dense, pred, problem.y_test, x_train, problem.y_train)
            coverage_label(ax, problem.y_test, pred)
            if method == "Ellipse+PAC":
                print(
                    f"N={n_samples}: PAC bound {pred.info['bound']:.2f} "
                    f"(trivial {pred.info['trivial_bound']:.2f})"
                )
        axes[row, 0].set_ylabel(f"N = {n_samples}", fontsize=9)
    for col, title in enumerate(TITLES):
        axes[0, col].set_title(title, fontsize=9.5, pad=12)
    for ax in axes.flat:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-150, 150)
        ax.tick_params(labelsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("x", fontsize=9)
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(pad=0.2, w_pad=0.25, h_pad=0.9, rect=(0, 0.06, 1, 1))
    output = output or Path(__file__).resolve().parent / "example_polynomial.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
