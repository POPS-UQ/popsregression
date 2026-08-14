"""
=======================================================================
Workshop demo figure: four uncertainty methods on the quartic benchmark
=======================================================================

Figure 1 of the Sim2Science workshop paper: the quartic-vs-oscillatory
benchmark of the package example (P = 5). A degree-4 polynomial is fit
to the noise-free oscillatory engine

    f(x) = 0.1 (x^3 + 0.01 x^4) + 10 x sin(x)

by four methods (columns: Bayesian Ridge, POPS Hypercube, POPS Ellipse,
POPS Ellipse + PAC) at N = 10 (top row) and N = 500 (bottom row).

Per panel: engine (black), train points (blue), +-2 sigma predictive
band (orange) and the outer band (grey): +-4 sigma for Bayesian Ridge,
sampled max/min for the hypercube, the ellipse support for the bare
ellipse, and the 2 sigma hyperposterior ensemble for + PAC.

Deterministic from the master seed; writes demo_workshop.png (dpi 200)
and demo_workshop.pdf.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

from popsregression import POPSRegression, POPSRegressionEllipse

N_ROWS = (10, 500)
METHODS = ("Bayesian Ridge", "POPS Hypercube", "POPS Ellipse",
           "POPS Ellipse + PAC")


def target_function(x):
    """Oscillatory engine of the package's quartic example."""
    return ((x)**3)*0.05 * np.exp(10.*x)/(1.0+np.exp(10.*x)) + 5.*np.cos(x) * x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--seed", type=int, default=42, help="master seed")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args(argv)

    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    x_plot = np.linspace(-10.0, 10.0, 400)
    y_plot = target_function(x_plot)
    poly = PolynomialFeatures(degree=4, include_bias=True)
    X_plot = poly.fit_transform(x_plot[:, None])

    rng = np.random.RandomState(args.seed)
    fig, axes = plt.subplots(
        2, 4, figsize=(9.0, 4.2), sharex=True, sharey=True,
        constrained_layout=True,
    )

    for row, n in enumerate(N_ROWS):
        # n - 2 uniform draws plus the two interval anchors, as in the
        # package's quartic example
        x_tr = np.sort(
            np.append(rng.uniform(-1, 1, n - 2), np.linspace(-1, 1, 2))
        ) * 10.0


        y_tr = target_function(x_tr)
        X_tr = poly.transform(x_tr[:, None])
        print(X_tr[0])
        delta = 1.0e-3 * float(y_tr.std())

        # Bayesian Ridge: epistemic-only sigma; outer band +-4 sigma
        br = BayesianRidge(fit_intercept=False)
        br.fit(X_tr, y_tr)
        m = br.predict(X_plot)
        s = np.sqrt(np.sum((X_plot @ br.sigma_) * X_plot, axis=1))
        panels = [(m, 2.0 * s, m - 4.0 * s, m + 4.0 * s)]

        # POPS hypercube: sampled max/min outer band
        np.random.seed(args.seed + n)  # the 'uniform' resampler is global
        pops = POPSRegression(
            fit_intercept=False, resample_density=max(10.0, 2.0e4 / n)
        )
        pops.fit(X_tr, y_tr)
        m, s, hi, lo = pops.predict(X_plot, return_std=True, return_bounds=True)
        panels.append((m, 2.0 * s, lo, hi))

        # POPS ellipse: support outer band
        ell = POPSRegressionEllipse(
            rank=32, max_iter=5000, fit_intercept=False,
            random_state=args.seed, delta=delta,
        )
        ell.fit(X_tr, y_tr)
        m, s, hi, lo = ell.predict(X_plot, return_std=True, return_bounds=True)
        panels.append((m, 2.0 * s, lo, hi))
        print(f"N = {n}: ellipse coverage_fraction_ = "
              f"{ell.coverage_fraction_:.4f}, n_iter_ = {ell.n_iter_}")

        # POPS ellipse + PAC: 2 sigma hyperposterior ensemble outer band
        pac = POPSRegressionEllipse(
            max_iter=5000, fit_intercept=False,
            random_state=args.seed, delta=delta, pac_bayes=True,
        )
        pac.fit(X_tr, y_tr)
        m, s, hi, lo = pac.predict(X_plot, return_std=True, return_bounds=True)
        panels.append((m, 2.0 * s, lo, hi))
        print(f"N = {n}: +PAC coverage_fraction_ = "
              f"{pac.coverage_fraction_:.4f}, n_iter_ = {pac.n_iter_}")

        for col, (mean, band, lo, hi) in enumerate(panels):
            ax = axes[row, col]
            ax.fill_between(x_plot, lo, hi, color="0.75", alpha=0.6, lw=0)
            ax.fill_between(x_plot, mean - band, mean + band,
                            color="tab:orange", alpha=0.45, lw=0)
            ax.plot(x_plot, mean, color="tab:orange", lw=1.8)
            ax.plot(x_plot, y_plot, "k-", lw=1.1)
            ax.plot(x_tr, y_tr, ".", color="tab:blue", ms=5)
            if row == 0:
                ax.set_title(METHODS[col])
            if row == len(N_ROWS) - 1:
                ax.set_xlabel("x")
        axes[row, 0].set_ylabel(f"N = {n}")

    axes[0, 0].set_xlim(-10, 10)
    axes[0, 0].set_ylim(-50, 50)

    handles = [
        Line2D([], [], color="k", lw=1.1, label="engine"),
        Line2D([], [], color="tab:blue", ls="", marker=".", ms=6,
               label="train"),
        Patch(facecolor="tab:orange", alpha=0.45,
              label=r"mean $\pm 2\sigma$"),
        Patch(facecolor="0.75", alpha=0.6, label="outer band"),
    ]
    axes[1, 0].legend(handles=handles, loc="lower right", frameon=False,
                      fontsize=10, handlelength=1.4, labelspacing=0.3,
                      borderaxespad=0.2)

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.outdir / "demo_workshop"
    fig.savefig(f"{stem}.png", dpi=200)
    fig.savefig(f"{stem}.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"figure -> {stem}.png / .pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
