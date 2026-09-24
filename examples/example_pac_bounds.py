"""Ellipse+PAC bounds from the repeated-split study (supplementary figure).

Reads ``generated/repeated_splits.csv`` (from ``example_repeated_splits.py``)
and plots, for each problem against the number of independent units per
parameter:

- the PAC bound on the floor-contaminated log risk, its trivial ceiling
  ``log(R_y / beta)`` and the held-out floor-contaminated log loss of the same
  predictive (the certified quantity, estimated on the test set);
- the decomposition of the bound into the hyperposterior-averaged empirical
  term, its Monte Carlo correction and the remaining complexity term.

Medians and central 90% bands over training draws. Writes ``pac_bounds.png``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from example_repeated_splits import PROBLEM_TITLES
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

HERE = Path(__file__).resolve().parent


def band(ax, x, values, **style):
    q = np.array([np.nanquantile(v, [0.05, 0.5, 0.95]) for v in values])
    ax.plot(x, q[:, 1], marker="o", ms=2.5, **style)
    ax.fill_between(x, q[:, 0], q[:, 2], color=style.get("color"), alpha=0.15, lw=0)


def main():
    frame = pd.read_csv(HERE / "generated" / "repeated_splits.csv")
    pac = frame[(frame.method == "Ellipse+PAC") & ~frame.failed]
    problems = [p for p in PROBLEM_TITLES if p in set(pac.problem)]
    fig, axes = plt.subplots(2, len(problems), figsize=(10, 4.8), squeeze=False)
    for c, name in enumerate(problems):
        sub = pac[pac.problem == name]
        g = sub.groupby("units_per_param")
        x = np.array(sorted(g.groups))
        groups = [g.get_group(v) for v in x]
        ax = axes[0, c]
        band(ax, x, [s.bound for s in groups], color="C1", label="PAC bound")
        band(
            ax,
            x,
            [s.pac_test_risk for s in groups],
            color="k",
            label="held-out floored log loss",
        )
        ax.axhline(
            sub.trivial_bound.iloc[0],
            color="0.5",
            ls="--",
            lw=1,
            label=r"trivial ceiling $\log(R_y/\beta)$",
        )
        ax.set_title(PROBLEM_TITLES[name], fontsize=9)
        ax = axes[1, c]
        band(
            ax,
            x,
            [s.empirical for s in groups],
            color="C0",
            label=r"$\mathbb{E}_{\pi_H}\hat G$",
        )
        band(
            ax, x, [s.monte_carlo for s in groups], color="C2", label="Monte Carlo term"
        )
        band(
            ax,
            x,
            [s.complexity for s in groups],
            color="C3",
            label="complexity (kl inverse)",
        )
        for ax in axes[:, c]:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(x))
            ax.xaxis.set_major_formatter(FixedFormatter([f"{v:.2g}" for v in x]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(labelsize=7)
        axes[1, c].set_xlabel("independent units per parameter", fontsize=8)
    axes[0, 0].set_ylabel("floor-contaminated log risk [nats]", fontsize=8)
    axes[1, 0].set_ylabel("per-fold bound terms [nats]", fontsize=8)
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    axes[1, 0].legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout(h_pad=0.8)
    out = HERE / "pac_bounds.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
