"""
Paper figure for the Eisenstein-Hu emulator example.

Imported by ``eh_emulator.py``; kept in its own module so the
figure-generation source contains only the statistics shown in the
figure. Layout: panel (a) as two shared-y sub-panels plus panels (b)
and (c), figsize (9.0, 3.2), dpi 200.

(a) k-slice of the BAO wiggle ratio at one fixed held-out theta, at the
    smallest and largest N (two shared-y sub-panels): engine y(k),
    joint-fit mean (smooth envelope, structurally unable to oscillate
    at the BAO frequency), the 95.45% (+-2 sigma mass) predictive band
    of the ellipse pushforward (dark orange; exact equal-tail
    projected-ball quantile, analytic from the support width), the full
    ellipse support (light orange) and the PAC ensemble band (grey).
    All bands are from a single joint fit evaluated along the slice -
    one posterior, no pointwise-vs-joint caveat.
(b) Test coverage vs N for the four methods, mean +- std over
    replicates, dotted reference at 1.
(c) Predictive-std decomposition vs N/P (log-log): the posterior
    (within-ellipse pushforward) component is misspecification-limited
    and flat, while the hyperposterior (ensemble spread of the ellipse
    parameters) decays with N/P - the parameter uncertainty the
    hierarchical layer adds at small N/P.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from scipy.special import betaincinv

METHOD_STYLES = [
    ("br", "tab:green", "s", r"Bayesian Ridge $\pm 4\sigma$"),
    ("hc", "tab:blue", "D", "POPS hypercube"),
    ("e", "tab:orange", "o", "POPS ellipse"),
    ("pac", "0.35", "^", "POPS ellipse + PAC"),
]


def make_figure(out_stem, slice_data, n_grid, coverage, n_over_p,
                decomposition):
    """Render and save the figure.

    Parameters
    ----------
    out_stem : path-like
        Output stem; ``<stem>.png`` (dpi 200) and ``<stem>.pdf`` are
        written.

    slice_data : tuple
        ``(kg, y_engine, cells)`` along the k-slice at the held-out
        theta, where ``cells`` is a list of two ``(n, curves)`` pairs
        (smallest and largest N) and ``curves`` maps ``mean``,
        ``e_lo``/``e_hi`` (ellipse support) and ``p_lo``/``p_hi``
        (PAC ensemble) to arrays over the k grid.

    n_grid : sequence of int
        Training sizes N (log x-axis of panel b).

    coverage : dict
        Keys ``'br', 'hc', 'e', 'pac'`` mapping to ``(mean, std)``
        arrays of test coverage over replicates, one entry per N.

    n_over_p : array
        ``N / P`` for each grid value (x-axis of panel c).

    decomposition : dict
        Keys ``'post'`` and ``'hyper'`` mapping to ``(mean, std)``
        arrays of the predictive-std components in units of
        std(y_test), one entry per N.
    """
    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })
    fig = plt.figure(figsize=(9.0, 3.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.5, 1.5])
    ax_a1 = fig.add_subplot(gs[0])
    ax_a2 = fig.add_subplot(gs[1], sharey=ax_a1)
    ax_b = fig.add_subplot(gs[2])
    ax_c = fig.add_subplot(gs[3])

    # ---- (a) the misspecification, physically: the BAO wiggle ----------
    kg, y_engine, cells = slice_data
    # exact equal-tail 95.45% (+-2 sigma mass) quantile of the
    # projected-ball pushforward: P(|t| <= q sqrt(v)) = I_{q^2}(1/2,
    # (n_dim+1)/2); Gaussian limit 2/sqrt(n_dim+2), analytic from the
    # cached support half-width - no additional simulation
    P = int(round(n_grid[0] / n_over_p[0]))
    n_dim = P + 1
    q95 = float(np.sqrt(betaincinv(0.5, 0.5 * (n_dim + 1.0), 0.9545)))
    for ax, (n_sl, c) in zip((ax_a1, ax_a2), cells):
        half_sup = 0.5 * (c["e_hi"] - c["e_lo"])
        mid = 0.5 * (c["e_hi"] + c["e_lo"])
        ax.fill_between(kg, c["p_lo"], c["p_hi"], color="0.75", alpha=0.6,
                        lw=0)
        ax.fill_between(kg, c["e_lo"], c["e_hi"], color="tab:orange",
                        alpha=0.25, lw=0)
        ax.fill_between(kg, mid - q95 * half_sup, mid + q95 * half_sup,
                        color="tab:orange", alpha=0.65, lw=0)
        ax.plot(kg, c["mean"], color="tab:orange", lw=1.4)
        ax.plot(kg, y_engine, "k-", lw=1.2)
        ax.set_xlim(kg[0], kg[-1])
        ax.set_xticks([0.05, 0.2, 0.35])
        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        ax.set_title(f"N = {n_sl}", fontsize=12)
    # clip the small-N extrapolation flare at the k-box edges so the
    # BAO oscillation stays legible
    amp_a = 4.5 * float(abs(y_engine).max())
    ax_a1.set_ylim(-amp_a, amp_a)
    ax_a1.set_ylabel(r"$\ln[P / P_{\rm nw}]$")
    ax_a2.tick_params(labelleft=False)

    # ---- (b) coverage vs N ---------------------------------------------
    for key, color, marker, label in METHOD_STYLES:
        m, s = coverage[key]
        ax_b.errorbar(n_grid, m, yerr=s, color=color, marker=marker,
                      ms=4.5, lw=1.4, capsize=2, label=label)
    ax_b.axhline(1.0, color="k", ls=":", lw=1.0)
    ax_b.set_xscale("log")
    ax_b.set_xticks(n_grid, [str(n) for n in n_grid], rotation=35,
                    ha="right", rotation_mode="anchor")
    ax_b.xaxis.set_minor_formatter(NullFormatter())
    ax_b.set_ylim(0.0, 1.02)
    ax_b.set_xlabel("N")
    ax_b.set_ylabel("test coverage")
    ax_b.legend(loc="lower left", fontsize=9, frameon=True,
                framealpha=0.9, edgecolor="none", handlelength=1.6,
                labelspacing=0.35)

    # ---- (c) predictive-std decomposition vs N/P -----------------------
    for key, color, marker, mfc, label in [
        ("post", "tab:orange", "o", "tab:orange", "posterior"),
        ("hyper", "0.35", "^", "0.35", "hyperposterior"),
    ]:
        m, s = decomposition[key]
        ax_c.errorbar(n_over_p, m, yerr=s, color=color, marker=marker,
                      ms=4.5, lw=1.4, mfc=mfc, capsize=2, label=label)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    mh, sh = decomposition["hyper"]
    mp, sp = decomposition["post"]
    y_lo = 0.75 * float((mh - sh).min())
    y_hi = 1.35 * float((mp + sp).max())
    ax_c.set_ylim(y_lo, y_hi)
    ax_c.set_xticks(n_over_p, [f"{v:.0f}" for v in n_over_p])
    ax_c.xaxis.set_minor_formatter(NullFormatter())
    ticks = [t for t in (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
             if y_lo <= t <= y_hi]
    ax_c.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax_c.yaxis.set_minor_formatter(NullFormatter())
    ax_c.set_xlabel("N / P")
    ax_c.set_ylabel(r"predictive std / std$(y)$")
    ax_c.legend(loc="center right", fontsize=9, frameon=True,
                framealpha=0.9, edgecolor="none", handlelength=1.6,
                labelspacing=0.35)

    for ax, lab in zip((ax_a1, ax_b, ax_c), "abc"):
        ax.text(0.03, 0.96, f"({lab})", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

    fig.savefig(f"{out_stem}.png", dpi=200)
    fig.savefig(f"{out_stem}.pdf", metadata={"CreationDate": None})
    plt.close(fig)
