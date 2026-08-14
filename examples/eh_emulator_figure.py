"""
Paper figure for the Eisenstein-Hu emulator example.

Imported by ``eh_emulator.py``; kept in its own module so the
figure-generation source contains only the statistics shown in the
figure. Layout: 1x3 panels, figsize (9.0, 3.2), dpi 200.

(a) 1D slice along omega_c through the box center at the smallest and
    largest N (two shared-y sub-panels): engine, fitted degree-2 mean,
    Bayesian Ridge +-4 sigma band, ellipse support band and PAC
    ensemble band. The Bayesian Ridge band visibly fails to cover the
    engine at small N and vanishes at large N, while the certified
    bands stay misspecification-limited.
(b) Test coverage vs N for the four methods, mean +- std over
    replicates, dotted reference at 1.
(c) PAC broadening of the support band vs N (left axis) and the
    converged ellipse band width relative to the data spread (right
    axis, muted).
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def make_figure(out_stem, slice_data, n_grid, coverage, broadening,
                rel_width):
    """Render and save the 3-panel figure.

    Parameters
    ----------
    out_stem : path-like
        Output stem; ``<stem>.png`` (dpi 200) and ``<stem>.pdf`` are
        written.

    slice_data : tuple
        ``(wc, y_engine, cells)`` along the omega_c slice, where
        ``cells`` is a list of two ``(n, curves)`` pairs (smallest and
        largest N) and ``curves`` maps ``mean``, ``e_lo``/``e_hi``
        (ellipse support), ``p_lo``/``p_hi`` (PAC ensemble) and
        ``b_lo``/``b_hi`` (Bayesian Ridge +-4 sigma) to arrays.

    n_grid : sequence of int
        Training sizes N (log x-axis of panels b and c).

    coverage : dict
        Keys ``'br', 'hc', 'e', 'pac'`` mapping to ``(mean, std)``
        arrays of test coverage over replicates, one entry per N.

    broadening : tuple of arrays
        ``(mean, std)`` of the PAC band broadening in percent, per N.

    rel_width : tuple of arrays
        ``(mean, std)`` of the ellipse band full width divided by
        std(y_test), per N.
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
    axes = [ax_a1, fig.add_subplot(gs[2]), fig.add_subplot(gs[3])]

    # ---- (a) the misspecification, physically, at small and large N ----
    wc, y_engine, cells = slice_data
    for ax, (n_sl, c) in zip((ax_a1, ax_a2), cells):
        ax.fill_between(wc, c["p_lo"], c["p_hi"], color="0.75", alpha=0.6,
                        lw=0)
        ax.fill_between(wc, c["e_lo"], c["e_hi"], color="tab:orange",
                        alpha=0.45, lw=0)
        ax.fill_between(wc, c["b_lo"], c["b_hi"], color="tab:green",
                        alpha=0.75, lw=0)
        ax.plot(wc, c["mean"], color="tab:orange", lw=1.4)
        ax.plot(wc, y_engine, "k-", lw=1.2)
        ax.set_xlim(wc[0], wc[-1])
        ax.set_xticks([0.08, 0.12, 0.16])
        ax.set_xlabel(r"$\omega_c$")
        ax.set_title(f"N = {n_sl}", fontsize=12)
    ax_a1.set_ylabel(r"$\ln P(k_*)$")
    ax_a2.tick_params(labelleft=False)

    # ---- (b) coverage vs N ---------------------------------------------
    ax = axes[1]
    for key, color, marker, label in [
        ("br", "tab:green", "s", r"Bayesian Ridge $\pm 4\sigma$"),
        ("hc", "tab:blue", "D", "POPS hypercube"),
        ("e", "tab:orange", "o", "POPS ellipse"),
        ("pac", "0.35", "^", "POPS ellipse + PAC"),
    ]:
        m, s = coverage[key]
        ax.errorbar(n_grid, m, yerr=s, color=color, marker=marker, ms=4.5,
                    lw=1.4, capsize=2, label=label)
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xticks(n_grid, [str(n) for n in n_grid])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("N")
    ax.set_ylabel("test coverage")
    ax.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.9,
              edgecolor="none", handlelength=1.6, labelspacing=0.35)

    # ---- (c) convergence of the hierarchical broadening ----------------
    ax = axes[2]
    m, s = broadening
    ax.errorbar(n_grid, m, yerr=s, color="0.35", marker="^", ms=4.5,
                lw=1.4, capsize=2)
    ax.set_xscale("log")
    ax.set_xticks(n_grid, [str(n) for n in n_grid])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.0, 58.0)
    ax.set_xlabel("N")
    ax.set_ylabel("PAC broadening (%)")

    axr = ax.twinx()
    mw, sw = rel_width
    axr.errorbar(n_grid, mw, yerr=sw, color="tab:orange", marker="o",
                 ms=4.5, lw=1.2, ls="--", mfc="none", capsize=2,
                 alpha=0.75)
    axr.set_ylim(0.7, 1.0)
    axr.set_ylabel(r"ellipse width / std$(y)$", color="tab:orange")
    axr.tick_params(axis="y", labelcolor="tab:orange")

    for ax, lab in zip(axes, "abc"):
        ax.text(0.03, 0.96, f"({lab})", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

    fig.savefig(f"{out_stem}.png", dpi=200)
    fig.savefig(f"{out_stem}.pdf", metadata={"CreationDate": None})
    plt.close(fig)
