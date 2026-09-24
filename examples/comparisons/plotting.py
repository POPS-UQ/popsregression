"""Shared band-plot style of the paper figures."""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUTER_COLOR = "#8FA4BF"
OUTER_ALPHA = 0.42
INNER_COLOR = "C1"
INNER_ALPHA = 0.45
LABEL_STYLE = dict(fontsize=6.5, ha="center", va="bottom")


def band_panel(ax, x, pred, truth, x_train=None, y_train=None):
    """Draw the 99.9% (outer) and 95.45% (inner) exact central intervals."""
    lo, hi = pred.intervals[0.999]
    ax.fill_between(x, lo, hi, color=OUTER_COLOR, alpha=OUTER_ALPHA, lw=0)
    lo, hi = pred.intervals[0.9545]
    ax.fill_between(x, lo, hi, color=INNER_COLOR, alpha=INNER_ALPHA, lw=0)
    ax.plot(x, truth, "k-", lw=1.3)
    ax.plot(x, pred.mean, color="C1", lw=1.6)
    if x_train is not None:
        ax.plot(x_train, y_train, "b.", ms=3.5)


def coverage_label(ax, y, pred):
    """Held-out coverage of both intervals, written above the panel."""
    parts = []
    for level, tag in ((0.9545, "95.45%"), (0.999, "99.9%")):
        lo, hi = pred.intervals[level]
        parts.append(f"{tag}: {np.mean((y >= lo) & (y <= hi)):.2f}")
    ax.text(
        0.5, 1.015, "cov. " + ", ".join(parts), transform=ax.transAxes, **LABEL_STYLE
    )


def legend_handles(train_label="training data"):
    return [
        Line2D([], [], color="k", lw=1.3, label="truth"),
        Line2D([], [], color="C1", lw=1.6, label="predictive mean"),
        Line2D([], [], color="b", marker=".", ls="", ms=5, label=train_label),
        Patch(color=INNER_COLOR, alpha=INNER_ALPHA, label="95.45% central interval"),
        Patch(color=OUTER_COLOR, alpha=OUTER_ALPHA, label="99.9% central interval"),
    ]
