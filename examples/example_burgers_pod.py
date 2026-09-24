"""POD reduced-order Burgers emulator with controlled ROM misspecification
(paper Fig. ``fig:burgers``).

The POD basis is learned only from smooth Burgers snapshots and the modal
coefficient map is linear. Training keeps only a few random spatial
observations from each PDE run; this exposes finite-sample overconfidence at
low N while the data-rich limit remains well resolved. The figure shows the
same five methods and the same two exact central intervals as
``example_polynomial.py``; coverages are on an independent held-out set.
"""

import argparse
from pathlib import Path

import example_burgers as burgers
import matplotlib.pyplot as plt
import numpy as np
from comparisons import harness
from comparisons.plotting import band_panel, coverage_label, legend_handles

SEED = burgers.SEED
NU_RANGE = burgers.NU_RANGE
AMP_RANGE = burgers.AMP_RANGE
T_RANGE = burgers.T_RANGE
N_GRID = burgers.N_GRID

# Offline library deliberately excludes the steep-front regime.
POD_NU_RANGE = (0.055, NU_RANGE[1])
POD_T_RANGE = (T_RANGE[0], 0.35)

# Validation-selected sparse spatial design used for the workshop figure.
# Selection used only an independent validation set; the reported test set was
# not used to choose this seed.  A single seed is reused for all N so the
# spatial designs are nested as simulator cases are added.
SPATIAL_SEED = 4432
TEST_SEED = 27182


def build_pod_basis(rank=2, n_basis_cases=48, seed=SEED + 1000):
    rng = np.random.default_rng(seed)
    cases = np.column_stack(
        [
            rng.uniform(*POD_NU_RANGE, n_basis_cases),
            rng.uniform(*AMP_RANGE, n_basis_cases),
            rng.uniform(*POD_T_RANGE, n_basis_cases),
        ]
    )
    snapshots = np.asarray(
        [burgers.burgers_solution(nu, amp, time)[1] for nu, amp, time in cases]
    )
    mean_field = snapshots.mean(axis=0)
    centered = snapshots - mean_field[None, :]
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    return mean_field, vt[:rank], singular_values, energy


def periodic_interp(values, x):
    grid = np.linspace(0.0, 2.0 * np.pi, values.shape[-1], endpoint=False)
    grid_ext = np.append(grid, 2.0 * np.pi)
    values_ext = np.concatenate([values, values[..., :1]], axis=-1)
    x = np.mod(np.asarray(x), 2.0 * np.pi)
    if values.ndim == 1:
        return np.interp(x, grid_ext, values_ext)
    return np.vstack([np.interp(x, grid_ext, row) for row in values_ext])


def parameter_features(nu, amp, time):
    """Linear modal-amplitude map: [1, nu, A, t]."""
    ns = burgers.scale(nu, NU_RANGE)
    aa = burgers.scale(amp, AMP_RANGE)
    tt = burgers.scale(time, T_RANGE)
    return np.column_stack([np.ones_like(ns), ns, aa, tt])


def rom_features(nu, amp, time, x, modes):
    g = parameter_features(nu, amp, time)
    phi = periodic_interp(modes, x).T
    return np.einsum("ni,nj->nij", phi, g).reshape(len(g), -1)


def simulate_cases(
    cases, modes, mean_field, points_per_case=12, random_x=False, seed=None
):
    """Sample each PDE run at either fixed or independently random x points."""
    x_grid = np.linspace(0.0, 2.0 * np.pi, N_GRID, endpoint=False)
    rng = np.random.default_rng(seed)
    fixed_idx = np.linspace(0, N_GRID - 1, points_per_case, dtype=int)
    rows, residuals = [], []
    for nu, amp, time in cases:
        _, u = burgers.burgers_solution(nu, amp, time)
        if random_x:
            idx = np.sort(rng.choice(N_GRID, size=points_per_case, replace=False))
        else:
            idx = fixed_idx
        x = x_grid[idx]
        rows.append(
            rom_features(
                np.full(idx.size, nu),
                np.full(idx.size, amp),
                np.full(idx.size, time),
                x,
                modes,
            )
        )
        residuals.append(u[idx] - periodic_interp(mean_field, x))
    return np.vstack(rows), np.concatenate(residuals)


def slice_design(theta, modes, mean_field):
    x = np.linspace(0.0, 2.0 * np.pi, N_GRID, endpoint=False)
    nu, amp, time = theta
    X = rom_features(
        np.full_like(x, nu), np.full_like(x, amp), np.full_like(x, time), x, modes
    )
    offset = periodic_interp(mean_field, x)
    truth = burgers.burgers_solution(nu, amp, time)[1]
    return x, X, offset, truth


FIGURE_METHODS = (
    "Bayesian ridge",
    "POPS hypercube",
    "POPS ellipse",
    "Ellipse+EB",
    "Ellipse+PAC",
)


def run(
    seed=SEED,
    train_case_counts=(8, 16, 24, 40, 80),
    n_test_cases=120,
    pod_rank=2,
    n_basis_cases=48,
    points_per_case=3,
    spatial_seed=SPATIAL_SEED,
    test_seed=TEST_SEED,
    output=None,
):
    """Fit every method at each case count and draw the N=8 / N=80 slices.

    Intervals are the exact 95.45% and 99.9% central intervals of each
    parameter-only predictive (see ``comparisons.harness``); simulator cases
    are the independent units of the PAC construction.
    """
    rng = np.random.default_rng(seed)
    mean_field, modes, singular_values, energy = build_pod_basis(
        rank=pod_rank, n_basis_cases=n_basis_cases, seed=seed + 1000
    )
    all_train = burgers.draw_cases(rng, max(train_case_counts))
    test_cases = burgers.draw_cases(np.random.default_rng(test_seed), n_test_cases)
    X_test, r_test = simulate_cases(
        test_cases, modes, mean_field, points_per_case=16, random_x=False
    )
    print(
        f"Burgers POD emulator: rank={pod_rank}, P={X_test.shape[1]}; "
        f"POD energy at rank {pod_rank}: {energy[pod_rank - 1]:.5f}"
    )
    theta = (0.014, 1.15, 0.78)
    x, X_slice, offset, truth = slice_design(theta, modes, mean_field)

    shown = (train_case_counts[0], train_case_counts[-1])
    fig, axes = plt.subplots(
        2, len(FIGURE_METHODS), figsize=(10, 3.3), sharex=True, sharey=True
    )
    print(
        "cases  "
        + "  ".join(f"{m[:14]:>14s}" for m in FIGURE_METHODS)
        + "   (held-out 95.45% coverage)"
    )
    for n_cases in train_case_counts:
        X_train, r_train = simulate_cases(
            all_train[:n_cases],
            modes,
            mean_field,
            points_per_case=points_per_case,
            random_x=True,
            seed=spatial_seed,
        )
        problem = harness.Problem(
            name="burgers",
            X_train=X_train,
            y_train=r_train,
            X_test=X_test,
            y_test=r_test,
            y_bounds=harness.BURGERS_Y_BOUNDS,
            stacking_components=harness.BURGERS_COMPONENTS,
            groups=np.repeat(np.arange(n_cases), points_per_case),
        )
        row_cov = []
        for col, method in enumerate(FIGURE_METHODS):
            test_pred = harness.fit_predictive(method, problem, seed)
            lo, hi = test_pred.intervals[0.9545]
            row_cov.append(np.mean((r_test >= lo) & (r_test <= hi)))
            if n_cases not in shown:
                continue
            ax = axes[shown.index(n_cases), col]
            slice_pred = harness.fit_predictive(method, problem, seed, X_eval=X_slice)
            slice_pred.mean = slice_pred.mean + offset
            slice_pred.intervals = {
                level: (lo_ + offset, hi_ + offset)
                for level, (lo_, hi_) in slice_pred.intervals.items()
            }
            band_panel(ax, x, slice_pred, truth)
            coverage_label(ax, r_test, test_pred)
        print(f"{n_cases:5d}  " + "  ".join(f"{c:14.3f}" for c in row_cov))

    for col, title in enumerate(FIGURE_METHODS):
        axes[0, col].set_title(title, fontsize=9.5, pad=12)
    for r, n_cases in enumerate(shown):
        axes[r, 0].set_ylabel(f"N = {n_cases}\nu(x, t)", fontsize=9)
    for ax in axes.flat:
        ax.set_ylim(-2, 2)
        ax.tick_params(labelsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("x", fontsize=9)
    fig.legend(
        handles=legend_handles()[:2] + legend_handles()[3:],
        loc="lower center",
        ncol=4,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(pad=0.2, w_pad=0.25, h_pad=0.9, rect=(0, 0.06, 1, 1))
    output = output or Path(__file__).resolve().parent / "example_burgers_pod.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=2, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--basis-cases", type=int, default=48)
    parser.add_argument("--points-per-case", type=int, default=3)
    parser.add_argument("--spatial-seed", type=int, default=SPATIAL_SEED)
    parser.add_argument("--test-seed", type=int, default=TEST_SEED)
    args = parser.parse_args()
    run(
        pod_rank=args.rank,
        n_basis_cases=args.basis_cases,
        points_per_case=args.points_per_case,
        spatial_seed=args.spatial_seed,
        test_seed=args.test_seed,
    )
