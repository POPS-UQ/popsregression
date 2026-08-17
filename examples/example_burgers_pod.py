"""POD reduced-order Burgers emulator with controlled basis misspecification.

This matches the presentation and numerical setup of ``example_burgers.py``.
Truth is the same deterministic viscous Burgers solver, but the emulator uses a
rank-r POD reduced basis built only from a smoother offline regime.  The
training/test cases span the full parameter domain, so low-viscosity,
late-time solutions expose genuine reduced-basis extrapolation/truncation
error even when the coefficient regression is data rich.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

from popsregression import POPSRegression, POPSRegressionEllipse

import example_burgers as burgers


SEED = burgers.SEED
NU_RANGE = burgers.NU_RANGE
AMP_RANGE = burgers.AMP_RANGE
T_RANGE = burgers.T_RANGE
N_GRID = burgers.N_GRID

# Offline POD library: deliberately smoother than the full target domain.
POD_NU_RANGE = (0.040, NU_RANGE[1])
POD_T_RANGE = (T_RANGE[0], 0.55)


def build_pod_basis(rank=3, n_basis_cases=48, seed=SEED + 1000):
    """Build a fixed rank-r POD basis from smooth-regime snapshots only."""
    rng = np.random.default_rng(seed)
    cases = np.column_stack([
        rng.uniform(*POD_NU_RANGE, n_basis_cases),
        rng.uniform(*AMP_RANGE, n_basis_cases),
        rng.uniform(*POD_T_RANGE, n_basis_cases),
    ])

    snapshots = []
    for nu, amp, time in cases:
        _, u = burgers.burgers_solution(nu, amp, time)
        snapshots.append(u)
    snapshots = np.asarray(snapshots)

    mean_field = snapshots.mean(axis=0)
    centered = snapshots - mean_field[None, :]
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    modes = vt[:rank]
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    return mean_field, modes, singular_values, energy


def periodic_interp(values, x):
    grid = np.linspace(0.0, 2.0 * np.pi, values.shape[-1], endpoint=False)
    grid_ext = np.append(grid, 2.0 * np.pi)
    values_ext = np.concatenate([values, values[..., :1]], axis=-1)
    x_mod = np.mod(np.asarray(x), 2.0 * np.pi)
    if values.ndim == 1:
        return np.interp(x_mod, grid_ext, values_ext)
    return np.vstack([np.interp(x_mod, grid_ext, row) for row in values_ext])


def parameter_features(nu, amp, time):
    """Low-order map for POD modal amplitudes; six terms per mode."""
    ns = burgers.scale(nu, NU_RANGE)
    aa = burgers.scale(amp, AMP_RANGE)
    tt = burgers.scale(time, T_RANGE)
    return np.column_stack([
        np.ones_like(ns), ns, aa, tt, aa * tt, ns * tt,
    ])


def rom_features(nu, amp, time, x, modes):
    g = parameter_features(nu, amp, time)
    phi = periodic_interp(modes, x).T
    return np.einsum("ni,nj->nij", phi, g).reshape(len(g), -1)


def simulate_cases(cases, modes, mean_field, points_per_case=12):
    idx = np.linspace(0, N_GRID - 1, points_per_case, dtype=int)
    x_grid = np.linspace(0.0, 2.0 * np.pi, N_GRID, endpoint=False)
    rows, residuals, values = [], [], []

    for nu, amp, time in cases:
        _, u = burgers.burgers_solution(nu, amp, time)
        x = x_grid[idx]
        rows.append(rom_features(
            np.full(idx.size, nu), np.full(idx.size, amp),
            np.full(idx.size, time), x, modes
        ))
        offset = periodic_interp(mean_field, x)
        residuals.append(u[idx] - offset)
        values.append(u[idx])

    return np.vstack(rows), np.concatenate(residuals), np.concatenate(values)


def slice_design(theta, modes, mean_field):
    x = np.linspace(0.0, 2.0 * np.pi, N_GRID, endpoint=False)
    nu, amp, time = theta
    X = rom_features(
        np.full_like(x, nu), np.full_like(x, amp), np.full_like(x, time),
        x, modes
    )
    offset = periodic_interp(mean_field, x)
    _, truth = burgers.burgers_solution(nu, amp, time)
    return x, X, offset, truth


def run(seed=SEED, train_case_counts=(8, 16, 24, 40, 80), n_test_cases=80,
        pod_rank=3, n_basis_cases=48):
    rng = np.random.default_rng(seed)
    mean_field, modes, singular_values, energy = build_pod_basis(
        rank=pod_rank, n_basis_cases=n_basis_cases, seed=seed + 1000
    )

    all_train = burgers.draw_cases(rng, max(train_case_counts))
    test_cases = burgers.draw_cases(rng, n_test_cases)
    X_test, r_test, _ = simulate_cases(
        test_cases, modes, mean_field, points_per_case=16
    )
    p = X_test.shape[1]

    print(
        f"Burgers POD emulator: rank={pod_rank}, P={p}; "
        f"basis nu={POD_NU_RANGE}, t={POD_T_RANGE}"
    )
    print(f"POD cumulative snapshot energy at rank {pod_rank}: {energy[pod_rank-1]:.5f}")
    print("cases rows rows/P  BRcov4s Ellcov PACcov  PAC+%")

    fitted = {}
    for n_cases in train_case_counts:
        X_train, r_train, _ = simulate_cases(
            all_train[:n_cases], modes, mean_field, points_per_case=12
        )

        bayes = BayesianRidge(fit_intercept=False).fit(X_train, r_train)
        hypercube = POPSRegression(
            minimum_relative_error=0.0, posterior="hypercube"
        ).fit(X_train, r_train)
        ellipse = POPSRegressionEllipse(random_state=seed).fit(X_train, r_train)
        pac = POPSRegressionEllipse(
            random_state=seed, pac_bayes=True
        ).fit(X_train, r_train)

        b_mean = bayes.predict(X_test)
        b_std = burgers.epistemic_bayes_std(bayes, X_test)
        b_cov = burgers.coverage(r_test, b_mean - 4*b_std, b_mean + 4*b_std)

        e_mean, e_hi, e_lo = ellipse.predict(X_test, return_bounds=True)
        e_cov = burgers.coverage(r_test, e_lo, e_hi)

        _, p_hi, p_lo, p_bstd = pac.predict(
            X_test, return_bounds=True, return_bound_std=True
        )
        p_cov = burgers.coverage(r_test, p_lo, p_hi)
        bare_width = (p_hi - p_lo) - 4.0*p_bstd
        valid = bare_width > 1e-12
        broadening = np.mean(
            (p_hi[valid] - p_lo[valid]) / bare_width[valid] - 1.0
        )

        fitted[n_cases] = (bayes, hypercube, ellipse, pac)
        print(
            f"{n_cases:5d} {len(r_train):4d} {len(r_train)/p:6.2f}"
            f"   {b_cov:7.3f} {e_cov:6.3f} {p_cov:6.3f}"
            f" {100*broadening:6.1f}%"
        )

    # Same hard slice as example_burgers.py, but now outside the smooth POD
    # library in both viscosity and time.
    theta = (0.014, 1.15, 0.78)
    x, X_slice, offset, truth = slice_design(theta, modes, mean_field)

    shown_counts = (train_case_counts[0], train_case_counts[-1])
    fig, axes = plt.subplots(2, 4, figsize=(8, 3), sharex=True, sharey=True)
    titles = [
        "Bayesian Ridge", "POPS Hypercube", "POPS Ellipse",
        "POPS Ellipse + PAC"
    ]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=10)

    for row_idx, n_cases in enumerate(shown_counts):
        bayes, hypercube, ellipse, pac = fitted[n_cases]

        b_resid = bayes.predict(X_slice)
        b_mean = offset + b_resid
        b_std = burgers.epistemic_bayes_std(bayes, X_slice)

        h_resid, h_std, h_hi, h_lo = hypercube.predict(
            X_slice, return_std=True, return_bounds=True
        )
        h_mean = offset + h_resid

        e_resid, e_hi, e_lo = ellipse.predict(X_slice, return_bounds=True)
        e_qlo, e_qhi = burgers.bare_percentile_interval(ellipse, X_slice)
        e_mean = offset + e_resid

        p_resid, p_hi, p_lo = pac.predict(X_slice, return_bounds=True)
        p_qlo, p_qhi = burgers.pac_percentile_interval(pac, X_slice)
        p_mean = offset + p_resid

        ax = axes[row_idx, 0]
        ax.fill_between(
            x, b_mean - 4*b_std, b_mean + 4*b_std,
            alpha=0.20, facecolor="0.85", label=r"max/min ($\pm4\sigma$)"
        )
        ax.fill_between(
            x, b_mean - 2*b_std, b_mean + 2*b_std,
            alpha=0.45, facecolor="C1", label=r"$95.45\%$ ($\pm2\sigma$)"
        )
        ax.plot(x, b_mean, "C1-", lw=2, label="mean")

        ax = axes[row_idx, 1]
        ax.fill_between(
            x, offset + h_lo, offset + h_hi,
            alpha=0.20, facecolor="0.5", label="max/min"
        )
        ax.fill_between(
            x, h_mean - 2*h_std, h_mean + 2*h_std,
            alpha=0.45, facecolor="C1", label=r"$95.45\%$"
        )
        ax.plot(x, h_mean, "C1-", lw=2)

        ax = axes[row_idx, 2]
        ax.fill_between(
            x, offset + e_lo, offset + e_hi,
            alpha=0.20, facecolor="0.5", label="max/min"
        )
        ax.fill_between(
            x, offset + e_qlo, offset + e_qhi,
            alpha=0.45, facecolor="C1", label=r"$95.45\%$"
        )
        ax.plot(x, e_mean, "C1-", lw=2)

        ax = axes[row_idx, 3]
        ax.fill_between(
            x, offset + p_lo, offset + p_hi,
            alpha=0.20, facecolor="0.5", label="max/min"
        )
        ax.fill_between(
            x, offset + p_qlo, offset + p_qhi,
            alpha=0.45, facecolor="C1", label=r"$95.45\%$"
        )
        ax.plot(x, p_mean, "C1-", lw=2)

        for col in range(4):
            ax = axes[row_idx, col]
            truth_label = "Truth" if col == 2 else "_nolegend_"
            ax.plot(x, truth, "k-", lw=1.5, label=truth_label)
            ax.tick_params(labelsize=8)
            if row_idx == 1:
                ax.set_xlabel("x", fontsize=9)
            ax.set_ylim(-2, 2)

        axes[row_idx, 0].set_ylabel(f"N = {n_cases}\nu(x,t)", fontsize=9)

    # Same two-legend convention as example_burgers.py.
    br_handles, br_labels = axes[0, 0].get_legend_handles_labels()
    br_order = [2, 1, 0]
    axes[0, 0].legend(
        [br_handles[i] for i in br_order],
        [br_labels[i] for i in br_order],
        fontsize=7, loc="lower left"
    )

    pops_handles, pops_labels = axes[0, 2].get_legend_handles_labels()
    label_to_handle = dict(zip(pops_labels, pops_handles))
    pops_order = ["Truth", r"$95.45\%$", "max/min"]
    axes[0, 1].legend(
        [label_to_handle[label] for label in pops_order],
        pops_order, fontsize=7, loc="lower left"
    )

    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
    out = f"example_burgers_pod_r{pod_rank}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved {out}")
    return fitted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank", type=int, default=3, choices=(2, 3, 4, 5),
        help="number of retained POD spatial modes"
    )
    parser.add_argument(
        "--basis-cases", type=int, default=48,
        help="smooth-regime snapshots used to build the offline POD basis"
    )
    args = parser.parse_args()
    run(pod_rank=args.rank, n_basis_cases=args.basis_cases)
