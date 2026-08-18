"""POD reduced-order Burgers emulator with controlled ROM misspecification.

Matches ``example_burgers.py`` in solver and plotting style.  The POD basis is
learned only from smooth Burgers snapshots and the modal coefficient map is
linear.  Training keeps only a few random spatial observations from each PDE
run; this exposes finite-sample overconfidence at low N while the data-rich
limit remains well resolved.
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

# Offline library deliberately excludes the steep-front regime.
POD_NU_RANGE = (0.055, NU_RANGE[1])
POD_T_RANGE = (T_RANGE[0], 0.35)

# Validation-selected sparse spatial design used for the workshop figure.
# Selection used only an independent validation set; the reported test set was
# not used to choose this seed.  A single seed is reused for all N so the
# spatial designs are nested as simulator cases are added.
SPATIAL_SEED = 4432
TEST_SEED = 27182

# Plot styling: use a distinct cool outer band so max/min is visually
# separable from the central 95.45% interval.
OUTER_COLOR = "#8FA4BF"
OUTER_ALPHA = 0.42
INNER_COLOR = "C1"
INNER_ALPHA = 0.45
COVERAGE_BBOX = dict(boxstyle="round,pad=0.18", fc="white", ec="0.65", alpha=0.90)


def build_pod_basis(rank=2, n_basis_cases=48, seed=SEED + 1000):
    rng = np.random.default_rng(seed)
    cases = np.column_stack([
        rng.uniform(*POD_NU_RANGE, n_basis_cases),
        rng.uniform(*AMP_RANGE, n_basis_cases),
        rng.uniform(*POD_T_RANGE, n_basis_cases),
    ])
    snapshots = np.asarray([
        burgers.burgers_solution(nu, amp, time)[1]
        for nu, amp, time in cases
    ])
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


def simulate_cases(cases, modes, mean_field, points_per_case=12,
                   random_x=False, seed=None):
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
        rows.append(rom_features(
            np.full(idx.size, nu), np.full(idx.size, amp),
            np.full(idx.size, time), x, modes
        ))
        residuals.append(u[idx] - periodic_interp(mean_field, x))
    return np.vstack(rows), np.concatenate(residuals)


def slice_design(theta, modes, mean_field):
    x = np.linspace(0.0, 2.0 * np.pi, N_GRID, endpoint=False)
    nu, amp, time = theta
    X = rom_features(
        np.full_like(x, nu), np.full_like(x, amp), np.full_like(x, time),
        x, modes
    )
    offset = periodic_interp(mean_field, x)
    truth = burgers.burgers_solution(nu, amp, time)[1]
    return x, X, offset, truth


def run(seed=SEED, train_case_counts=(8, 16, 24, 40, 80), n_test_cases=120,
        pod_rank=2, n_basis_cases=48, points_per_case=3,
        spatial_seed=SPATIAL_SEED, test_seed=TEST_SEED):
    rng = np.random.default_rng(seed)
    mean_field, modes, singular_values, energy = build_pod_basis(
        rank=pod_rank, n_basis_cases=n_basis_cases, seed=seed + 1000
    )

    all_train = burgers.draw_cases(rng, max(train_case_counts))
    test_cases = burgers.draw_cases(np.random.default_rng(test_seed), n_test_cases)
    X_test, r_test = simulate_cases(
        test_cases, modes, mean_field, points_per_case=16, random_x=False
    )
    p = X_test.shape[1]

    print(
        f"Burgers POD emulator: rank={pod_rank}, P={p}; "
        f"basis nu={POD_NU_RANGE}, t={POD_T_RANGE}; "
        f"random training x, {points_per_case} points/case; "
        f"spatial_seed={spatial_seed}"
    )
    print(f"POD cumulative snapshot energy at rank {pod_rank}: {energy[pod_rank-1]:.5f}")
    print(
        "cases rows rows/P  BRcov4s  Hcov  Ellcov PACcov  PAC+%   "
        "slice H/E/P max"
    )

    theta = (0.014, 1.15, 0.78)
    x, X_slice, offset, truth = slice_design(theta, modes, mean_field)
    target_slice = truth - offset

    fitted = {}
    coverages = {}
    for n_cases in train_case_counts:
        X_train, r_train = simulate_cases(
            all_train[:n_cases], modes, mean_field,
            points_per_case=points_per_case, random_x=True,
            seed=spatial_seed,
        )
        bayes = BayesianRidge(fit_intercept=False).fit(X_train, r_train)
        hypercube = POPSRegression(
            minimum_relative_error=0.0, posterior="hypercube"
        ).fit(X_train, r_train)
        ellipse = POPSRegressionEllipse(random_state=seed).fit(X_train, r_train)
        pac = POPSRegressionEllipse(random_state=seed, pac_bayes=True).fit(
            X_train, r_train
        )

        b_mean = bayes.predict(X_test)
        b_std = burgers.epistemic_bayes_std(bayes, X_test)
        b_cov = burgers.coverage(r_test, b_mean - 4*b_std, b_mean + 4*b_std)

        _, _, h_hi_test, h_lo_test = hypercube.predict(
            X_test, return_std=True, return_bounds=True
        )
        h_cov = burgers.coverage(r_test, h_lo_test, h_hi_test)

        _, e_hi_test, e_lo_test = ellipse.predict(X_test, return_bounds=True)
        e_cov = burgers.coverage(r_test, e_lo_test, e_hi_test)

        _, p_hi_test, p_lo_test, p_bstd = pac.predict(
            X_test, return_bounds=True, return_bound_std=True
        )
        p_cov = burgers.coverage(r_test, p_lo_test, p_hi_test)
        bare_width = (p_hi_test - p_lo_test) - 4.0*p_bstd
        valid = bare_width > 1e-12
        broadening = np.mean(
            (p_hi_test[valid] - p_lo_test[valid]) / bare_width[valid] - 1.0
        )

        _, _, h_hi_slice, h_lo_slice = hypercube.predict(
            X_slice, return_std=True, return_bounds=True
        )
        _, e_hi_slice, e_lo_slice = ellipse.predict(X_slice, return_bounds=True)
        _, p_hi_slice, p_lo_slice = pac.predict(X_slice, return_bounds=True)
        h_slice_cov = burgers.coverage(target_slice, h_lo_slice, h_hi_slice)
        e_slice_cov = burgers.coverage(target_slice, e_lo_slice, e_hi_slice)
        p_slice_cov = burgers.coverage(target_slice, p_lo_slice, p_hi_slice)

        coverages[n_cases] = {
            "bayes": b_cov,
            "hyper": h_cov,
            "ellipse": e_cov,
            "pac": p_cov,
        }
        fitted[n_cases] = (bayes, hypercube, ellipse, pac)
        print(
            f"{n_cases:5d} {len(r_train):4d} {len(r_train)/p:6.2f}"
            f"   {b_cov:7.3f} {h_cov:5.3f} {e_cov:6.3f} {p_cov:6.3f}"
            f" {100*broadening:6.1f}%   "
            f"{h_slice_cov:4.2f}/{e_slice_cov:4.2f}/{p_slice_cov:4.2f}"
        )

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
        ax.fill_between(x, b_mean-4*b_std, b_mean+4*b_std,
                        alpha=OUTER_ALPHA, facecolor=OUTER_COLOR,
                        label=r"max/min ($\pm4\sigma$)")
        ax.fill_between(x, b_mean-2*b_std, b_mean+2*b_std,
                        alpha=INNER_ALPHA, facecolor=INNER_COLOR,
                        label=r"$95.45\%$ ($\pm2\sigma$)")
        ax.plot(x, b_mean, "C1-", lw=2, label="mean")

        ax = axes[row_idx, 1]
        ax.fill_between(x, offset+h_lo, offset+h_hi,
                        alpha=OUTER_ALPHA, facecolor=OUTER_COLOR, label="max/min")
        ax.fill_between(x, h_mean-2*h_std, h_mean+2*h_std,
                        alpha=INNER_ALPHA, facecolor=INNER_COLOR, label=r"$95.45\%$")
        ax.plot(x, h_mean, "C1-", lw=2)

        ax = axes[row_idx, 2]
        ax.fill_between(x, offset+e_lo, offset+e_hi,
                        alpha=OUTER_ALPHA, facecolor=OUTER_COLOR, label="max/min")
        ax.fill_between(x, offset+e_qlo, offset+e_qhi,
                        alpha=INNER_ALPHA, facecolor=INNER_COLOR, label=r"$95.45\%$")
        ax.plot(x, e_mean, "C1-", lw=2)

        ax = axes[row_idx, 3]
        ax.fill_between(x, offset+p_lo, offset+p_hi,
                        alpha=OUTER_ALPHA, facecolor=OUTER_COLOR, label="max/min")
        ax.fill_between(x, offset+p_qlo, offset+p_qhi,
                        alpha=INNER_ALPHA, facecolor=INNER_COLOR, label=r"$95.45\%$")
        ax.plot(x, p_mean, "C1-", lw=2)

        # Annotate held-out coverage of the outer interval. These values come
        # from the independent test set, not the displayed slice.
        cov = coverages[n_cases]
        coverage_text = [
            rf"$4\sigma$ cov. = {cov['bayes']:.3f}",
            f"cov. = {cov['hyper']:.3f}",
            f"cov. = {cov['ellipse']:.3f}",
            f"cov. = {cov['pac']:.3f}",
        ]
        for col, text in enumerate(coverage_text):
            axes[row_idx, col].text(
                0.97, 0.95, text,
                transform=axes[row_idx, col].transAxes,
                ha="right", va="top", fontsize=6.5,
                bbox=COVERAGE_BBOX, zorder=10,
            )

        for col in range(4):
            ax = axes[row_idx, col]
            truth_label = "Truth" if col == 2 else "_nolegend_"
            ax.plot(x, truth, "k-", lw=1.5, label=truth_label)
            ax.tick_params(labelsize=8)
            if row_idx == 1:
                ax.set_xlabel("x", fontsize=9)
            ax.set_ylim(-2, 2)
        axes[row_idx, 0].set_ylabel(f"N = {n_cases}\nu(x,t)", fontsize=9)

    br_handles, br_labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(
        [br_handles[i] for i in [2, 1, 0]],
        [br_labels[i] for i in [2, 1, 0]],
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
    stem = (
        f"example_burgers_pod_randomx_r{pod_rank}_m{points_per_case}"
        f"_s{spatial_seed}"
    )
    fig.savefig(stem + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    print(f"Saved {stem}.png and {stem}.pdf")
    return fitted


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
