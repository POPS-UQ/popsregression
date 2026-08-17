"""POD reduced-order Burgers emulator for the Sim2Science workshop.

Truth is a deterministic periodic viscous Burgers solver.  The emulator uses a
fixed offline POD basis and a low-order polynomial map from (nu, A, t) to POD
modal amplitudes.  A small retained POD rank therefore leaves genuine
low-rank ROM truncation error even as the regression data increase.

The figure matches the workshop style used by the polynomial and harmonic
Burgers examples: rows are training-set sizes and columns are BayesianRidge,
POPS Hypercube, POPS Ellipse, and POPS Ellipse + PAC.  Only two legends are
shown for readability.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import gammaln
from scipy.stats import beta
from sklearn.linear_model import BayesianRidge

from popsregression import POPSRegression, POPSRegressionEllipse


SEED = 7
NU_RANGE = (0.012, 0.08)
AMP_RANGE = (0.7, 1.3)
T_RANGE = (0.25, 0.85)
N_GRID = 96


def burgers_solution(nu, amplitude, t_final, n_grid=N_GRID):
    """Periodic viscous Burgers solution from u(x,0)=A sin(x)."""
    x = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    dx = x[1] - x[0]
    u0 = amplitude * np.sin(x)

    def rhs(_, u):
        flux = 0.5 * u**2
        flux_x = (np.roll(flux, -1) - np.roll(flux, 1)) / (2.0 * dx)
        u_xx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2
        return -flux_x + nu * u_xx

    sol = solve_ivp(
        rhs,
        (0.0, t_final),
        u0,
        method="RK45",
        rtol=2e-6,
        atol=2e-8,
        max_step=0.01,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return x, sol.y[:, -1]


def scale(v, limits):
    lo, hi = limits
    return 2.0 * (np.asarray(v) - lo) / (hi - lo) - 1.0


def draw_cases(rng, n):
    return np.column_stack(
        [
            rng.uniform(*NU_RANGE, n),
            rng.uniform(*AMP_RANGE, n),
            rng.uniform(*T_RANGE, n),
        ]
    )


def build_pod_basis(rank=3, n_basis_cases=48, seed=SEED + 1000):
    """Build a fixed offline POD basis from an independent snapshot library."""
    rng = np.random.default_rng(seed)
    cases = draw_cases(rng, n_basis_cases)
    snapshots = []
    for nu, amp, time in cases:
        _, u = burgers_solution(nu, amp, time)
        snapshots.append(u)
    snapshots = np.asarray(snapshots)
    mean_field = snapshots.mean(axis=0)
    centered = snapshots - mean_field[None, :]
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    modes = vt[:rank]
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    return mean_field, modes, singular_values, energy


def periodic_interp(values, x):
    """Periodic linear interpolation of a field stored on the solver grid."""
    grid = np.linspace(0.0, 2.0 * np.pi, values.shape[-1], endpoint=False)
    grid_ext = np.append(grid, 2.0 * np.pi)
    values_ext = np.concatenate([values, values[..., :1]], axis=-1)
    x_mod = np.mod(np.asarray(x), 2.0 * np.pi)
    if values.ndim == 1:
        return np.interp(x_mod, grid_ext, values_ext)
    return np.vstack(
        [np.interp(x_mod, grid_ext, row) for row in values_ext]
    )


def parameter_features(nu, amp, time):
    """Small coefficient map; six terms per retained POD mode."""
    ns = scale(nu, NU_RANGE)
    aa = scale(amp, AMP_RANGE)
    tt = scale(time, T_RANGE)
    return np.column_stack(
        [
            np.ones_like(ns),
            ns,
            aa,
            tt,
            aa * tt,
            ns * tt,
        ]
    )


def rom_features(nu, amp, time, x, modes):
    """Tensor product of parameter features and retained POD modes."""
    g = parameter_features(nu, amp, time)
    phi = periodic_interp(modes, x).T
    # One global linear model: sum_j sum_m beta_jm phi_j(x) g_m(theta).
    return np.einsum("ni,nj->nij", phi, g).reshape(len(g), -1)


def mean_at_x(mean_field, x):
    return periodic_interp(mean_field, x)


def simulate_cases(cases, modes, mean_field, points_per_case=12):
    """Sparse field observations and corresponding POD-ROM regression rows."""
    idx = np.linspace(0, N_GRID - 1, points_per_case, dtype=int)
    rows, residuals, values = [], [], []
    x_grid = np.linspace(0.0, 2.0 * np.pi, N_GRID, endpoint=False)
    for nu, amp, time in cases:
        _, u = burgers_solution(nu, amp, time)
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
        values.append(u[idx])
        residuals.append(u[idx] - mean_at_x(mean_field, x))
    return np.vstack(rows), np.concatenate(residuals), np.concatenate(values)


def slice_design(theta, modes, mean_field):
    x = np.linspace(0.0, 2.0 * np.pi, N_GRID, endpoint=False)
    nu, amp, time = theta
    X = rom_features(
        np.full_like(x, nu),
        np.full_like(x, amp),
        np.full_like(x, time),
        x,
        modes,
    )
    offset = mean_at_x(mean_field, x)
    _, truth = burgers_solution(nu, amp, time)
    return x, X, offset, truth


def epistemic_bayes_std(model, X):
    return np.sqrt(np.einsum("ij,jk,ik->i", X, model.sigma_, X))


def coverage(y, lo, hi):
    return np.mean((y >= lo) & (y <= hi))


def projected_ball_fraction(dim, half_percentile=0.5 - 0.023):
    """Support fraction for the central 95.45% scalar ball projection."""
    a = 0.5 * (dim + 1.0)
    return 2.0 * beta.ppf(0.5 + half_percentile, a, a) - 1.0


def bare_percentile_interval(model, X, half_percentile=0.5 - 0.023):
    mean = model.predict(X)
    Xc, Z = model._whitened_design(np.asarray(X, dtype=float))
    v = model._squared_widths(Xc, Z) + model.delta**2
    q = projected_ball_fraction(model._ball_dim, half_percentile)
    hw = q * np.sqrt(v)
    return mean - hw, mean + hw


def pac_percentile_interval(model, X, half_percentile=0.5 - 0.023):
    """Central interval with same 2-sigma hyperposterior envelope as max/min."""
    mean = model.predict(X)
    Xc, Z = model._whitened_design(np.asarray(X, dtype=float))
    v = model._squared_widths(Xc, Z) + model.delta**2
    q = projected_ball_fraction(model._ball_dim, half_percentile)
    Z2 = Z * Z
    sigma_u_proj = Z2 @ model._sigma_U
    mean_var = Z2 @ model._sigma_c
    var_v = 4.0 * np.sum((Z @ model.U_) ** 2 * sigma_u_proj, axis=1)
    q_bound_std = np.sqrt(mean_var + q * q * var_v / (4.0 * v))
    hw = q * np.sqrt(v) + 2.0 * q_bound_std
    return mean - hw, mean + hw


def projected_ball_nll(y, mean, lo, hi, dim, delta=1e-3):
    radius = 0.5 * (hi - lo)
    v = np.maximum(radius**2, 0.0) + delta**2
    q = 1.0 - (y - mean) ** 2 / v
    inside = q > 0.0
    if not np.all(inside):
        return np.inf, int(np.sum(~inside))
    log_c = (
        gammaln(0.5 * dim + 1.0)
        - 0.5 * np.log(np.pi)
        - gammaln(0.5 * (dim + 1.0))
    )
    k = 0.5 * (dim - 1.0)
    return float(np.mean(0.5 * np.log(v) - log_c - k * np.log(q))), 0


def run(
    seed=SEED,
    train_case_counts=(10, 16, 24, 40, 80),
    n_test_cases=80,
    pod_rank=3,
    n_basis_cases=48,
):
    rng = np.random.default_rng(seed)
    mean_field, modes, singular_values, energy = build_pod_basis(
        rank=pod_rank, n_basis_cases=n_basis_cases, seed=seed + 1000
    )
    all_train = draw_cases(rng, max(train_case_counts))
    test_cases = draw_cases(rng, n_test_cases)

    X_test, r_test, y_test = simulate_cases(
        test_cases, modes, mean_field, points_per_case=16
    )
    p = X_test.shape[1]
    uniform_nll = float(np.log(np.ptp(r_test)))

    print(
        f"Burgers POD emulator: rank={pod_rank}, P={p}, "
        f"offline basis cases={n_basis_cases}; deterministic; no noise"
    )
    print(f"POD cumulative snapshot energy at rank {pod_rank}: {energy[pod_rank-1]:.5f}")
    print(f"Uniform-reference residual NLL: {uniform_nll:.3f} nats")
    print(
        "cases rows rows/P  BRcov4s HycCov EllCov PACCov  PAC+%   "
        "G_test   bound   gap  cert"
    )

    records, fitted = [], {}
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
        b_std = epistemic_bayes_std(bayes, X_test)
        b_cov = coverage(r_test, b_mean - 4 * b_std, b_mean + 4 * b_std)

        _, h_hi, h_lo = hypercube.predict(X_test, return_bounds=True)
        h_cov = coverage(r_test, h_lo, h_hi)

        e_mean, e_hi, e_lo = ellipse.predict(X_test, return_bounds=True)
        e_cov = coverage(r_test, e_lo, e_hi)
        g_test, uncovered = projected_ball_nll(
            r_test, e_mean, e_lo, e_hi, ellipse._ball_dim, ellipse.delta
        )

        _, p_hi, p_lo, p_bstd = pac.predict(
            X_test, return_bounds=True, return_bound_std=True
        )
        p_cov = coverage(r_test, p_lo, p_hi)
        bare_width = (p_hi - p_lo) - 4.0 * p_bstd
        valid = bare_width > 1e-12
        broadening = np.mean(
            (p_hi[valid] - p_lo[valid]) / bare_width[valid] - 1.0
        )
        gap = pac.bound_ - g_test
        certified = np.isfinite(pac.bound_) and pac.bound_ < uniform_nll

        records.append(
            dict(
                cases=n_cases,
                n=len(r_train),
                ratio=len(r_train) / p,
                bayes_coverage=b_cov,
                hypercube_coverage=h_cov,
                ellipse_coverage=e_cov,
                pac_coverage=p_cov,
                pac_broadening=broadening,
                g_test=g_test,
                uncovered=uncovered,
                uniform_nll=uniform_nll,
                bound=pac.bound_,
                bound_gap=gap,
                nonvacuous=certified,
                objective=pac.objective_,
                kl=pac.kl_,
            )
        )
        fitted[n_cases] = (bayes, hypercube, ellipse, pac)
        g_txt = " inf" if not np.isfinite(g_test) else f"{g_test:7.3f}"
        print(
            f"{n_cases:5d} {len(r_train):4d} {len(r_train)/p:6.2f}"
            f"   {b_cov:7.3f} {h_cov:6.3f} {e_cov:6.3f} {p_cov:6.3f}"
            f" {100*broadening:6.1f}% {g_txt} {pac.bound_:7.3f}"
            f" {gap:6.3f}   {'YES' if certified else 'no'}"
        )
        if uncovered:
            print(
                f"      note: {uncovered}/{len(r_test)} test points outside bare support"
            )

    # A hard low-viscosity/late-time slice emphasizes low-rank ROM error.
    theta = (0.014, 1.15, 0.78)
    x, X_slice, offset, truth = slice_design(theta, modes, mean_field)

    shown_counts = (train_case_counts[0], train_case_counts[-1])
    fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
    titles = [
        "Bayesian Ridge",
        "POPS Hypercube",
        "POPS Ellipse",
        "POPS Ellipse + PAC",
    ]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=10)

    for row_idx, n_cases in enumerate(shown_counts):
        bayes, hypercube, ellipse, pac = fitted[n_cases]

        # Bayesian Ridge.
        b_resid = bayes.predict(X_slice)
        b_mean = offset + b_resid
        b_std = epistemic_bayes_std(bayes, X_slice)
        ax = axes[row_idx, 0]
        ax.fill_between(
            x,
            b_mean - 4 * b_std,
            b_mean + 4 * b_std,
            alpha=0.20,
            facecolor="0.85",
            label=r"max/min ($\pm4\sigma$)",
        )
        ax.fill_between(
            x,
            b_mean - 2 * b_std,
            b_mean + 2 * b_std,
            alpha=0.45,
            facecolor="C1",
            label=r"$95.45\%$ ($\pm2\sigma$)",
        )
        ax.plot(x, b_mean, "C1-", lw=2, label="mean")

        # POPS Hypercube.
        h_resid, h_std, h_hi, h_lo = hypercube.predict(
            X_slice, return_std=True, return_bounds=True
        )
        h_mean = offset + h_resid
        ax = axes[row_idx, 1]
        ax.fill_between(
            x, offset + h_lo, offset + h_hi,
            alpha=0.20, facecolor="0.85", label="max/min"
        )
        ax.fill_between(
            x, h_mean - 2 * h_std, h_mean + 2 * h_std,
            alpha=0.45, facecolor="C1", label=r"$95.45\%$"
        )
        ax.plot(x, h_mean, "C1-", lw=2)

        # POPS Ellipse.
        e_resid, e_hi, e_lo = ellipse.predict(X_slice, return_bounds=True)
        e_qlo, e_qhi = bare_percentile_interval(ellipse, X_slice)
        e_mean = offset + e_resid
        ax = axes[row_idx, 2]
        ax.fill_between(
            x, offset + e_lo, offset + e_hi,
            alpha=0.20, facecolor="0.85", label="max/min"
        )
        ax.fill_between(
            x, offset + e_qlo, offset + e_qhi,
            alpha=0.45, facecolor="C1", label=r"$95.45\%$"
        )
        ax.plot(x, e_mean, "C1-", lw=2)

        # POPS Ellipse + PAC.
        p_resid, p_hi, p_lo = pac.predict(X_slice, return_bounds=True)
        p_qlo, p_qhi = pac_percentile_interval(pac, X_slice)
        p_mean = offset + p_resid
        ax = axes[row_idx, 3]
        ax.fill_between(
            x, offset + p_lo, offset + p_hi,
            alpha=0.20, facecolor="0.85", label="max/min"
        )
        ax.fill_between(
            x, offset + p_qlo, offset + p_qhi,
            alpha=0.45, facecolor="C1", label=r"$95.45\%$"
        )
        ax.plot(x, p_mean, "C1-", lw=2)

        for col in range(4):
            ax = axes[row_idx, col]
            ax.plot(x, truth, "k-", lw=1.5, label="Truth")
            ax.tick_params(labelsize=8)
            if row_idx == 1:
                ax.set_xlabel("x", fontsize=9)
        axes[row_idx, 0].set_ylabel(f"N = {n_cases}\nu(x,t)", fontsize=9)

    # Two legends only, matching the workshop figures.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    keep = [labels.index("mean"), labels.index(r"$95.45\%$ ($\pm2\sigma$)"),
            labels.index(r"max/min ($\pm4\sigma$)")]
    axes[0, 0].legend(
        [handles[i] for i in keep], [labels[i] for i in keep],
        fontsize=7, loc="lower left"
    )

    handles, labels = axes[0, 2].get_legend_handles_labels()
    keep = [labels.index("Truth"), labels.index(r"$95.45\%$"), labels.index("max/min")]
    axes[0, 2].legend(
        [handles[i] for i in keep], [labels[i] for i in keep],
        fontsize=7, loc="lower left"
    )

    fig.tight_layout(pad=0.7, w_pad=0.4, h_pad=0.3)
    out = f"burgers_pod_sim2science_r{pod_rank}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {out} and {out.replace('.png', '.pdf')}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank",
        type=int,
        default=3,
        choices=(2, 3, 4, 5),
        help="number of retained POD spatial modes",
    )
    parser.add_argument(
        "--basis-cases",
        type=int,
        default=48,
        help="independent offline snapshots used to construct the POD basis",
    )
    args = parser.parse_args()
    run(pod_rank=args.rank, n_basis_cases=args.basis_cases)
