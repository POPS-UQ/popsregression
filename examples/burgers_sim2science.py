"""Small-N POPS/PAC-Bayes emulator for viscous Burgers' equation.

The deterministic PDE simulator is emulated by a truncated Fourier surrogate.
Choose 2, 3, or 4 retained harmonics to control model-form misspecification.
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
    x = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    dx = x[1] - x[0]
    u0 = amplitude * np.sin(x)

    def rhs(_, u):
        flux = 0.5 * u**2
        flux_x = (np.roll(flux, -1) - np.roll(flux, 1)) / (2.0 * dx)
        u_xx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2
        return -flux_x + nu * u_xx

    sol = solve_ivp(
        rhs, (0.0, t_final), u0, method="RK45", rtol=2e-6,
        atol=2e-8, max_step=0.01
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return x, sol.y[:, -1]


def scale(v, limits):
    lo, hi = limits
    return 2.0 * (np.asarray(v) - lo) / (hi - lo) - 1.0


def raw_inputs(nu, amp, time, x):
    return np.column_stack([
        scale(nu, NU_RANGE), scale(amp, AMP_RANGE), scale(time, T_RANGE), x
    ])


def make_features(z, harmonic_order=2):
    """Truncated Fourier surrogate; harmonic_order may be 2, 3, or 4."""
    if harmonic_order not in (2, 3, 4):
        raise ValueError("harmonic_order must be 2, 3, or 4")
    nu, amp, time, x = z.T
    cols = [np.ones(len(z)), nu, amp, time, amp * time, nu * time]
    for n in range(1, harmonic_order + 1):
        sn, cn = np.sin(n * x), np.cos(n * x)
        cols.extend([sn, cn, amp * sn, time * sn])
        if n == 1:
            cols.extend([time * cn, nu * sn, nu * cn])
    return np.column_stack(cols)


def simulate_cases(cases, points_per_case=12):
    rows, values, case_ids = [], [], []
    idx = np.linspace(0, N_GRID - 1, points_per_case, dtype=int)
    for j, (nu, amp, time) in enumerate(cases):
        x, u = burgers_solution(nu, amp, time)
        rows.append(raw_inputs(
            np.full(idx.size, nu), np.full(idx.size, amp),
            np.full(idx.size, time), x[idx]
        ))
        values.append(u[idx])
        case_ids.extend([j] * idx.size)
    return np.vstack(rows), np.concatenate(values), np.asarray(case_ids)


def draw_cases(rng, n):
    return np.column_stack([
        rng.uniform(*NU_RANGE, n), rng.uniform(*AMP_RANGE, n),
        rng.uniform(*T_RANGE, n)
    ])


def epistemic_bayes_std(model, X):
    return np.sqrt(np.einsum("ij,jk,ik->i", X, model.sigma_, X))


def coverage(y, lo, hi):
    return np.mean((y >= lo) & (y <= hi))


def projected_ball_fraction(dim, half_percentile=0.33):
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
    var_v = 4.0 * np.sum((Z @ model.U_)**2 * sigma_u_proj, axis=1)
    q_bound_std = np.sqrt(mean_var + q*q * var_v / (4.0 * v))
    hw = q * np.sqrt(v) + 2.0 * q_bound_std
    return mean - hw, mean + hw


def projected_ball_nll(y, mean, lo, hi, dim, delta=1e-3):
    radius = 0.5 * (hi - lo)
    v = np.maximum(radius**2, 0.0) + delta**2
    q = 1.0 - (y - mean)**2 / v
    inside = q > 0.0
    if not np.all(inside):
        return np.inf, int(np.sum(~inside))
    log_c = (
        gammaln(0.5*dim + 1.0) - 0.5*np.log(np.pi)
        - gammaln(0.5*(dim + 1.0))
    )
    k = 0.5 * (dim - 1.0)
    return float(np.mean(0.5*np.log(v) - log_c - k*np.log(q))), 0


def run(seed=SEED, train_case_counts=(10, 16, 24, 40, 80), n_test_cases=80,
        harmonic_order=2):
    rng = np.random.default_rng(seed)
    all_train = draw_cases(rng, max(train_case_counts))
    test_cases = draw_cases(rng, n_test_cases)

    z_test, y_test, _ = simulate_cases(test_cases, points_per_case=16)
    X_test = make_features(z_test, harmonic_order)
    p = X_test.shape[1]
    uniform_nll = float(np.log(np.ptp(y_test)))

    print(f"Burgers emulator: harmonics={harmonic_order}, P={p}; deterministic; no noise")
    print(f"Uniform-reference NLL on test range: {uniform_nll:.3f} nats")
    print("cases rows rows/P  BRcov4s Ellcov PACcov  PAC+%   G_test   bound   gap  cert")

    records, fitted = [], {}
    for n_cases in train_case_counts:
        z_train, y_train, _ = simulate_cases(
            all_train[:n_cases], points_per_case=12
        )
        X_train = make_features(z_train, harmonic_order)
        bayes = BayesianRidge(fit_intercept=False).fit(X_train, y_train)
        hypercube = POPSRegression(
            minimum_relative_error=0.0, posterior="hypercube"
        ).fit(X_train, y_train)
        ellipse = POPSRegressionEllipse(random_state=seed).fit(X_train, y_train)
        pac = POPSRegressionEllipse(
            random_state=seed, pac_bayes=True
        ).fit(X_train, y_train)

        b_mean = bayes.predict(X_test)
        b_std = epistemic_bayes_std(bayes, X_test)
        b_cov = coverage(y_test, b_mean - 4*b_std, b_mean + 4*b_std)
        e_mean, e_hi, e_lo = ellipse.predict(X_test, return_bounds=True)
        e_cov = coverage(y_test, e_lo, e_hi)
        g_test, uncovered = projected_ball_nll(
            y_test, e_mean, e_lo, e_hi, ellipse._ball_dim, ellipse.delta
        )
        _, p_hi, p_lo, p_bstd = pac.predict(
            X_test, return_bounds=True, return_bound_std=True
        )
        p_cov = coverage(y_test, p_lo, p_hi)
        bare_width = (p_hi - p_lo) - 4.0*p_bstd
        valid = bare_width > 1e-12
        broadening = np.mean(
            (p_hi[valid] - p_lo[valid])/bare_width[valid] - 1.0
        )
        gap = pac.bound_ - g_test
        certified = np.isfinite(pac.bound_) and pac.bound_ < uniform_nll
        records.append(dict(
            cases=n_cases, n=len(y_train), ratio=len(y_train)/p,
            bayes_coverage=b_cov, ellipse_coverage=e_cov,
            pac_coverage=p_cov, pac_broadening=broadening,
            g_test=g_test, uncovered=uncovered,
            uniform_nll=uniform_nll, bound=pac.bound_,
            bound_gap=gap, nonvacuous=certified,
            objective=pac.objective_, kl=pac.kl_
        ))
        fitted[n_cases] = (bayes, hypercube, ellipse, pac)
        g_txt = " inf" if not np.isfinite(g_test) else f"{g_test:7.3f}"
        print(
            f"{n_cases:5d} {len(y_train):4d} {len(y_train)/p:6.2f}"
            f"   {b_cov:7.3f} {e_cov:6.3f} {p_cov:6.3f}"
            f" {100*broadening:6.1f}% {g_txt} {pac.bound_:7.3f}"
            f" {gap:6.3f}   {'YES' if certified else 'no'}"
        )
        if uncovered:
            print(f"      note: {uncovered}/{len(y_test)} test points outside bare support")

    theta = (0.014, 1.15, 0.78)
    x, truth = burgers_solution(*theta)
    z_slice = raw_inputs(
        np.full_like(x, theta[0]), np.full_like(x, theta[1]),
        np.full_like(x, theta[2]), x
    )
    X_slice = make_features(z_slice, harmonic_order)

    shown_counts = (train_case_counts[0], train_case_counts[-1])
    fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
    titles = [
        "Bayesian Ridge", "POPS Hypercube", "POPS Ellipse",
        "POPS Ellipse + PAC"
    ]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=10)

    for row_idx, n_cases in enumerate(shown_counts):
        bayes, hypercube, ellipse, pac = fitted[n_cases]

        b_mean = bayes.predict(X_slice)
        b_std = epistemic_bayes_std(bayes, X_slice)
        h_mean, h_std, h_hi, h_lo = hypercube.predict(
            X_slice, return_std=True, return_bounds=True
        )
        e_mean, e_hi, e_lo = ellipse.predict(X_slice, return_bounds=True)
        e_qlo, e_qhi = bare_percentile_interval(ellipse, X_slice)
        p_mean, p_hi, p_lo = pac.predict(X_slice, return_bounds=True)
        p_qlo, p_qhi = pac_percentile_interval(pac, X_slice)

        # Bayesian Ridge
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

        # POPS Hypercube
        ax = axes[row_idx, 1]
        ax.fill_between(x, h_lo, h_hi, alpha=0.20, facecolor="0.85",
                        label="max/min")
        ax.fill_between(
            x, h_mean - 2*h_std, h_mean + 2*h_std,
            alpha=0.45, facecolor="C1", label=r"$95.45\%$"
        )
        ax.plot(x, h_mean, "C1-", lw=2)

        # POPS Ellipse
        ax = axes[row_idx, 2]
        ax.fill_between(x, e_lo, e_hi, alpha=0.20, facecolor="0.85",
                        label="max/min")
        ax.fill_between(x, e_qlo, e_qhi, alpha=0.45, facecolor="C1",
                        label=r"$95.45\%$")
        ax.plot(x, e_mean, "C1-", lw=2)

        # POPS Ellipse + PAC
        ax = axes[row_idx, 3]
        ax.fill_between(x, p_lo, p_hi, alpha=0.20, facecolor="0.85",
                        label="max/min")
        ax.fill_between(x, p_qlo, p_qhi, alpha=0.45, facecolor="C1",
                        label=r"$95.45\%$")
        ax.plot(x, p_mean, "C1-", lw=2)

        for col in range(4):
            ax = axes[row_idx, col]
            truth_label = "Truth" if col == 2 else "_nolegend_"
            ax.plot(x, truth, "k-", lw=1.5, label=truth_label)
            ax.tick_params(labelsize=8)
            if row_idx == 1:
                ax.set_xlabel("x", fontsize=9)

        axes[row_idx, 0].set_ylabel(f"N = {n_cases}\nu(x,t)", fontsize=9)

    # Two legends only: one explains Bayesian notation, one explains POPS.
    # Use the top row so the bottom row remains visually uncluttered.
    br_handles, br_labels = axes[0, 0].get_legend_handles_labels()
    # Reorder to mean, 95.45%, max/min.
    br_order = [2, 1, 0]
    axes[0, 0].legend(
        [br_handles[i] for i in br_order],
        [br_labels[i] for i in br_order],
        fontsize=7, loc="lower left"
    )

    pops_handles, pops_labels = axes[0, 2].get_legend_handles_labels()
    # Reorder to Truth, 95.45%, max/min.
    label_to_handle = dict(zip(pops_labels, pops_handles))
    pops_order = ["Truth", r"$95.45\%$", "max/min"]
    axes[0, 2].legend(
        [label_to_handle[label] for label in pops_order],
        pops_order,
        fontsize=7, loc="lower left"
    )

    fig.tight_layout()
    out = f"burgers_sim2science_h{harmonic_order}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {out} and {out.replace('.png', '.pdf')}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--harmonics", type=int, choices=(2, 3, 4), default=2,
        help="number of Fourier harmonics retained in the surrogate"
    )
    args = parser.parse_args()
    run(harmonic_order=args.harmonics)
