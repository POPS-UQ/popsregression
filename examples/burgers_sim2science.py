"""Small-N POPS/PAC-Bayes emulator for viscous Burgers' equation.

A cheap deterministic PDE simulator is emulated by a deliberately restricted
17-parameter linear basis.  The basis captures the fundamental and leading
second spatial harmonic, but omits the higher harmonics generated as the
Burgers front steepens.  The example is therefore a compact
simulation-to-science demonstration of model-form misspecification: ordinary
parameter uncertainty contracts with data while POPS retains uncertainty
associated with unresolved physics.

The PAC diagnostics mirror the EH workshop example: the script reports the
held-out projected-ball negative log predictive density G_test, the PAC bound,
and a uniform-density reference.  A certificate is labelled non-vacuous when
bound_ < G_uniform.  With the default hyperprior_center='phase1' this is an
empirical-Bayes diagnostic; use 'warm_start' for the caveat-free formal bound.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import gammaln
from scipy.stats import beta
from sklearn.linear_model import BayesianRidge

from popsregression import POPSRegressionEllipse


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


def raw_inputs(nu, amp, time, x):
    """Scaled physical coordinates (nu, amplitude, time, sin x, cos x)."""
    return np.column_stack(
        [
            scale(nu, NU_RANGE),
            scale(amp, AMP_RANGE),
            scale(time, T_RANGE),
            np.sin(x),
            np.cos(x),
        ]
    )


def make_features(z):
    """Restricted P=17 basis with only the leading nonlinear harmonic.

    Burgers evolution transfers power from sin(x) into successively higher
    harmonics as the front steepens.  Including sin(2x)/cos(2x) makes the mean
    visibly more realistic than a fundamental-only surrogate, while omission
    of n>=3 harmonics leaves structured, irreducible model-form error.
    """
    nu, amp, time, sx, cx = z.T
    s2 = 2.0 * sx * cx
    c2 = cx * cx - sx * sx
    return np.column_stack(
        [
            np.ones(len(z)),
            nu,
            amp,
            time,
            sx,
            cx,
            s2,
            c2,
            amp * sx,
            time * sx,
            time * cx,
            nu * sx,
            nu * cx,
            amp * time,
            nu * time,
            amp * s2,
            time * s2,
        ]
    )


def simulate_cases(cases, points_per_case=12):
    """Turn simulator cases into sparse joint (theta,x)->u observations."""
    rows, values, case_ids = [], [], []
    idx = np.linspace(0, N_GRID - 1, points_per_case, dtype=int)
    for j, (nu, amp, time) in enumerate(cases):
        x, u = burgers_solution(nu, amp, time)
        rows.append(
            raw_inputs(
                np.full(idx.size, nu),
                np.full(idx.size, amp),
                np.full(idx.size, time),
                x[idx],
            )
        )
        values.append(u[idx])
        case_ids.extend([j] * idx.size)
    return np.vstack(rows), np.concatenate(values), np.asarray(case_ids)


def draw_cases(rng, n):
    return np.column_stack(
        [
            rng.uniform(*NU_RANGE, n),
            rng.uniform(*AMP_RANGE, n),
            rng.uniform(*T_RANGE, n),
        ]
    )


def epistemic_bayes_std(model, X):
    """BayesianRidge parameter uncertainty only; no residual-noise term."""
    return np.sqrt(np.einsum("ij,jk,ik->i", X, model.sigma_, X))


def coverage(y, lo, hi):
    return np.mean((y >= lo) & (y <= hi))


def central_projected_ball_interval(mean, lo, hi, dim, half_percentile=0.33):
    """Central 50 +/- half_percentile interval inside a ball pushforward.

    A scalar projection of a uniform d-ball has (z+1)/2 distributed as
    Beta((d+1)/2, (d+1)/2) on z in [-1, 1].  ``half_percentile=0.33``
    therefore gives the central 17th--83rd percentile band requested for the
    plot.  For POPS+PAC the same fractional projected-ball interval is drawn
    inside the displayed PAC-expanded min/max envelope as a visualization.
    """
    a = 0.5 * (dim + 1.0)
    q_hi = 2.0 * beta.ppf(0.5 + half_percentile, a, a) - 1.0
    radius = 0.5 * (hi - lo)
    return mean - q_hi * radius, mean + q_hi * radius


def projected_ball_nll(y, mean, lo, hi, dim, delta=1e-3):
    """Mean exact projected-ball NLL for a uniform dim-ball pushforward.

    ``lo`` and ``hi`` are the bare ellipsoid support bounds. Points outside
    support have infinite NLL; this is reported rather than clipped.
    """
    radius = 0.5 * (hi - lo)
    v = np.maximum(radius**2, 0.0) + delta**2
    q = 1.0 - (y - mean) ** 2 / v
    inside = q > 0.0
    if not np.all(inside):
        return np.inf, int(np.sum(~inside))

    log_c = gammaln(0.5 * dim + 1.0) - 0.5 * np.log(np.pi) - gammaln(
        0.5 * (dim + 1.0)
    )
    k = 0.5 * (dim - 1.0)
    nll = 0.5 * np.log(v) - log_c - k * np.log(q)
    return float(np.mean(nll)), 0


def run(seed=SEED, train_case_counts=(6, 10, 16, 24, 40), n_test_cases=80):
    rng = np.random.default_rng(seed)
    all_train = draw_cases(rng, max(train_case_counts))
    test_cases = draw_cases(rng, n_test_cases)

    z_test, y_test, _ = simulate_cases(test_cases, points_per_case=16)
    X_test = make_features(z_test)
    p = X_test.shape[1]
    uniform_nll = float(np.log(np.ptp(y_test)))

    print(f"Burgers emulator: P={p}; deterministic simulator; no observation noise")
    print(f"Uniform-reference NLL on test range: {uniform_nll:.3f} nats")
    print(
        "cases rows rows/P  BRcov4s Ellcov PACcov  PAC+%   G_test   bound   gap  cert"
    )

    records = []
    fitted = {}
    for n_cases in train_case_counts:
        z_train, y_train, _ = simulate_cases(
            all_train[:n_cases], points_per_case=12
        )
        X_train = make_features(z_train)

        bayes = BayesianRidge(fit_intercept=False).fit(X_train, y_train)
        ellipse = POPSRegressionEllipse(random_state=seed).fit(X_train, y_train)
        pac = POPSRegressionEllipse(random_state=seed, pac_bayes=True).fit(
            X_train, y_train
        )

        b_mean = bayes.predict(X_test)
        b_std = epistemic_bayes_std(bayes, X_test)
        b_cov = coverage(y_test, b_mean - 4 * b_std, b_mean + 4 * b_std)

        e_mean, e_hi, e_lo = ellipse.predict(X_test, return_bounds=True)
        e_cov = coverage(y_test, e_lo, e_hi)
        g_test, uncovered = projected_ball_nll(
            y_test, e_mean, e_lo, e_hi, ellipse._ball_dim, delta=ellipse.delta
        )

        _, p_hi, p_lo, p_bstd = pac.predict(
            X_test, return_bounds=True, return_bound_std=True
        )
        p_cov = coverage(y_test, p_lo, p_hi)
        bare_width = (p_hi - p_lo) - 4.0 * p_bstd
        valid = bare_width > 1e-12
        broadening = np.mean(
            (p_hi[valid] - p_lo[valid]) / bare_width[valid] - 1.0
        )

        gap = pac.bound_ - g_test
        certified = np.isfinite(pac.bound_) and pac.bound_ < uniform_nll
        row = dict(
            cases=n_cases,
            n=len(y_train),
            ratio=len(y_train) / p,
            bayes_coverage=b_cov,
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
        records.append(row)
        fitted[n_cases] = (bayes, ellipse, pac)
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
        np.full_like(x, theta[0]),
        np.full_like(x, theta[1]),
        np.full_like(x, theta[2]),
        x,
    )
    X_slice = make_features(z_slice)

    shown_counts = (train_case_counts[0], train_case_counts[-1])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    column_titles = ["BayesianRidge", "POPS ellipse", "POPS + PAC"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title)

    for row_idx, n_cases in enumerate(shown_counts):
        bayes, ellipse, pac = fitted[n_cases]
        b_mean = bayes.predict(X_slice)
        b_std = epistemic_bayes_std(bayes, X_slice)
        e_mean, e_hi, e_lo = ellipse.predict(X_slice, return_bounds=True)
        p_mean, p_hi, p_lo = pac.predict(X_slice, return_bounds=True)

        e_qlo, e_qhi = central_projected_ball_interval(
            e_mean, e_lo, e_hi, ellipse._ball_dim
        )
        p_qlo, p_qhi = central_projected_ball_interval(
            p_mean, p_lo, p_hi, pac._ball_dim
        )

        ax = axes[row_idx, 0]
        ax.plot(x, truth, "k-", lw=2, label="Burgers truth")
        ax.plot(x, b_mean, "C1-", lw=2, label="emulator mean")
        ax.fill_between(
            x,
            b_mean - 4 * b_std,
            b_mean + 4 * b_std,
            alpha=0.20,
            label=r"$\pm4\sigma$ epistemic",
        )
        ax.fill_between(
            x,
            b_mean - b_std,
            b_mean + b_std,
            alpha=0.45,
            label=r"$\pm1\sigma$ epistemic",
        )

        ax = axes[row_idx, 1]
        ax.plot(x, truth, "k-", lw=2, label="Burgers truth")
        ax.plot(x, e_mean, "C1-", lw=2, label="POPS mean")
        ax.fill_between(x, e_lo, e_hi, alpha=0.20, label="100% support")
        ax.fill_between(
            x, e_qlo, e_qhi, alpha=0.45, label="17th--83rd percentile"
        )

        ax = axes[row_idx, 2]
        ax.plot(x, truth, "k-", lw=2, label="Burgers truth")
        ax.plot(x, p_mean, "C1-", lw=2, label="POPS mean")
        ax.fill_between(x, p_lo, p_hi, alpha=0.20, label="100% PAC envelope")
        ax.fill_between(
            x, p_qlo, p_qhi, alpha=0.45, label="17th--83rd percentile"
        )

        axes[row_idx, 0].set_ylabel(f"{n_cases} simulator cases\nu(x,t)")
        for col in range(3):
            axes[row_idx, col].legend(fontsize=8, loc="lower left")
            if row_idx == 1:
                axes[row_idx, col].set_xlabel("x")

    fig.suptitle(
        "Viscous Burgers: structured model-form error survives as data increase\n"
        r"$u_t + u u_x = \nu u_{xx}$; two-harmonic surrogate, P=" + str(p)
    )
    fig.tight_layout()
    fig.savefig("burgers_sim2science.png", dpi=180, bbox_inches="tight")
    return records


if __name__ == "__main__":
    run()
