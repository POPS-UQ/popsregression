"""Small-N POPS/PAC-Bayes emulator for viscous Burgers' equation.

The simulator is intentionally cheap and deterministic.  The surrogate is
intentionally misspecified: a low-order polynomial cannot reproduce the
steepening front at low viscosity.  This makes the example useful for showing
that ordinary parameter uncertainty contracts with N while model-form
uncertainty remains, and that the PAC hyperposterior matters most at small N.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

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
        # Conservative centered flux plus centered diffusion.  The viscosity
        # range keeps this stable/accurate for the workshop-sized grid.
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
    """Scaled emulator coordinates (nu, amplitude, time, periodic x)."""
    return np.column_stack(
        [
            scale(nu, NU_RANGE),
            scale(amp, AMP_RANGE),
            scale(time, T_RANGE),
            np.sin(x),
            np.cos(x),
        ]
    )


def simulate_cases(cases, points_per_case=12):
    """Turn expensive simulator cases into joint (theta,x)->u observations."""
    rows, values, case_ids = [], [], []
    # Fixed spatial subsampling: each simulator call supplies a sparse field.
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


def make_features(z, poly=None):
    # Degree 2 gives only 21 coefficients for five raw coordinates.  It is
    # deliberately too smooth to represent a sharp Burgers front.
    if poly is None:
        poly = PolynomialFeatures(degree=2, include_bias=True)
        return poly.fit_transform(z), poly
    return poly.transform(z), poly


def epistemic_bayes_std(model, X):
    """BayesianRidge parameter uncertainty only; no residual-noise term."""
    return np.sqrt(np.einsum("ij,jk,ik->i", X, model.sigma_, X))


def coverage(y, lo, hi):
    return np.mean((y >= lo) & (y <= hi))


def run(seed=SEED, train_case_counts=(3, 6, 12, 24), n_test_cases=80):
    rng = np.random.default_rng(seed)
    all_train = draw_cases(rng, max(train_case_counts))
    test_cases = draw_cases(rng, n_test_cases)

    z_test, y_test, _ = simulate_cases(test_cases, points_per_case=16)
    X_test, poly = make_features(z_test)
    p = X_test.shape[1]

    print(f"Burgers emulator: P={p}; deterministic simulator; no observation noise")
    print("cases  rows  rows/P   BR cov(4s)  ellipse cov  PAC cov  PAC broadening  bound")

    records = []
    fitted = {}
    for n_cases in train_case_counts:
        z_train, y_train, _ = simulate_cases(
            all_train[:n_cases], points_per_case=12
        )
        X_train, _ = make_features(z_train, poly)

        bayes = BayesianRidge(fit_intercept=False).fit(X_train, y_train)
        ellipse = POPSRegressionEllipse(random_state=seed).fit(X_train, y_train)
        pac = POPSRegressionEllipse(random_state=seed, pac_bayes=True).fit(
            X_train, y_train
        )

        b_mean = bayes.predict(X_test)
        b_std = epistemic_bayes_std(bayes, X_test)
        b_cov = coverage(y_test, b_mean - 4 * b_std, b_mean + 4 * b_std)

        _, e_hi, e_lo = ellipse.predict(X_test, return_bounds=True)
        e_cov = coverage(y_test, e_lo, e_hi)

        _, p_hi, p_lo, p_bstd = pac.predict(
            X_test, return_bounds=True, return_bound_std=True
        )
        p_cov = coverage(y_test, p_lo, p_hi)
        bare_width = (p_hi - p_lo) - 4.0 * p_bstd
        valid = bare_width > 1e-12
        broadening = np.mean(
            (p_hi[valid] - p_lo[valid]) / bare_width[valid] - 1.0
        )

        row = dict(
            cases=n_cases,
            n=len(y_train),
            ratio=len(y_train) / p,
            bayes_coverage=b_cov,
            ellipse_coverage=e_cov,
            pac_coverage=p_cov,
            pac_broadening=broadening,
            bound=pac.bound_,
            objective=pac.objective_,
            kl=pac.kl_,
        )
        records.append(row)
        fitted[n_cases] = (bayes, ellipse, pac, poly)
        print(
            f"{n_cases:5d} {len(y_train):5d} {len(y_train)/p:7.2f}"
            f"      {b_cov:7.3f}      {e_cov:7.3f}  {p_cov:7.3f}"
            f"       {100*broadening:7.1f}%   {pac.bound_:8.3f}"
        )

    # A hard, low-viscosity held-out slice makes the missing front physics
    # visible.  Plot the smallest and largest training sets.
    theta = (0.014, 1.15, 0.78)
    x, truth = burgers_solution(*theta)
    z_slice = raw_inputs(
        np.full_like(x, theta[0]),
        np.full_like(x, theta[1]),
        np.full_like(x, theta[2]),
        x,
    )
    X_slice, _ = make_features(z_slice, poly)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
    for col, n_cases in enumerate((train_case_counts[0], train_case_counts[-1])):
        bayes, ellipse, pac, _ = fitted[n_cases]
        b_mean = bayes.predict(X_slice)
        b_std = epistemic_bayes_std(bayes, X_slice)
        p_mean, p_hi, p_lo = pac.predict(X_slice, return_bounds=True)

        ax = axes[0, col]
        ax.plot(x, truth, "k-", lw=2, label="Burgers truth")
        ax.plot(x, b_mean, "C1-", lw=2, label="quadratic emulator")
        ax.fill_between(x, b_mean - 4*b_std, b_mean + 4*b_std, alpha=0.25,
                        label=r"BayesianRidge $\pm4\sigma$")
        ax.set_title(f"{n_cases} simulator cases")
        ax.set_ylabel("u(x,t)")
        ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.plot(x, truth, "k-", lw=2, label="Burgers truth")
        ax.plot(x, p_mean, "C1-", lw=2, label="POPS mean")
        ax.fill_between(x, p_lo, p_hi, alpha=0.3, label="POPS ellipse + PAC")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Viscous Burgers: structured model-form error survives as data increase\n"
        r"$u_t + u u_x = \nu u_{xx}$; quadratic surrogate, P=" + str(p)
    )
    fig.tight_layout()
    fig.savefig("burgers_sim2science.png", dpi=180, bbox_inches="tight")
    return records


if __name__ == "__main__":
    run()
