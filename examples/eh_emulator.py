"""
==========================================================================
Certified emulator uncertainty for the Eisenstein-Hu linear power spectrum
==========================================================================

Second numerical demonstration for the Sim2Science paper: a polynomial
surrogate for the sigma8-normalized Eisenstein & Hu (1998) linear matter
power spectrum, y = ln P(k*) at k* = 0.15 h/Mpc.  The scale k* sits inside
the baryon-acoustic-oscillation envelope, so the target oscillates along
the sound-horizon direction of parameter space; a low-degree polynomial in
the 5 cosmological parameters cuts through the BAO wiggle, providing the
structural (noise-free) misspecification that
:class:`~popsregression.POPSRegressionEllipse` is built to certify.

The script is deterministic end-to-end from a single master seed and
produces

* ``eh_emulator.png`` / ``eh_emulator.pdf`` - the 3-panel paper figure
  (full page width, ~0.35 page height);
* ``eh_emulator_summary.md`` - the summary statistics quoted in the paper
  text (degree scan, coverage, band widths, PAC broadening, certificate
  gap, timing) plus the acceptance-check report.

The EH98 transfer function (Eisenstein & Hu 1998, ApJ 496, 605,
eqs. 2-24) is implemented directly - no CAMB/CLASS dependency - and is
validated before the data pool is generated: the closed-form sound
horizon (eq. 6) is checked against direct quadrature of
``s = int c_s dz / H`` to better than 1e-3, ``k_eq`` against its exact
square-root form, and the wiggle/no-wiggle ratio must oscillate about 1
with percent-level amplitude in the BAO range.

Run ``python eh_emulator.py`` for the full experiment (about 10-20 min)
or ``python eh_emulator.py --quick`` for a reduced smoke test.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

from popsregression import POPSRegression, POPSRegressionEllipse
from popsregression._projected_ball import projected_ball_logpdf

from eh_emulator_figure import make_figure

# --------------------------------------------------------------------------
# Protocol constants (one global calibration, frozen; nothing is tuned
# per N or per replicate)
# --------------------------------------------------------------------------

C_KMS = 299792.458  # speed of light [km/s]
T_CMB = 2.7255  # CMB temperature [K]

# Uniform parameter box: (omega_c, omega_b, h, n_s, sigma8)
BOX_LOW = np.array([0.08, 0.019, 0.60, 0.92, 0.70])
BOX_HIGH = np.array([0.16, 0.026, 0.80, 1.00, 0.95])
PARAM_NAMES = (r"\omega_c", r"\omega_b", "h", "n_s", r"\sigma_8")

K_STAR = 0.15  # QoI scale [h/Mpc], inside the BAO envelope
K_REF = 0.05  # reference scale of the secondary (appendix) QoI [h/Mpc]

# Fixed sigma8 normalization grid, identical for every sample (re-gridding
# per sample would inject pseudo-noise into the noise-free engine).
K_GRID = np.logspace(-4.0, 2.0, 2048)  # [h/Mpc]
LN_K_GRID = np.log(K_GRID)

M_POOL = 40_000
N_TEST = 8_000
N_VAL = 4_096
N_GRID = (256, 1024, 4096, 16384)
N_REPS = 10
SCAN_DEGREES = (1, 2, 3, 4)
SCAN_N_TRAIN = 16_384
RMSE_WINDOW = (0.04, 0.12)  # acceptance window for the degree scan
MAX_P = 150  # keeps the support/std ratio sqrt(P+2) <= ~12

RANK = 32
MAX_ITER = 5_000
N_RHO_STAGES = 4  # len of the default rho_schedule
HC_MIN_SAMPLES = 100_000  # hypercube max/min sample count (sampled baseline)


# --------------------------------------------------------------------------
# EH98 engine (Eisenstein & Hu 1998, ApJ 496, 605).  Wavenumbers in Mpc^-1
# and lengths in Mpc inside this block, following the published fit.
# --------------------------------------------------------------------------


def eh98_parameters(omega_m, omega_b, tcmb=T_CMB):
    """Derived EH98 quantities (eqs. 2-7, 11-15, 23-24)."""
    theta = tcmb / 2.7
    f_b = omega_b / omega_m
    f_c = 1.0 - f_b

    z_eq = 2.50e4 * omega_m / theta**4  # eq. (2); really 1 + z_eq
    k_eq = 7.46e-2 * omega_m / theta**2  # eq. (3) [Mpc^-1]

    # drag epoch, eq. (4)
    b1 = 0.313 * omega_m**-0.419 * (1.0 + 0.607 * omega_m**0.674)
    b2 = 0.238 * omega_m**0.223
    z_d = (
        1291.0
        * omega_m**0.251
        / (1.0 + 0.659 * omega_m**0.828)
        * (1.0 + b1 * omega_b**b2)
    )

    # baryon-to-photon momentum density, eq. (5)
    R_d = 31.5 * omega_b / theta**4 * (1.0e3 / (1.0 + z_d))
    R_eq = 31.5 * omega_b / theta**4 * (1.0e3 / z_eq)

    # sound horizon at the drag epoch, eq. (6) [Mpc]
    s = (
        2.0
        / (3.0 * k_eq)
        * np.sqrt(6.0 / R_eq)
        * np.log((np.sqrt(1.0 + R_d) + np.sqrt(R_d + R_eq)) / (1.0 + np.sqrt(R_eq)))
    )

    # Silk damping scale, eq. (7) [Mpc^-1]
    k_silk = 1.6 * omega_b**0.52 * omega_m**0.73 * (1.0 + (10.4 * omega_m) ** -0.95)

    # CDM suppression alpha_c, eq. (11)
    a1 = (46.9 * omega_m) ** 0.670 * (1.0 + (32.1 * omega_m) ** -0.532)
    a2 = (12.0 * omega_m) ** 0.424 * (1.0 + (45.0 * omega_m) ** -0.582)
    alpha_c = a1**-f_b * a2 ** -(f_b**3)

    # CDM shift beta_c, eq. (12)
    bb1 = 0.944 / (1.0 + (458.0 * omega_m) ** -0.708)
    bb2 = (0.395 * omega_m) ** -0.0266
    beta_c = 1.0 / (1.0 + bb1 * (f_c**bb2 - 1.0))

    # baryon amplitude alpha_b, eqs. (14)-(15)
    y = z_eq / (1.0 + z_d)
    Gy = y * (
        -6.0 * np.sqrt(1.0 + y)
        + (2.0 + 3.0 * y)
        * np.log((np.sqrt(1.0 + y) + 1.0) / (np.sqrt(1.0 + y) - 1.0))
    )
    alpha_b = 2.07 * k_eq * s * (1.0 + R_d) ** -0.75 * Gy

    # baryon envelope shift, eq. (24), and node shift, eq. (23)
    beta_b = 0.5 + f_b + (3.0 - 2.0 * f_b) * np.sqrt((17.2 * omega_m) ** 2 + 1.0)
    beta_node = 8.41 * omega_m**0.435

    return dict(
        theta=theta, f_b=f_b, f_c=f_c, z_eq=z_eq, k_eq=k_eq, z_d=z_d,
        R_d=R_d, R_eq=R_eq, s=s, k_silk=k_silk, alpha_c=alpha_c,
        beta_c=beta_c, alpha_b=alpha_b, beta_b=beta_b, beta_node=beta_node,
    )


def _T0_tilde(q, alpha_c, beta_c):
    """Pressure-free growth form T~_0, eqs. (18)-(20)."""
    C = 14.2 / alpha_c + 386.0 / (1.0 + 69.9 * q**1.08)
    L = np.log(np.e + 1.8 * beta_c * q)
    return L / (L + C * q * q)


def eh98_transfer(k_mpc, pars):
    """Full EH98 transfer function with BAO, eqs. (16)-(24); k in Mpc^-1."""
    k = np.asarray(k_mpc, dtype=float)
    q = k / (13.41 * pars["k_eq"])  # eq. (10)
    ks = k * pars["s"]

    # CDM piece, eqs. (17)-(18)
    f = 1.0 / (1.0 + (ks / 5.4) ** 4)
    T_c = f * _T0_tilde(q, 1.0, pars["beta_c"]) + (1.0 - f) * _T0_tilde(
        q, pars["alpha_c"], pars["beta_c"]
    )

    # baryon piece with the j_0 wiggle term, eqs. (21)-(22)
    s_tilde = pars["s"] / (1.0 + (pars["beta_node"] / ks) ** 3) ** (1.0 / 3.0)
    j0 = np.sinc(k * s_tilde / np.pi)  # j_0(x) = sin(x)/x
    T_b = (
        _T0_tilde(q, 1.0, 1.0) / (1.0 + (ks / 5.2) ** 2)
        + pars["alpha_b"]
        / (1.0 + (pars["beta_b"] / ks) ** 3)
        * np.exp(-((k / pars["k_silk"]) ** 1.4))
    ) * j0

    return pars["f_b"] * T_b + pars["f_c"] * T_c  # eq. (16)


def eh98_sound_horizon_fit(omega_m, omega_b):
    """Approximate sound horizon, eq. (26) [Mpc] (published fitting form)."""
    return 44.5 * np.log(9.83 / omega_m) / np.sqrt(1.0 + 10.0 * omega_b**0.75)


def eh98_transfer_nowiggle(k_mpc, omega_m, omega_b, tcmb=T_CMB):
    """No-wiggle EH98 fitting form, eqs. (26), (28)-(31); k in Mpc^-1."""
    k = np.asarray(k_mpc, dtype=float)
    theta = tcmb / 2.7
    f_b = omega_b / omega_m
    s = eh98_sound_horizon_fit(omega_m, omega_b)
    alpha_g = (
        1.0
        - 0.328 * np.log(431.0 * omega_m) * f_b
        + 0.38 * np.log(22.3 * omega_m) * f_b**2
    )  # eq. (31)
    gamma_eff = omega_m * (alpha_g + (1.0 - alpha_g) / (1.0 + (0.43 * k * s) ** 4))
    q_eff = k * theta**2 / gamma_eff  # eq. (28), k in Mpc^-1
    L0 = np.log(2.0 * np.e + 1.8 * q_eff)  # eq. (29)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q_eff)  # eq. (30)
    return L0 / (L0 + C0 * q_eff * q_eff)


def _sound_horizon_quadrature(omega_m, omega_b, tcmb=T_CMB):
    """Direct quadrature of ``s = int_0^{a_d} c_s da / (a^2 H)``.

    Matter + radiation early universe with 1 + z_eq and R(a) exactly as in
    the EH98 parameterization, so this equals the closed form (eq. 6) up
    to quadrature error and the 7.46e-2 rounding of eq. (3): an
    independent typo check on the sound-horizon algebra.
    """
    theta = tcmb / 2.7
    z_eq = 2.50e4 * omega_m / theta**4
    b1 = 0.313 * omega_m**-0.419 * (1.0 + 0.607 * omega_m**0.674)
    b2 = 0.238 * omega_m**0.223
    z_d = (
        1291.0
        * omega_m**0.251
        / (1.0 + 0.659 * omega_m**0.828)
        * (1.0 + b1 * omega_b**b2)
    )
    a_d = 1.0 / (1.0 + z_d)
    a_eq = 1.0 / z_eq
    r_c = 31.5e3 * omega_b / theta**4  # R(a) = r_c * a

    def integrand(a):
        cs = C_KMS / np.sqrt(3.0 * (1.0 + r_c * a))
        H = 100.0 * np.sqrt(omega_m * a**-3 * (1.0 + a_eq / a))  # [km/s/Mpc]
        return cs / (a * a * H)

    return quad(integrand, 0.0, a_d, limit=200)[0]


def validate_eh98(checks):
    """Run the EH98 acceptance checks; return a report dict."""
    report = {}
    # sCDM-like fiducial of the EH98 figures plus the box center
    fiducials = {
        "sCDM (Omega_0=1, Omega_b=0.05, h=0.5)": (0.25, 0.0125, 0.5),
        "box center": (
            0.5 * (BOX_LOW[0] + BOX_HIGH[0]) + 0.5 * (BOX_LOW[1] + BOX_HIGH[1]),
            0.5 * (BOX_LOW[1] + BOX_HIGH[1]),
            0.5 * (BOX_LOW[2] + BOX_HIGH[2]),
        ),
    }
    for name, (om, ob, h) in fiducials.items():
        pars = eh98_parameters(om, ob)
        s_quad = _sound_horizon_quadrature(om, ob)
        s_fit = eh98_sound_horizon_fit(om, ob)
        k_eq_sqrt = 100.0 / C_KMS * np.sqrt(2.0 * om * pars["z_eq"])
        rel_s = abs(pars["s"] / s_quad - 1.0)
        rel_k = abs(pars["k_eq"] / k_eq_sqrt - 1.0)
        rel_fit = abs(s_fit / pars["s"] - 1.0)
        t_low = float(eh98_transfer(1e-6, pars))
        k_bao = np.logspace(np.log10(0.05), np.log10(0.5), 400) * h  # Mpc^-1
        ratio = eh98_transfer(k_bao, pars) / eh98_transfer_nowiggle(k_bao, om, ob)
        n_cross = int(np.sum(np.diff(np.sign(ratio - 1.0)) != 0))
        amp = float(np.max(np.abs(ratio - 1.0)))
        report[name] = dict(
            omega_m=om, omega_b=ob, h=h, s=pars["s"], s_quad=s_quad,
            s_fit=s_fit, rel_s=rel_s, rel_k=rel_k, rel_fit=rel_fit,
            z_eq=pars["z_eq"], z_d=pars["z_d"], k_eq=pars["k_eq"],
            t_low=t_low, wiggle_crossings=n_cross, wiggle_amplitude=amp,
        )
        checks.record(
            f"EH98 sound horizon vs quadrature ({name})",
            rel_s < 1e-3,
            f"rel = {rel_s:.2e} (eq. 6 = {pars['s']:.3f} Mpc)",
        )
        checks.record(
            f"EH98 k_eq vs exact sqrt form ({name})",
            rel_k < 1e-3,
            f"rel = {rel_k:.2e}",
        )
        checks.record(
            f"EH98 sound horizon vs published eq. 26 fit ({name})",
            rel_fit < 2.5e-2,
            f"rel = {rel_fit:.2e} (published accuracy ~2%)",
        )
        checks.record(
            f"EH98 T(k->0) -> 1 ({name})", abs(t_low - 1.0) < 1e-3,
            f"T(1e-6/Mpc) = {t_low:.6f}",
        )
        checks.record(
            f"EH98 wiggle/no-wiggle oscillates about 1 ({name})",
            n_cross >= 4 and 5e-3 < amp < 0.10,
            f"{n_cross} sign changes, max |ratio-1| = {100 * amp:.2f}%",
        )
    return report


# --------------------------------------------------------------------------
# Quantities of interest
# --------------------------------------------------------------------------


def _tophat_window(x):
    """Fourier top-hat W(x) = 3 (sin x - x cos x) / x^3, stable at x -> 0."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, x)
    w = 3.0 * (np.sin(xs) - xs * np.cos(xs)) / xs**3
    return np.where(small, 1.0 - x * x / 10.0, w)


def ln_power_qoi(theta):
    """Return ``(ln P(k*), ln[P(k*)/P(k_ref)])`` for one parameter vector.

    ``theta = (omega_c, omega_b, h, n_s, sigma8)``; flat universe,
    z = 0, k in h/Mpc and P in (Mpc/h)^3, normalized to sigma8 with the
    fixed-grid trapezoid top-hat integral.
    """
    omega_c, omega_b, h, n_s, sigma8 = theta
    pars = eh98_parameters(omega_c + omega_b, omega_b)
    T_grid = eh98_transfer(K_GRID * h, pars)
    P_unit = K_GRID**n_s * T_grid**2
    integrand = (
        K_GRID**3 * P_unit / (2.0 * np.pi**2) * _tophat_window(8.0 * K_GRID) ** 2
    )
    sigma8_sq_unit = np.trapezoid(integrand, LN_K_GRID)
    ln_amp = 2.0 * np.log(sigma8) - np.log(sigma8_sq_unit)
    T_pts = eh98_transfer(np.array([K_STAR, K_REF]) * h, pars)
    ln_p_star = ln_amp + n_s * np.log(K_STAR) + 2.0 * np.log(T_pts[0])
    ln_p_ref = ln_amp + n_s * np.log(K_REF) + 2.0 * np.log(T_pts[1])
    return ln_p_star, ln_p_star - ln_p_ref


def scale_to_box(theta):
    """Min-max scale parameters to [-1, 1] using the box (not the sample)."""
    return 2.0 * (theta - BOX_LOW) / (BOX_HIGH - BOX_LOW) - 1.0


# --------------------------------------------------------------------------
# Acceptance-check accumulator
# --------------------------------------------------------------------------


class Checks:
    """Collect acceptance checks; report all, suppress none."""

    def __init__(self):
        self.rows = []

    def record(self, name, passed, detail=""):
        self.rows.append((name, bool(passed), detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")

    @property
    def all_passed(self):
        return all(passed for _, passed, _ in self.rows)


def _seed_int(master_seed, *key):
    """Deterministic 31-bit integer seed from the master seed and a key."""
    state = np.random.SeedSequence([int(master_seed), *map(int, key)])
    return int(state.generate_state(1)[0] % (2**31))


# --------------------------------------------------------------------------
# Model fits for one (N, replicate) cell
# --------------------------------------------------------------------------


def hypercube_minmax(pops, X_aug, block=5_000):
    """Sampled hypercube max/min bounds, blocked over posterior samples."""
    samples = pops.posterior_samples_
    y_max = np.full(X_aug.shape[0], -np.inf)
    y_min = np.full(X_aug.shape[0], np.inf)
    for j in range(0, samples.shape[1], block):
        blk = X_aug @ samples[:, j : j + block]
        y_max = np.maximum(y_max, blk.max(axis=1))
        y_min = np.minimum(y_min, blk.min(axis=1))
    return y_max, y_min


def fit_pops_hypercube(F_tr, y_tr, seed, n_min_samples):
    """POPS hypercube fit with at least ``n_min_samples`` posterior draws."""
    density = max(1.0, 1.5 * n_min_samples / F_tr.shape[0])
    for _ in range(4):
        np.random.seed(seed)  # the 'uniform' resampler uses global state
        pops = POPSRegression(fit_intercept=True, resample_density=density)
        pops.fit(F_tr, y_tr)
        if pops.posterior_samples_.shape[1] >= n_min_samples:
            return pops
        density *= 2.0
    return pops


def fit_cell(F_tr, y_tr, F_te, y_te, P, seed_ell, seed_pops, checks=None,
             tag="", n_hc_samples=HC_MIN_SAMPLES, return_models=False):
    """Fit the four methods on one training subset; return the metric row.

    The feature matrices must already be standardized (per-column, on this
    training split); the same matrices are used by all four methods.
    """
    delta = 1e-3 * float(y_tr.std())
    row = dict(delta=delta)

    # -- BayesianRidge, +-4 sigma epistemic band -------------------------
    br = BayesianRidge(fit_intercept=True)
    br.fit(F_tr, y_tr)
    br_mean = br.predict(F_te)
    br_std = np.sqrt(np.sum((F_te @ br.sigma_) * F_te, axis=1))
    row["cov_br"] = float(np.mean(np.abs(y_te - br_mean) <= 4.0 * br_std))

    # -- POPS hypercube, sampled max/min ---------------------------------
    pops = fit_pops_hypercube(F_tr, y_tr, seed_pops, n_hc_samples)
    row["n_hc_samples"] = int(pops.posterior_samples_.shape[1])
    F_te_aug = np.hstack([F_te, np.ones((F_te.shape[0], 1))])
    hc_mean = pops.predict(F_te)
    dev_max, dev_min = hypercube_minmax(pops, F_te_aug)
    hc_max, hc_min = hc_mean + dev_max, hc_mean + dev_min
    row["cov_hc"] = float(np.mean((y_te >= hc_min) & (y_te <= hc_max)))
    width_hc = hc_max - hc_min
    row["width_hc"] = float(width_hc.mean())

    # -- POPSRegressionEllipse, bare -------------------------------------
    ell = POPSRegressionEllipse(
        rank=RANK, max_iter=MAX_ITER, fit_intercept=True,
        random_state=seed_ell, delta=delta,
    )
    ell.fit(F_tr, y_tr)
    e_mean, e_max, e_min = ell.predict(F_te, return_bounds=True)
    row["cov_e"] = float(np.mean((y_te >= e_min) & (y_te <= e_max)))
    row["width_e"] = float((e_max - e_min).mean())
    row["ratio_e_hc"] = float(((e_max - e_min) / width_hc).mean())
    row["objective"] = float(ell.objective_)
    row["covfrac_bare"] = float(ell.coverage_fraction_)
    row["n_iter_bare"] = int(ell.n_iter_)
    row["rank_"] = int(ell.rank_)

    # test-set empirical generalization error of the fitted pushforward:
    # recover sqrt(v) from the bare-bound spread, n_dim = P + 1
    half = 0.5 * (e_max - e_min)
    logp = projected_ball_logpdf(y_te - e_mean, half, P + 1)
    inside = np.isfinite(logp)
    row["G_test"] = float(-logp[inside].mean())
    row["n_uncovered_test"] = int((~inside).sum())

    # -- POPSRegressionEllipse + PAC-Bayes layer -------------------------
    pac = POPSRegressionEllipse(
        rank=RANK, max_iter=MAX_ITER, fit_intercept=True,
        random_state=seed_ell, delta=delta, pac_bayes=True,
    )
    pac.fit(F_tr, y_tr)
    p_mean, p_max, p_min, p_bstd = pac.predict(
        F_te, return_bounds=True, return_bound_std=True
    )
    row["cov_pac"] = float(np.mean((y_te >= p_min) & (y_te <= p_max)))
    row["width_pac"] = float((p_max - p_min).mean())
    row["ratio_pac_hc"] = float(((p_max - p_min) / width_hc).mean())
    row["broaden_pct"] = float(
        100.0 * np.mean((0.5 * (p_max - p_min) - half) / half)
    )
    row["bound"] = float(pac.bound_)
    row["kl"] = float(pac.kl_)
    row["gamma"] = float(pac.gamma_)
    row["tau2"] = float(pac.tau2_)
    row["covfrac_pac"] = float(pac.coverage_fraction_)
    row["n_iter_pac"] = int(pac.n_iter_)
    row["pac_contains_bare"] = bool(np.all(p_max > e_max) & np.all(p_min < e_min))
    row["pac_map_matches_bare"] = bool(
        np.allclose(pac.coef_, ell.coef_) and np.allclose(pac.U_, ell.U_)
    )

    if checks is not None:
        checks.record(
            f"fit converged & covering ({tag})",
            row["covfrac_bare"] == 1.0
            and row["covfrac_pac"] == 1.0
            and row["n_iter_bare"] < N_RHO_STAGES * MAX_ITER
            and row["n_iter_pac"] < N_RHO_STAGES * MAX_ITER,
            f"coverage_fraction_ = ({row['covfrac_bare']:.4f}, "
            f"{row['covfrac_pac']:.4f}), n_iter_ = ({row['n_iter_bare']}, "
            f"{row['n_iter_pac']}) of cap {N_RHO_STAGES * MAX_ITER}",
        )
    if return_models:
        return row, (br, pops, ell, pac)
    return row


def _agg(rows, key):
    """(mean, std, min, max) of one metric over replicate rows."""
    v = np.array([r[key] for r in rows], dtype=float)
    return v.mean(), v.std(), v.min(), v.max()



# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--seed", type=int, default=0, help="master seed")
    ap.add_argument("--quick", action="store_true",
                    help="reduced smoke test (small pool, 2 N values)")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent,
                    help="output directory for figure and summary")
    ap.add_argument("--skip-p2000", action="store_true",
                    help="skip the large P~2000 timing attempt")
    args = ap.parse_args(argv)

    master = args.seed
    if args.quick:
        m_pool, n_test, n_val = 12_000, 3_000, 2_048
        n_grid, n_reps = (256, 1024), 3
        scan_n_train, hc_min = 6_000, 20_000
        do_timing = False
    else:
        m_pool, n_test, n_val = M_POOL, N_TEST, N_VAL
        n_grid, n_reps = N_GRID, N_REPS
        scan_n_train, hc_min = SCAN_N_TRAIN, HC_MIN_SAMPLES
        do_timing = True
    n_train_region = m_pool - n_test - n_val

    checks = Checks()
    t_start = time.perf_counter()

    # ---- 1. engine validation (before generating the pool) -------------
    print("== EH98 engine validation ==")
    eh_report = validate_eh98(checks)
    if not checks.all_passed:
        print("EH98 validation failed; stopping before pool generation.")
        return 1

    # ---- 2. pool --------------------------------------------------------
    print(f"== evaluating pool (M = {m_pool}) ==", flush=True)
    rng_pool = np.random.default_rng(np.random.SeedSequence([master, 1]))
    thetas = BOX_LOW + (BOX_HIGH - BOX_LOW) * rng_pool.random((m_pool, 5))
    t0 = time.perf_counter()
    qoi = np.array([ln_power_qoi(t) for t in thetas])
    t_pool = time.perf_counter() - t0
    y_all, y2_all = qoi[:, 0], qoi[:, 1]
    print(f"   {t_pool:.1f} s; std(y) = {y_all.std():.4f}")

    X_box = scale_to_box(thetas)
    idx_test = np.arange(m_pool - n_test, m_pool)
    idx_val = np.arange(n_train_region, n_train_region + n_val)
    y_test = y_all[idx_test]

    # ---- 3. degree scan (one global calibration, frozen) ----------------
    print("== BayesianRidge degree scan ==")
    scan_rows = []
    idx_scan = np.arange(scan_n_train)
    for degree in SCAN_DEGREES:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        F = poly.fit_transform(X_box)
        mu, sd = F[idx_scan].mean(0), F[idx_scan].std(0)
        Fs = (F - mu) / sd
        br = BayesianRidge(fit_intercept=True)
        br.fit(Fs[idx_scan], y_all[idx_scan])
        pred = br.predict(Fs[idx_val])
        rmse = float(np.sqrt(np.mean((pred - y_all[idx_val]) ** 2)))
        rel = rmse / float(y_all[idx_val].std())
        scan_rows.append(dict(degree=degree, P=F.shape[1], rmse=rmse, rel=rel))
        print(f"   degree {degree}: P = {F.shape[1]:4d}, "
              f"val RMSE = {rmse:.5f} ({100 * rel:.2f}% of std)")

    eligible = [
        r for r in scan_rows
        if RMSE_WINDOW[0] <= r["rel"] <= RMSE_WINDOW[1]
        and r["P"] <= MAX_P and min(n_grid) >= 4 * r["P"]
    ]
    checks.record(
        "degree scan lands in the misspecification window",
        len(eligible) > 0,
        f"window = [{100 * RMSE_WINDOW[0]:.0f}%, {100 * RMSE_WINDOW[1]:.0f}%], "
        + ", ".join(f"deg {r['degree']}: {100 * r['rel']:.1f}%" for r in scan_rows),
    )
    if not eligible:
        print("Degree scan failed to land in the target window; stopping "
              "(mis-calibrated example, see acceptance check 2).")
        return 1
    chosen = max(eligible, key=lambda r: r["degree"])
    degree, P = chosen["degree"], chosen["P"]
    print(f"   frozen: degree = {degree}, P = {P} "
          f"(N/P = {', '.join(f'{n / P:.0f}' for n in n_grid)})")

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    F_pool = poly.fit_transform(X_box)

    # ---- 4. replicate fits over the N grid ------------------------------
    print("== replicate fits ==")
    results = {}
    panel_models = None
    for n in n_grid:
        rows = []
        for rep in range(n_reps):
            rng_rep = np.random.default_rng(
                np.random.SeedSequence([master, 2, n, rep])
            )
            idx_tr = rng_rep.choice(n_train_region, size=n, replace=False)
            F_tr_raw, y_tr = F_pool[idx_tr], y_all[idx_tr]
            mu, sd = F_tr_raw.mean(0), F_tr_raw.std(0)
            F_tr = (F_tr_raw - mu) / sd
            F_te = (F_pool[idx_test] - mu) / sd
            want_models = n == 1024 and rep == 0
            out = fit_cell(
                F_tr, y_tr, F_te, y_test, P,
                seed_ell=_seed_int(master, 3, n, rep),
                seed_pops=_seed_int(master, 4, n, rep),
                checks=checks, tag=f"N={n}, rep={rep}",
                n_hc_samples=hc_min, return_models=want_models,
            )
            if want_models:
                row, models = out
                panel_models = (idx_tr, mu, sd, models)
            else:
                row = out
            row["N"], row["rep"] = n, rep
            rows.append(row)
        results[n] = rows
        print(f"   N = {n:6d}: coverage BR/HC/E/PAC = "
              f"{_agg(rows, 'cov_br')[0]:.3f} / {_agg(rows, 'cov_hc')[0]:.3f} / "
              f"{_agg(rows, 'cov_e')[0]:.3f} / {_agg(rows, 'cov_pac')[0]:.3f}, "
              f"bound = {_agg(rows, 'bound')[0]:+.3f}, "
              f"G_test = {_agg(rows, 'G_test')[0]:+.3f}", flush=True)

    n_panel = 1024 if 1024 in n_grid else n_grid[-1]
    if panel_models is None:  # quick grids without N = 1024
        raise RuntimeError("panel replicate (N = 1024, rep = 0) missing")

    # ---- 5. acceptance checks on the aggregated results ------------------
    all_rows = [r for n in n_grid for r in results[n]]
    checks.record(
        "PAC bounds strictly contain bare bounds at every test point",
        all(r["pac_contains_bare"] for r in all_rows),
        f"{sum(r['pac_contains_bare'] for r in all_rows)}/{len(all_rows)} fits",
    )
    checks.record(
        "PAC (phase1) MAP identical to bare optimum",
        all(r["pac_map_matches_bare"] for r in all_rows),
        "coef_ and U_ allclose in every fit",
    )
    bounds_mean = [_agg(results[n], "bound")[0] for n in n_grid]
    finite = all(np.isfinite(r["bound"]) for r in all_rows)
    tail = [b for n, b in zip(n_grid, bounds_mean) if n >= 1024]
    monotone = all(b2 <= b1 + 1e-12 for b1, b2 in zip(tail, tail[1:]))
    checks.record(
        "bound_ finite and monotone non-increasing for N >= 1024",
        finite and monotone,
        "mean bound_ = " + " -> ".join(f"{b:+.3f}" for b in bounds_mean),
    )
    viol = [(r["N"], r["rep"]) for r in all_rows if r["bound"] < r["G_test"]]
    checks.record(
        "bound_ >= G_test for every fit",
        len(viol) == 0,
        "no violations" if not viol else f"violations at {viol}",
    )
    ref_nats = float(np.log(y_test.max() - y_test.min()))
    nonvac = [n for n, b in zip(n_grid, bounds_mean) if b < ref_nats]
    checks.record(
        "certificate non-vacuous (bound_ < ln-uniform reference)",
        len(nonvac) > 0,
        f"reference = {ref_nats:+.3f} nats; non-vacuous at N in {nonvac}",
    )

    # determinism: refit the panel replicate and compare bitwise
    idx_tr, mu, sd, (br0, pops0, ell0, pac0) = panel_models
    F_tr = (F_pool[idx_tr] - mu) / sd
    ell_re = POPSRegressionEllipse(
        rank=RANK, max_iter=MAX_ITER, fit_intercept=True,
        random_state=_seed_int(master, 3, n_panel, 0),
        delta=1e-3 * float(y_all[idx_tr].std()),
    )
    ell_re.fit(F_tr, y_all[idx_tr])
    checks.record(
        "deterministic across reruns",
        np.array_equal(ell_re.coef_, ell0.coef_)
        and np.array_equal(ell_re.U_, ell0.U_),
        f"bitwise-identical refit at N = {n_panel}, rep = 0",
    )

    # ---- 6. panel (a) slice ----------------------------------------------
    wc = np.linspace(BOX_LOW[0], BOX_HIGH[0], 400)
    center = 0.5 * (BOX_LOW + BOX_HIGH)
    th_slice = np.tile(center, (wc.size, 1))
    th_slice[:, 0] = wc
    y_slice = np.array([ln_power_qoi(t)[0] for t in th_slice])
    F_slice = (poly.transform(scale_to_box(th_slice)) - mu) / sd
    m_sl, e_hi, e_lo = ell0.predict(F_slice, return_bounds=True)
    _, p_hi, p_lo = pac0.predict(F_slice, return_bounds=True)
    slice_data = (wc, y_slice, m_sl, e_lo, e_hi, p_lo, p_hi)

    # ---- 7. appendix variants at N = n_panel, single replicate -----------
    print("== appendix: estimator variants ==")
    y_tr = y_all[idx_tr]
    F_te = (F_pool[idx_test] - mu) / sd
    delta = 1e-3 * float(y_tr.std())
    variants = {}
    for name, kwargs in {
        "default (phase1, frozen center)": dict(),
        "optimize_center=True": dict(optimize_center=True),
        "hyperprior_center='warm_start'": dict(hyperprior_center="warm_start"),
    }.items():
        est = POPSRegressionEllipse(
            rank=RANK, max_iter=MAX_ITER, fit_intercept=True,
            random_state=_seed_int(master, 3, n_panel, 0), delta=delta,
            pac_bayes=True, **kwargs,
        )
        est.fit(F_tr, y_tr)
        vm, v_hi, v_lo, v_bstd = est.predict(
            F_te, return_bounds=True, return_bound_std=True
        )
        half_map = 0.5 * (v_hi - v_lo) - 2.0 * v_bstd  # MAP-ellipse support
        logp = projected_ball_logpdf(y_test - vm, half_map, P + 1)
        inside = np.isfinite(logp)
        variants[name] = dict(
            coverage=float(np.mean((y_test >= v_lo) & (y_test <= v_hi))),
            width_std=float((v_hi - v_lo).mean() / y_test.std()),
            objective=float(est.objective_),
            G_test=float(-logp[inside].mean()),
            n_uncovered=int((~inside).sum()),
            bound=float(est.bound_), kl=float(est.kl_),
            gamma=float(est.gamma_), tau2=float(est.tau2_),
            covfrac=float(est.coverage_fraction_), n_iter=int(est.n_iter_),
        )
        print(f"   {name}: bound = {est.bound_:+.3f}, "
              f"coverage = {variants[name]['coverage']:.3f}")
        checks.record(
            f"variant converged & covering ({name})",
            est.coverage_fraction_ == 1.0
            and est.n_iter_ < N_RHO_STAGES * MAX_ITER,
            f"coverage_fraction_ = {est.coverage_fraction_:.4f}, "
            f"n_iter_ = {est.n_iter_}",
        )

    # ---- 8. appendix: secondary QoI at N = n_panel, single replicate -----
    print("== appendix: secondary QoI ln[P(0.15)/P(0.05)] ==")
    y2_tr, y2_te = y2_all[idx_tr], y2_all[idx_test]
    br2 = BayesianRidge(fit_intercept=True)
    br2.fit(F_tr, y2_tr)
    F_val = (F_pool[idx_val] - mu) / sd
    rel2 = float(
        np.sqrt(np.mean((br2.predict(F_val) - y2_all[idx_val]) ** 2))
        / y2_all[idx_val].std()
    )
    row2 = fit_cell(
        F_tr, y2_tr, F_te, y2_te, P,
        seed_ell=_seed_int(master, 5, n_panel, 0),
        seed_pops=_seed_int(master, 6, n_panel, 0),
        checks=checks, tag=f"secondary QoI, N={n_panel}",
        n_hc_samples=hc_min,
    )
    print(f"   val rel RMSE = {100 * rel2:.2f}%, coverage E/PAC = "
          f"{row2['cov_e']:.3f}/{row2['cov_pac']:.3f}")

    # ---- 9. timing -------------------------------------------------------
    timing = {}
    if do_timing:
        print("== timing ==", flush=True)
        rng_t = np.random.default_rng(np.random.SeedSequence([master, 7]))
        idx_t = rng_t.choice(n_train_region, size=16_384, replace=False)
        for label, deg in [("degree 4 (P = 125)", 4)]:
            poly_t = PolynomialFeatures(degree=deg, include_bias=False)
            F_t = poly_t.fit_transform(X_box[idx_t])
            F_t = (F_t - F_t.mean(0)) / F_t.std(0)
            y_t = y_all[idx_t]
            est = POPSRegressionEllipse(
                rank=RANK, max_iter=MAX_ITER, fit_intercept=True,
                random_state=_seed_int(master, 8, deg), pac_bayes=True,
                delta=1e-3 * float(y_t.std()),
            )
            t0 = time.perf_counter()
            est.fit(F_t, y_t)
            dt = time.perf_counter() - t0
            timing[label] = dict(
                P=F_t.shape[1], N=F_t.shape[0], seconds=dt,
                n_iter=int(est.n_iter_),
                converged=bool(est.n_iter_ < N_RHO_STAGES * MAX_ITER),
                covfrac=float(est.coverage_fraction_), bound=float(est.bound_),
            )
            print(f"   {label}: {dt:.1f} s, n_iter_ = {est.n_iter_}")
        if not args.skip_p2000:
            # the paper's standing P ~ 2000 timing: degree 9 gives P = 2001
            poly_t = PolynomialFeatures(degree=9, include_bias=False)
            F_t = poly_t.fit_transform(X_box[idx_t])
            F_t = (F_t - F_t.mean(0)) / F_t.std(0)
            y_t = y_all[idx_t]
            est = POPSRegressionEllipse(
                rank=RANK, max_iter=MAX_ITER, fit_intercept=True,
                random_state=_seed_int(master, 8, 9), pac_bayes=True,
                delta=1e-3 * float(y_t.std()),
            )
            t0 = time.perf_counter()
            est.fit(F_t, y_t)
            dt = time.perf_counter() - t0
            timing["degree 9 (P = 2001)"] = dict(
                P=F_t.shape[1], N=F_t.shape[0], seconds=dt,
                n_iter=int(est.n_iter_),
                converged=bool(est.n_iter_ < N_RHO_STAGES * MAX_ITER),
                covfrac=float(est.coverage_fraction_), bound=float(est.bound_),
            )
            print(f"   degree 9 (P = 2001): {dt:.1f} s, "
                  f"n_iter_ = {est.n_iter_}, converged = "
                  f"{timing['degree 9 (P = 2001)']['converged']}")

    # ---- 10. figure ------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_stem = args.outdir / "eh_emulator"
    coverage = {
        short: (
            np.array([_agg(results[n], key)[0] for n in n_grid]),
            np.array([_agg(results[n], key)[1] for n in n_grid]),
        )
        for short, key in (
            ("br", "cov_br"), ("hc", "cov_hc"),
            ("e", "cov_e"), ("pac", "cov_pac"),
        )
    }
    broadening = (
        np.array([_agg(results[n], "broaden_pct")[0] for n in n_grid]),
        np.array([_agg(results[n], "broaden_pct")[1] for n in n_grid]),
    )
    rel_width = (
        np.array([_agg(results[n], "width_e")[0] for n in n_grid])
        / y_test.std(),
        np.array([_agg(results[n], "width_e")[1] for n in n_grid])
        / y_test.std(),
    )
    make_figure(out_stem, slice_data, n_grid, coverage, broadening, rel_width)
    print(f"figure -> {out_stem}.png / .pdf")

    # ---- 11. summary -----------------------------------------------------
    total_min = (time.perf_counter() - t_start) / 60.0
    write_summary(
        args.outdir / "eh_emulator_summary.md", master, args.quick,
        eh_report, scan_rows, degree, P, n_grid, n_reps, results,
        ref_nats, variants, rel2, row2, timing, checks, y_all, y_test,
        m_pool, n_test, n_val, n_panel, total_min,
    )
    print(f"summary -> {args.outdir / 'eh_emulator_summary.md'}")
    print(f"total: {total_min:.1f} min; acceptance checks "
          f"{'ALL PASSED' if checks.all_passed else 'FAILED'}")
    return 0 if checks.all_passed else 1


# --------------------------------------------------------------------------
# Summary writer
# --------------------------------------------------------------------------


def write_summary(path, master, quick, eh_report, scan_rows, degree, P,
                  n_grid, n_reps, results, ref_nats, variants, rel2, row2,
                  timing, checks, y_all, y_test, m_pool, n_test, n_val,
                  n_panel, total_min):
    def fmt(mean, std, digits=3):
        return f"{mean:.{digits}f} +/- {std:.{digits}f}"

    L = []
    L.append("# Eisenstein-Hu emulator example: summary statistics\n")
    L.append(f"Master seed {master}"
             + (" (QUICK smoke-test mode; not paper numbers)" if quick else "")
             + f"; total runtime {total_min:.1f} min. "
             "Deterministic end-to-end (timing section excepted).\n")
    L.append(f"QoI: y = ln P(k*) at k* = {K_STAR} h/Mpc, sigma8-normalized "
             "EH98 linear matter power spectrum, z = 0, flat universe, "
             f"T_CMB = {T_CMB} K, no noise anywhere (eps = 0). "
             f"Pool M = {m_pool} uniform draws over the box; test = last "
             f"{n_test}; validation = {n_val}; std(y) = {y_all.std():.4f}, "
             f"test range = {y_test.max() - y_test.min():.4f}.\n")

    L.append("## Engine validation\n")
    L.append("| fiducial | s eq.(6) [Mpc] | vs quadrature | vs eq.(26) fit "
             "| k_eq vs exact | wiggle/no-wiggle |")
    L.append("|---|---|---|---|---|---|")
    for name, r in eh_report.items():
        L.append(
            f"| {name} | {r['s']:.3f} | {r['rel_s']:.1e} | {r['rel_fit']:.1e} "
            f"| {r['rel_k']:.1e} | {r['wiggle_crossings']} crossings, "
            f"max {100 * r['wiggle_amplitude']:.1f}% |")
    L.append("\nThe closed-form sound horizon is checked against direct "
             "quadrature of s = int c_s da/(a^2 H) (same early-universe "
             "parameterization), the published eq. 26 fitting form (~2% "
             "accuracy), and k_eq against its exact square-root form; the "
             "wiggle/no-wiggle transfer ratio oscillates about 1 with "
             "percent-level amplitude in the BAO range, as required.\n")

    L.append("## Degree scan (misspecification calibration)\n")
    L.append("| degree | P | val RMSE | RMSE / std(y) |")
    L.append("|---|---|---|---|")
    for r in scan_rows:
        star = " **(frozen)**" if r["degree"] == degree else ""
        L.append(f"| {r['degree']}{star} | {r['P']} | {r['rmse']:.5f} "
                 f"| {100 * r['rel']:.2f}% |")
    L.append(f"\nFrozen choice: degree {degree}, P = {P} "
             f"(BayesianRidge on {SCAN_N_TRAIN if not quick else 'reduced'} "
             f"train rows, validated on the {n_val}-sample split; window "
             f"[{100 * RMSE_WINDOW[0]:.0f}%, {100 * RMSE_WINDOW[1]:.0f}%]). "
             "Higher degrees are too well-specified (below the window "
             "floor), lower too coarse. All quoted fits are safely "
             f"underparametrized: N/P = "
             + ", ".join(f"{n / P:.1f}" for n in n_grid)
             + f" for N = {', '.join(str(n) for n in n_grid)}. "
             f"Support/std ratio sqrt(P + 2) = {np.sqrt(P + 2):.1f}. "
             f"Note rank = {RANK} >= n_dim = {P + 1}: the low-rank update "
             "is full-rank here (rank_ = n_dim), so rank truncation is not "
             "a binding approximation in this example.\n")

    L.append(f"## Test coverage per N (mean +/- std [min] over {n_reps} "
             "replicates)\n")
    L.append("| N | BayesianRidge +/-4sigma | POPS hypercube max/min* "
             "| ellipse support | ellipse+PAC ensemble |")
    L.append("|---|---|---|---|---|")
    for n in n_grid:
        cells = []
        for key in ("cov_br", "cov_hc", "cov_e", "cov_pac"):
            m, s, lo, _ = _agg(results[n], key)
            cells.append(f"{fmt(m, s)} [{lo:.3f}]")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    n_hc = int(np.min([r["n_hc_samples"] for n in n_grid for r in results[n]]))
    L.append(f"\n*Sampled with >= {n_hc} posterior draws per fit. The "
             "sampled hypercube max/min under-covers its own analytic "
             "pushforward by pure concentration of measure in moderate-to-"
             "high P; at this P the effect is mild but present, which is a "
             "genuine advantage of the ellipse's analytic pushforward "
             "(see appendix discussion).\n")
    unc = {n: _agg(results[n], "n_uncovered_test") for n in n_grid}
    L.append("Uncovered test points for the bare ellipse (excluded from "
             "G_test, never clipped): "
             + "; ".join(f"N = {n}: mean {unc[n][0]:.1f}, max {unc[n][3]:.0f}"
                         f" of {len(y_test)}" for n in n_grid) + ".\n")

    L.append("## Band widths\n")
    L.append("| N | ellipse / hypercube | +PAC / hypercube "
             "| ellipse full width / std(y_test) |")
    L.append("|---|---|---|---|")
    for n in n_grid:
        m1, s1, _, _ = _agg(results[n], "ratio_e_hc")
        m2, s2, _, _ = _agg(results[n], "ratio_pac_hc")
        w, sw, _, _ = _agg(results[n], "width_e")
        L.append(f"| {n} | {fmt(100 * m1, 100 * s1, 1)}% "
                 f"| {fmt(100 * m2, 100 * s2, 1)}% "
                 f"| {fmt(w / y_test.std(), sw / y_test.std(), 2)} |")
    L.append("\nWidth ratios are means over the test set of pointwise "
             "band-width ratios; the last column is quoted against the "
             "data spread as a sampling-artifact-free alternative. "
             "Finding: at this moderate P the certified ellipse support is "
             "systematically WIDER than the sampled hypercube max/min range "
             "(ratio > 100%), unlike the O(50-80%) anticipated in the "
             "handoff. The interior-point condition forces the ellipse to "
             "cover every training residual, while the sampled hypercube "
             "band spans only the bulk of the pointwise corrections (its "
             "test coverage is below 1 above); the anticipated regime "
             "presupposes the strong sampling-concentration of much larger "
             "P. Reported as-is, not tuned away.\n")

    L.append("## PAC broadening of the support band\n")
    L.append("| N | mean broadening (+%) |")
    L.append("|---|---|")
    for n in n_grid:
        m, s, _, _ = _agg(results[n], "broaden_pct")
        L.append(f"| {n} | +{fmt(m, s, 1)}% |")
    bro = [f"+{_agg(results[n], 'broaden_pct')[0]:.0f}%" for n in n_grid]
    L.append(f"\nDecay {' -> '.join(bro)} over N: the hyperposterior "
             "concentrates on the phase-1 optimum at rate N.\n")

    L.append("## Certificate vs truth (nats; mean +/- std over "
             "replicates)\n")
    L.append("| N | bound_ | G_test | objective_ (train) | gap "
             "bound_ - G_test | KL | gamma_ |")
    L.append("|---|---|---|---|---|---|---|")
    for n in n_grid:
        b, sb, _, _ = _agg(results[n], "bound")
        g, sg, _, _ = _agg(results[n], "G_test")
        o, so, _, _ = _agg(results[n], "objective")
        gaps = np.array([r["bound"] - r["G_test"] for r in results[n]])
        k, sk, _, _ = _agg(results[n], "kl")
        gm, sgm, _, _ = _agg(results[n], "gamma")
        L.append(f"| {n} | {fmt(b, sb)} | {fmt(g, sg)} | {fmt(o, so)} "
                 f"| {fmt(gaps.mean(), gaps.std())} | {fmt(k, sk, 1)} "
                 f"| {fmt(gm, sgm, 1)} |")
    nonvac = [n for n in n_grid if _agg(results[n], "bound")[0] < ref_nats]
    L.append(f"\nln-uniform reference (uniform density over the test data "
             f"range): {ref_nats:+.3f} nats. The certificate is "
             f"non-vacuous (bound_ below the reference) at "
             f"N in {{{', '.join(str(n) for n in nonvac)}}}"
             + (" - every quoted N." if len(nonvac) == len(n_grid) else ".")
             + "\n")

    L.append(f"## Appendix: estimator variants (N = {n_panel}, single "
             "replicate)\n")
    L.append("| variant | coverage | width/std | objective_ | G_test "
             "| bound_ | KL | gamma_ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, v in variants.items():
        L.append(f"| {name} | {v['coverage']:.3f} | {v['width_std']:.2f} "
                 f"| {v['objective']:+.3f} | {v['G_test']:+.3f} "
                 f"| {v['bound']:+.3f} | {v['kl']:.1f} | {v['gamma']:.1f} |")
    L.append("\n'phase1' centers the hyperprior on the phase-1 optimum "
             "(empirical-Bayes caveat on the formal reading of bound_); "
             "'warm_start' is the caveat-free certificate; "
             "optimize_center=True tightens the fit at the cost of a "
             "less conservative center.\n")

    L.append(f"## Appendix: secondary QoI y = ln[P({K_STAR})/P({K_REF})] "
             f"(N = {n_panel}, single replicate)\n")
    L.append(f"BayesianRidge val RMSE {100 * rel2:.2f}% of std(y2). "
             f"Coverage BR/HC/ellipse/+PAC = {row2['cov_br']:.3f} / "
             f"{row2['cov_hc']:.3f} / {row2['cov_e']:.3f} / "
             f"{row2['cov_pac']:.3f}; bound_ = {row2['bound']:+.3f}, "
             f"G_test = {row2['G_test']:+.3f}, PAC broadening "
             f"+{row2['broaden_pct']:.1f}%. Same mechanics as the primary "
             "QoI: dimensionless BAO-envelope amplitude ratio.\n")

    L.append("## Timing\n")
    if timing:
        for label, t in timing.items():
            conv = "converged" if t["converged"] else "NOT converged (omit)"
            L.append(f"- {label}, rank {RANK}, N = {t['N']}, pac_bayes=True: "
                     f"{t['seconds']:.1f} s, n_iter_ = {t['n_iter']} ({conv})")
        L.append("")
    else:
        L.append("Skipped (quick mode).\n")

    L.append("## Acceptance checks\n")
    for name, passed, detail in checks.rows:
        L.append(f"- [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    L.append(f"\n**{'ALL CHECKS PASSED' if checks.all_passed else 'SOME CHECKS FAILED'}**\n")

    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
