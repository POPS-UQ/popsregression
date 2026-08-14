"""
=====================================================================
Darcy flow: POPS ellipse uncertainty for a high-dimensional surrogate
=====================================================================

Second numerical example for the Sim2Science paper: a misspecified
linear surrogate of a 2D Darcy-flow engine, comparing
:class:`~sklearn.linear_model.BayesianRidge` (collapse baseline),
:class:`~popsregression.POPSRegression` (hypercube baseline),
:class:`~popsregression.POPSRegressionEllipse` (bare ellipse) and the
same estimator with the closed-form PAC-Bayes layer.

Engine: ``-div(a grad u) = 1`` on the unit square with ``u = 0`` on the
boundary, ``a = exp(g)`` a log-Gaussian random field (squared-exponential
kernel, lengthscale 0.2, unit variance), solved by 5-point finite
differences on a 64x64 interior grid with harmonic face averaging. The
engine is deterministic (epsilon = 0): all predictive width comes from
misspecification. Scalar QoI: mean flux through the right boundary,
``y = int a du/dx dx2`` at ``x1 = 1``.

Surrogate: the leading d = 32 KL coordinates of ``g`` (the exact
realization coordinates used to synthesize the field) through a fixed
random-Fourier-feature map of dimension P = 512. Misspecification is
transparent and two-fold: the discarded KL tail is invisible to the
model, and finite P cannot represent the nonlinear z -> flux map.

Deterministic end-to-end from fixed seeds. Outputs (next to this file):
``darcy_flow.pdf``/``.png`` (one 3-panel figure) and
``darcy_flow_summary.txt``/``.json`` (paper summary statistics).

Run ``python darcy_flow.py`` for the full protocol (tens of minutes;
dominated by the N = 4096 ellipse fits) or ``--quick`` for a reduced
smoke test that exercises every code path.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import os
import pickle
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.linalg import eigh
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import BayesianRidge
from threadpoolctl import threadpool_limits

from popsregression import POPSRegression, POPSRegressionEllipse
from popsregression._projected_ball import projected_ball_logpdf

# --------------------------------------------------------------------
# Configuration (all seeds fixed: the whole script is deterministic)
# --------------------------------------------------------------------
N_GRID = 64  # interior FD nodes per dimension
N_NODES = N_GRID + 2  # full nodal grid including the boundary
H_GRID = 1.0 / (N_GRID + 1)
LENGTHSCALE = 0.2  # SE kernel lengthscale of the log-field g

MASTER_SEED = 20260814  # data pool
RFF_SEED = 1234  # the single fixed RFF draw
D_KL = 32  # leading KL coordinates fed to the surrogate
P_RFF = 512  # random Fourier features (default run)
P_RFF_TIMING = 2048  # timing/scaling run
RFF_SIGMA_GRID = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)

M_POOL = 12000
M_TEST = 2000
N_TRAIN_SIZES = (64, 256, 1024, 4096)
N_REPLICATES = 10
RANK = 32
DELTA_REL = 1e-3  # width floor: delta = DELTA_REL * std(y_train)
BR_N_SIGMA = 4.0  # BayesianRidge bounds: +/- 4 epistemic sigma


# --------------------------------------------------------------------
# Engine: 2D Darcy flow, self-contained finite-difference solver
# --------------------------------------------------------------------
def kl_basis(n_nodes=N_NODES, lengthscale=LENGTHSCALE):
    """Exact KL of the SE-kernel GRF on the nodal grid.

    The squared-exponential kernel is separable, so the 2D grid
    eigenpairs are tensor products of 1D eigenpairs with eigenvalue
    products ``lam_i * lam_j``. Returns the 1D eigenvectors ``V``
    (columns, descending eigenvalue), eigenvalues ``lam`` and the
    flattened 2D mode ordering by descending product eigenvalue
    (stable, hence deterministic).
    """
    x = np.linspace(0.0, 1.0, n_nodes)
    K1 = np.exp(-0.5 * ((x[:, None] - x[None, :]) / lengthscale) ** 2)
    lam, V = eigh(K1)
    lam = np.maximum(lam[::-1], 0.0)
    V = V[:, ::-1]
    order = np.argsort(-np.outer(lam, lam).ravel(), kind="stable")
    return V, lam, order


def sample_field(xi, V, lam):
    """Synthesize ``g = sum_ij sqrt(lam_i lam_j) xi_ij v_i v_j^T``."""
    return V @ (np.sqrt(np.outer(lam, lam)) * xi) @ V.T


def darcy_solve(a):
    """5-point FD solve of ``-div(a grad u) = 1``, ``u = 0`` on the boundary.

    ``a`` lives on the full ``(N_GRID + 2)^2`` nodal grid; face
    conductivities are harmonic means of the adjacent nodal values. The
    sparse system is SPD; returns ``u`` on the full nodal grid.
    """
    n = N_GRID
    h2 = H_GRID * H_GRID

    def harm(a1, a2):
        return 2.0 * a1 * a2 / (a1 + a2)

    ai = a[1:-1, 1:-1]
    aW = harm(a[:-2, 1:-1], ai)
    aE = harm(ai, a[2:, 1:-1])
    aS = harm(a[1:-1, :-2], ai)
    aN = harm(ai, a[1:-1, 2:])

    # flatten k = (ix - 1) * n + (iy - 1): y-neighbours at k +/- 1,
    # x-neighbours at k +/- n; boundary neighbours drop out of the stencil
    diag = (aW + aE + aS + aN).ravel() / h2
    off_N = (-aN / h2).ravel()
    off_S = (-aS / h2).ravel()
    off_E = (-aE / h2).ravel()
    off_W = (-aW / h2).ravel()
    off_N.reshape(n, n)[:, -1] = 0.0
    off_S.reshape(n, n)[:, 0] = 0.0

    A = sp.diags(
        [diag, off_N[:-1], off_S[1:], off_E[:-n], off_W[n:]],
        [0, 1, -1, n, -n],
        format="csc",
    )
    u = np.zeros((n + 2, n + 2))
    u[1:-1, 1:-1] = spsolve(A, np.ones(n * n)).reshape(n, n)
    return u


def flux_qoi(a, u):
    """Mean flux through the right boundary: ``int a du/dx dx2`` at x1 = 1.

    One-sided finite difference in x, trapezoid in x2 (the integrand
    vanishes at the corners since u = 0 along the whole boundary).
    """
    q = a[-1, :] * (u[-1, :] - u[-2, :]) / H_GRID
    return 0.5 * H_GRID * float(np.sum(q[1:] + q[:-1]))


def sensor_qoi(u, x1=0.75, x2=0.5):
    """Alternative QoI for robustness checks: u at a fixed interior sensor."""
    n = N_NODES - 1
    fx, fy = x1 * n, x2 * n
    i, j = int(fx), int(fy)
    tx, ty = fx - i, fy - j
    return (
        u[i, j] * (1 - tx) * (1 - ty)
        + u[i + 1, j] * tx * (1 - ty)
        + u[i, j + 1] * (1 - tx) * ty
        + u[i + 1, j + 1] * tx * ty
    )


def validate_solver():
    """Check the FD solver against the analytic series for a == 1.

    For ``a == 1`` the PDE is ``-Lap u = 1`` with the classical double
    sine series solution; the 5-point scheme must agree to O(h^2).
    """
    u_fd = darcy_solve(np.ones((N_NODES, N_NODES)))
    x = np.linspace(0.0, 1.0, N_NODES)
    u_ref = np.zeros((N_NODES, N_NODES))
    for m in range(1, 101, 2):
        for k in range(1, 101, 2):
            c = 16.0 / (np.pi**4 * m * k * (m * m + k * k))
            u_ref += c * np.outer(np.sin(m * np.pi * x), np.sin(k * np.pi * x))
    rel_err = np.linalg.norm(u_fd - u_ref) / np.linalg.norm(u_ref)
    assert rel_err < 1e-3, f"solver validation failed: rel err {rel_err:.3e}"
    return rel_err


def generate_pool(m_pool):
    """Generate the (field, QoI) pool from the master seed.

    The surrogate inputs are the exact KL coordinates used to
    synthesize each field (leading D_KL modes of the sorted product
    spectrum) — no re-projection. Single-threaded BLAS keeps the pool
    bit-reproducible.
    """
    with threadpool_limits(limits=1):
        V, lam, order = kl_basis()
        rng = np.random.default_rng(MASTER_SEED)
        Z = np.empty((m_pool, D_KL))
        y_flux = np.empty(m_pool)
        y_sensor = np.empty(m_pool)
        for m in range(m_pool):
            xi = rng.standard_normal((N_NODES, N_NODES))
            Z[m] = xi.ravel()[order[:D_KL]]
            a = np.exp(sample_field(xi, V, lam))
            u = darcy_solve(a)
            y_flux[m] = flux_qoi(a, u)
            y_sensor[m] = sensor_qoi(u)
    return Z, y_flux, y_sensor


# --------------------------------------------------------------------
# Surrogate: random Fourier features of the leading KL coordinates
# --------------------------------------------------------------------
def rff_draw(p_rff, seed=RFF_SEED):
    """The single fixed RFF draw (unit-scale frequencies, phases)."""
    rng = np.random.default_rng(seed)
    omega0 = rng.standard_normal((D_KL, p_rff))
    b = rng.uniform(0.0, 2.0 * np.pi, p_rff)
    return omega0, b


def rff_features(Z, omega0, b, sigma):
    """``phi(z) = sqrt(2/P) cos(Omega z + b)`` with ``Omega = omega0 / sigma``."""
    p_rff = omega0.shape[1]
    return np.sqrt(2.0 / p_rff) * np.cos(Z @ (omega0 / sigma) + b)


def standardize(F_train, *F_other):
    """Per-column standardization, fit on the train split only."""
    mu = F_train.mean(axis=0)
    sd = F_train.std(axis=0)
    sd = np.where(sd > 0.0, sd, 1.0)
    return tuple((F - mu) / sd for F in (F_train,) + F_other)


def tune_sigma_rff(Z_pool, y_pool, omega0, b):
    """Tune the RFF lengthscale once on a coarse grid (never per N).

    BayesianRidge validation RMSE on a fixed train/validation split of
    the training pool (both disjoint from the dense test set).
    """
    n_pool = Z_pool.shape[0]
    tr = slice(0, min(2048, n_pool // 2))
    va = slice(max(n_pool - 2000, n_pool // 2), n_pool)
    results = []
    with threadpool_limits(limits=1):
        for sigma in RFF_SIGMA_GRID:
            F_tr, F_va = standardize(
                rff_features(Z_pool[tr], omega0, b, sigma),
                rff_features(Z_pool[va], omega0, b, sigma),
            )
            br = BayesianRidge()
            br.fit(F_tr, y_pool[tr])
            rmse = float(np.sqrt(np.mean((br.predict(F_va) - y_pool[va]) ** 2)))
            results.append((sigma, rmse))
    sigma_best = min(results, key=lambda r: r[1])[0]
    return sigma_best, results


# --------------------------------------------------------------------
# Protocol: four models per (N, replicate)
# --------------------------------------------------------------------
def fit_replicate(F_train, y_train, F_test, y_test, rep, pops_seed):
    """Fit the four models on one standardized train subset.

    Returns a dict of per-model records. The bare ellipse and the
    PAC-Bayes ellipse share the same phase-1 optimum by construction
    (``hyperprior_center='phase1'`` never changes the fit), which is
    asserted here as the containment invariant.
    """
    F_tr, F_te = standardize(F_train, F_test)
    delta = DELTA_REL * float(y_train.std())
    rec = {"delta": delta}

    # 1. BayesianRidge: epistemic-only collapse baseline, bounds +/- 4 sigma
    t0 = time.perf_counter()
    br = BayesianRidge()
    br.fit(F_tr, y_train)
    m = br.predict(F_te)
    s_epi = np.sqrt(np.sum((F_te @ br.sigma_) * F_te, axis=1))
    rec["br"] = {
        "time": time.perf_counter() - t0,
        "coverage": float(np.mean(np.abs(y_test - m) <= BR_N_SIGMA * s_epi)),
        "mean_width": float(np.mean(2.0 * BR_N_SIGMA * s_epi)),
    }

    # 2. POPS hypercube baseline: sampled max/min bounds (existing API).
    #    The hypercube sampler draws from the global numpy RNG; seed it
    #    so reruns are bit-identical.
    t0 = time.perf_counter()
    np.random.seed(pops_seed)
    hyc = POPSRegression(fit_intercept=True)
    hyc.fit(F_tr, y_train)
    _, h_max, h_min = hyc.predict(F_te, return_bounds=True)
    rec["hypercube"] = {
        "time": time.perf_counter() - t0,
        "coverage": float(np.mean((y_test >= h_min) & (y_test <= h_max))),
        "mean_width": float(np.mean(h_max - h_min)),
    }

    # 3. Bare ellipse: support bounds m +/- sqrt(v)
    t0 = time.perf_counter()
    ell = POPSRegressionEllipse(
        rank=RANK, delta=delta, random_state=rep, fit_intercept=True
    )
    ell.fit(F_tr, y_train)
    _, e_max, e_min = ell.predict(F_te, return_bounds=True)
    rec["ellipse"] = {
        "time": time.perf_counter() - t0,
        "coverage": float(np.mean((y_test >= e_min) & (y_test <= e_max))),
        "mean_width": float(np.mean(e_max - e_min)),
        "coverage_fraction": float(ell.coverage_fraction_),
        "objective": float(ell.objective_),
        "n_iter": int(ell.n_iter_),
    }

    # 4. Ellipse + closed-form PAC-Bayes layer: bounds m +/- (sqrt(v) + 2 s_b)
    t0 = time.perf_counter()
    pac = POPSRegressionEllipse(
        rank=RANK,
        delta=delta,
        random_state=rep,
        fit_intercept=True,
        pac_bayes=True,
    )
    pac.fit(F_tr, y_train)
    m_p, p_max, p_min, p_bstd = pac.predict(
        F_te, return_bounds=True, return_bound_std=True
    )
    rec["pac"] = {
        "time": time.perf_counter() - t0,
        "coverage": float(np.mean((y_test >= p_min) & (y_test <= p_max))),
        "mean_width": float(np.mean(p_max - p_min)),
        "coverage_fraction": float(pac.coverage_fraction_),
        "objective": float(pac.objective_),
        "n_iter": int(pac.n_iter_),
        "bound": float(pac.bound_),
        "kl": float(pac.kl_),
        "gamma": float(pac.gamma_),
        "tau2": float(pac.tau2_),
    }

    # Test-set empirical generalization error G_hat_test of the fitted
    # (phase-1) ellipse: exact projected-ball log density with support
    # half-width sqrt(v), recovered as half the bare-bound spread (the
    # PAC bounds minus the 2 sigma_b ensemble broadening). Test points
    # outside the support are coverage failures, reported separately —
    # never clipped into the barrier.
    half_width = 0.5 * (p_max - p_min) - 2.0 * p_bstd
    logp = projected_ball_logpdf(y_test - m_p, half_width, P_RFF + 1)
    covered = np.isfinite(logp)
    rec["G_test"] = float(-logp[covered].mean())
    rec["G_test_n_uncovered"] = int(np.sum(~covered))

    # PAC broadening over the bare support (mean over test points)
    rec["pac_broadening"] = float(np.mean((p_max - p_min) / (e_max - e_min) - 1.0))

    # Invariants: identical phase-1 optimum; strict containment of the
    # bare bounds in the PAC bounds at every test point.
    assert np.allclose(
        e_max, p_max - 2.0 * p_bstd, rtol=1e-9, atol=1e-12
    ), "phase-1 optima of bare and PAC fits differ"
    assert np.all(p_max >= e_max) and np.all(
        p_min <= e_min
    ), "PAC bounds do not contain bare ellipse bounds"

    # Loud coverage flag (rather than silently widening delta)
    for name in ("ellipse", "pac"):
        if rec[name]["coverage_fraction"] != 1.0:
            print(
                f"*** WARNING: {name} fit has coverage_fraction_ = "
                f"{rec[name]['coverage_fraction']:.6f} != 1.0 "
                f"(rep={rep}) — barrier failed to cover all training "
                "points; do NOT silently widen delta. ***",
                flush=True,
            )

    rec["panel_a"] = (m_p, p_max, p_min, e_max, e_min)
    return rec


def run_protocol(
    Z_pool,
    y_pool,
    Z_test,
    y_test,
    omega0,
    b,
    sigma_rff,
    train_sizes,
    n_replicates,
    n_jobs=1,
):
    """Full data protocol: all N, all replicates, fixed feature draw.

    Replicates are embarrassingly parallel; every replicate is seeded
    independently and runs with single-threaded BLAS, so the results
    are bit-identical for any ``n_jobs`` (multi-threaded BLAS changes
    reduction order, which perturbs the L-BFGS iterate stream at the
    last digit).
    """
    with threadpool_limits(limits=1):
        F_test_raw = rff_features(Z_test, omega0, b, sigma_rff)
    results = {}
    panel_a = {}
    for n_train in train_sizes:

        def one_rep(rep):
            with threadpool_limits(limits=1):
                rng = np.random.default_rng([rep, n_train, MASTER_SEED])
                idx = rng.choice(Z_pool.shape[0], size=n_train, replace=False)
                F_train_raw = rff_features(Z_pool[idx], omega0, b, sigma_rff)
                return fit_replicate(
                    F_train_raw,
                    y_pool[idx],
                    F_test_raw,
                    y_test,
                    rep,
                    pops_seed=12345 + 1000 * rep + n_train,
                )

        reps = Parallel(n_jobs=n_jobs)(
            delayed(one_rep)(rep) for rep in range(n_replicates)
        )
        for rep, rec in enumerate(reps):
            if rep == 0:
                panel_a[n_train] = rec.pop("panel_a")
            else:
                rec.pop("panel_a")
            t_all = sum(rec[k]["time"] for k in ("br", "hypercube", "ellipse", "pac"))
            print(
                f"N={n_train:<5d} rep={rep}: "
                f"cov(BR,H,E,P)=({rec['br']['coverage']:.3f},"
                f"{rec['hypercube']['coverage']:.3f},"
                f"{rec['ellipse']['coverage']:.3f},"
                f"{rec['pac']['coverage']:.3f})  "
                f"bound={rec['pac']['bound']:+.3f}  "
                f"G_test={rec['G_test']:+.3f}  "
                f"t={t_all:.1f}s",
                flush=True,
            )
        results[n_train] = reps
    return results, panel_a


# --------------------------------------------------------------------
# Timing and appendix runs
# --------------------------------------------------------------------
def timing_run(Z_pool, y_pool, sigma_rff, n_train=4096):
    """Single-threaded wall-clock of one PAC ellipse fit at P = 2048.

    Includes feature standardization and the internal POPS pre-fit
    (baseline='pops'), as quoted in the paper.
    """
    omega0, b = rff_draw(P_RFF_TIMING, seed=RFF_SEED)
    rng = np.random.default_rng([0, n_train, MASTER_SEED])
    idx = rng.choice(Z_pool.shape[0], size=n_train, replace=False)
    F_train_raw = rff_features(Z_pool[idx], omega0, b, sigma_rff)
    y_train = y_pool[idx]
    delta = DELTA_REL * float(y_train.std())
    with threadpool_limits(limits=1):
        t0 = time.perf_counter()
        (F_tr,) = standardize(F_train_raw)
        model = POPSRegressionEllipse(
            rank=RANK,
            delta=delta,
            random_state=0,
            fit_intercept=True,
            pac_bayes=True,
        )
        model.fit(F_tr, y_train)
        wall = time.perf_counter() - t0
    return {
        "wall_seconds": wall,
        "n_train": n_train,
        "p_rff": P_RFF_TIMING,
        "rank": RANK,
        "coverage_fraction": float(model.coverage_fraction_),
        "bound": float(model.bound_),
        "n_iter": int(model.n_iter_),
    }


def appendix_runs(Z_pool, y_pool, Z_test, y_test, omega0, b, sigma_rff, n_train=256):
    """Appendix variants at one N (single replicate, rep 0).

    - ``optimize_center=True``: joint center/width optimization
      (tighter, less conservative at small N).
    - ``hyperprior_center='warm_start'``: prior center chosen without
      seeing the phase-1 optimum, so ``bound_`` is free of the
      empirical-Bayes (data-dependent prior) caveat of ``'phase1'``.
    """
    rng = np.random.default_rng([0, n_train, MASTER_SEED])
    idx = rng.choice(Z_pool.shape[0], size=n_train, replace=False)
    out = {}
    with threadpool_limits(limits=1):
        F_tr, F_te = standardize(
            rff_features(Z_pool[idx], omega0, b, sigma_rff),
            rff_features(Z_test, omega0, b, sigma_rff),
        )
        y_train = y_pool[idx]
        delta = DELTA_REL * float(y_train.std())
        for tag, kwargs in [
            ("phase1_center_frozen", {}),
            ("optimize_center", {"optimize_center": True}),
            ("warm_start_center", {"hyperprior_center": "warm_start"}),
        ]:
            model = POPSRegressionEllipse(
                rank=RANK,
                delta=delta,
                random_state=0,
                fit_intercept=True,
                pac_bayes=True,
                **kwargs,
            )
            model.fit(F_tr, y_train)
            _, p_max, p_min, _ = model.predict(
                F_te, return_bounds=True, return_bound_std=True
            )
            out[tag] = {
                "bound": float(model.bound_),
                "kl": float(model.kl_),
                "objective": float(model.objective_),
                "coverage_fraction": float(model.coverage_fraction_),
                "test_coverage": float(np.mean((y_test >= p_min) & (y_test <= p_max))),
                "mean_width": float(np.mean(p_max - p_min)),
            }
    return out


# --------------------------------------------------------------------
# Figure: 3 panels, full page width, ~0.35 page height
# --------------------------------------------------------------------
def make_figure(results, panel_a, y_test, train_sizes, path_base):
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    ax_a, ax_b, ax_c = axes
    n_arr = np.array(train_sizes, dtype=float)
    y_std = float(y_test.std())

    # ---- (a) predictions at small N (one replicate) ----
    # In P = 513 feature dimensions the support half-width is
    # sqrt(P + 2) ~ 23x the pushforward std, so support bands dwarf the
    # data spread when N << P (whitening-inflated extrapolation
    # directions). Show the smallest N whose bands stay legible.
    panel_a_n = train_sizes[-1]
    for n in train_sizes:
        mean_width = np.mean([r["pac"]["mean_width"] for r in results[n]])
        if mean_width <= 15.0 * y_std:
            panel_a_n = n
            break
    m_p, p_max, p_min, e_max, e_min = panel_a[panel_a_n]
    order = np.argsort(y_test)
    sub = order[np.linspace(0, y_test.size - 1, 200).astype(int)]
    rank = np.arange(sub.size)
    ax_a.fill_between(
        rank,
        p_min[sub],
        p_max[sub],
        facecolor="0.5",
        alpha=0.25,
        label="PAC ensemble bounds",
        lw=0,
    )
    ax_a.fill_between(
        rank,
        e_min[sub],
        e_max[sub],
        facecolor="C1",
        alpha=0.45,
        label="ellipse support",
        lw=0,
    )
    ax_a.plot(rank, m_p[sub], "C1-", lw=1.2, label="predicted mean")
    ax_a.plot(rank, y_test[sub], "k-", lw=0.8, label="true QoI")
    ax_a.set_xlabel(f"test point (sorted by true QoI), $N = {panel_a_n}$")
    ax_a.set_ylabel(r"boundary flux $y$")
    ax_a.legend(fontsize=6.5, loc="upper left", frameon=False)

    # ---- (b) coverage & width vs N ----
    styles = {
        "br": ("C2", "s", "BayesianRidge"),
        "hypercube": ("0.35", "D", "POPS hypercube"),
        "ellipse": ("C1", "o", "POPS ellipse"),
        "pac": ("C0", "^", "ellipse + PAC"),
    }
    for key, (color, marker, label) in styles.items():
        cov = np.array([[r[key]["coverage"] for r in results[n]] for n in train_sizes])
        ax_b.errorbar(
            n_arr,
            cov.mean(axis=1),
            yerr=cov.std(axis=1),
            color=color,
            marker=marker,
            ms=3.5,
            lw=1.2,
            capsize=2,
            label=label,
        )
    ax_b.set_xscale("log")
    ax_b.set_xticks(n_arr, [str(n) for n in train_sizes])
    ax_b.minorticks_off()
    ax_b.set_xlabel(r"training size $N$")
    ax_b.set_ylabel("test coverage")
    ax_b.set_ylim(-0.04, 1.09)
    ax_b.axhline(1.0, color="0.8", lw=0.6, zorder=0)
    # mean bound width in units of std(y_test), log scale (dashed):
    # the sampled hypercube collapses whenever the mean fit
    # interpolates, so the spec's width-relative-to-hypercube ratio is
    # reported in the summary table instead where it is finite.
    ax_bw = ax_b.twinx()
    for key, (color, marker, label) in styles.items():
        w = np.array(
            [[r[key]["mean_width"] / y_std for r in results[n]] for n in train_sizes]
        )
        ax_bw.plot(n_arr, w.mean(axis=1), color=color, lw=1.0, ls="--", alpha=0.6)
    ax_bw.set_yscale("log")
    ax_bw.set_ylabel(r"mean width / std($y$)  (dashed)", fontsize=8)
    ax_b.legend(fontsize=6.5, loc=(0.02, 0.42), frameon=False)

    # ---- (c) certificate vs truth ----
    curves = {
        "bound": ("C0", "o", "-", r"PAC certificate (bound$\_$)"),
        "G_test": ("k", "s", "--", r"$\hat G_{\rm test}$"),
        "objective": ("C1", "^", ":", r"$\hat G_{\rm train}$ (objective$\_$)"),
    }
    for key, (color, marker, ls, label) in curves.items():
        if key == "bound":
            vals = np.array(
                [[r["pac"]["bound"] for r in results[n]] for n in train_sizes]
            )
        elif key == "G_test":
            vals = np.array([[r["G_test"] for r in results[n]] for n in train_sizes])
        else:
            vals = np.array(
                [[r["pac"]["objective"] for r in results[n]] for n in train_sizes]
            )
        ax_c.errorbar(
            n_arr,
            vals.mean(axis=1),
            yerr=vals.std(axis=1),
            color=color,
            marker=marker,
            ms=3.5,
            lw=1.2,
            ls=ls,
            capsize=2,
            label=label,
        )
    # 1/N decay guide anchored where the mean bound peaks
    mean_bounds = np.array(
        [np.mean([r["pac"]["bound"] for r in results[n]]) for n in train_sizes]
    )
    i0 = int(np.argmax(mean_bounds))
    g_inf = np.mean([r["G_test"] for r in results[train_sizes[-1]]])
    guide = g_inf + (mean_bounds[i0] - g_inf) * n_arr[i0] / n_arr
    ax_c.plot(
        n_arr[i0:],
        guide[i0:],
        color="0.75",
        lw=0.8,
        zorder=0,
        label=r"$\sim 1/N$",
    )
    ax_c.set_xscale("log")
    ax_c.set_xticks(n_arr, [str(n) for n in train_sizes])
    ax_c.minorticks_off()
    ax_c.set_xlabel(r"training size $N$")
    ax_c.set_ylabel("generalization error (nats)")
    ax_c.legend(fontsize=6.5, frameon=False)

    for ax, tag in zip(axes, ("(a)", "(b)", "(c)")):
        ax.text(
            0.02,
            0.98,
            tag,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )
    fig.tight_layout(pad=0.4)
    fig.savefig(path_base + ".pdf")
    fig.savefig(path_base + ".png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------
# Summary statistics for the paper text
# --------------------------------------------------------------------
def summarize(results, train_sizes, sigma_rff, sigma_scan, rel_err, timing, appendix):
    lines = []

    def add(s=""):
        lines.append(s)
        print(s, flush=True)

    add("Darcy flow example — summary statistics")
    add("=" * 60)
    add(f"solver validation (a == 1) rel. L2 error: {rel_err:.3e}")
    add(
        f"sigma_RFF = {sigma_rff}  (BayesianRidge validation scan: "
        + ", ".join(f"{s:g}: {r:.3e}" for s, r in sigma_scan)
        + ")"
    )
    add()

    add("Test coverage (mean +/- std over replicates)")
    add(
        f"{'N':>6} {'BayesianRidge':>16} {'hypercube':>16} "
        f"{'ellipse':>16} {'ellipse+PAC':>16}"
    )
    for n in train_sizes:
        row = f"{n:>6}"
        for key in ("br", "hypercube", "ellipse", "pac"):
            c = np.array([r[key]["coverage"] for r in results[n]])
            row += f"  {c.mean():.3f} +/- {c.std():.3f}"
        add(row)
    add()

    add("Mean bound width relative to hypercube (mean +/- std)")
    add(f"{'N':>6} {'ellipse':>16} {'ellipse+PAC':>16}")
    for n in train_sizes:
        row = f"{n:>6}"
        for key in ("ellipse", "pac"):
            w = np.array(
                [
                    r[key]["mean_width"] / r["hypercube"]["mean_width"]
                    for r in results[n]
                ]
            )
            row += f"  {w.mean():.3f} +/- {w.std():.3f}"
        add(row)
    add()

    add("PAC broadening over bare ellipse: mean(PAC width / bare width - 1)")
    row1, row2 = f"{'N':>6}", f"{'+%':>6}"
    for n in train_sizes:
        broad = np.array([r["pac_broadening"] for r in results[n]])
        row1 += f" {n:>10}"
        row2 += f" {100 * broad.mean():>+9.1f}%"
    add(row1)
    add(row2)
    add()

    add("PAC certificate vs truth (mean +/- std; nats)")
    add(
        f"{'N':>6} {'bound_':>16} {'G_test':>16} {'gap':>16} "
        f"{'train objective_':>18} {'uncovered':>10}"
    )
    for n in train_sizes:
        bound = np.array([r["pac"]["bound"] for r in results[n]])
        gtest = np.array([r["G_test"] for r in results[n]])
        obj = np.array([r["pac"]["objective"] for r in results[n]])
        unc = np.array([r["G_test_n_uncovered"] for r in results[n]])
        add(
            f"{n:>6}  {bound.mean():+.3f} +/- {bound.std():.3f}"
            f"  {gtest.mean():+.3f} +/- {gtest.std():.3f}"
            f"  {(bound - gtest).mean():+.3f} +/- {(bound - gtest).std():.3f}"
            f"  {obj.mean():+.3f} +/- {obj.std():.3f}"
            f"  {unc.mean():>8.1f}"
        )
    add()

    add("PAC diagnostics (mean over replicates)")
    add(f"{'N':>6} {'kl_':>10} {'gamma_':>10} {'covfrac':>10} {'n_iter_':>10}")
    for n in train_sizes:
        kl = np.mean([r["pac"]["kl"] for r in results[n]])
        ga = np.mean([r["pac"]["gamma"] for r in results[n]])
        cf = np.min([r["pac"]["coverage_fraction"] for r in results[n]])
        ni = np.mean([r["pac"]["n_iter"] for r in results[n]])
        add(f"{n:>6} {kl:>10.2f} {ga:>10.1f} {cf:>10.4f} {ni:>10.0f}")
    add()

    if timing is not None:
        add(
            f"Timing: single-threaded PAC ellipse fit at P={timing['p_rff']}, "
            f"rank={timing['rank']}, N={timing['n_train']} "
            f"(incl. POPS pre-fit): {timing['wall_seconds']:.1f} s "
            f"({timing['n_iter']} L-BFGS iterations)"
        )
        add()

    if appendix is not None:
        add("Appendix variants at N = 256 (single replicate)")
        add(
            f"{'variant':>22} {'bound_':>10} {'test cov':>10} "
            f"{'mean width':>12} {'covfrac':>9}"
        )
        for tag, r in appendix.items():
            add(
                f"{tag:>22} {r['bound']:>+10.3f} {r['test_coverage']:>10.3f} "
                f"{r['mean_width']:>12.4f} {r['coverage_fraction']:>9.4f}"
            )
        add(
            "  (phase1 bound_ carries the empirical-Bayes caveat of a "
            "data-dependent prior center; warm_start does not.)"
        )
        add()

    add("Notes for the paper text (observed regimes)")
    add("-" * 60)
    add(
        "* N < P (interpolation regime, N = 64 and 256 with P = 512): the\n"
        "  noise-free engine lets the BayesianRidge mean interpolate, so\n"
        "  its epistemic band covers via null-space variance (coverage\n"
        "  near 1), the POPS pointwise corrections nearly vanish (sampled\n"
        "  hypercube collapses), and the ellipse support at test points\n"
        "  is inflated by whitening-clipped extrapolation directions.\n"
        "  The section-6 expectations hold in the N >= P regime."
    )
    add(
        "* The sampled hypercube max/min under-covers at every N here: in\n"
        "  P = 513 dimensions the projections of uniformly sampled box\n"
        "  points concentrate far inside the true box support (a purely\n"
        "  high-dimensional effect, absent in the P = 5 quartic example).\n"
        "  The optimized ellipse pushforward needs no sampling and does\n"
        "  not suffer from this."
    )
    add(
        "* The support half-width is sqrt(P + 2) ~ 23x the pushforward\n"
        "  std in P = 513 dimensions, so support bands are intrinsically\n"
        "  several times the data spread even at N/P = 8."
    )
    add(
        "* bound_ exceeds G_hat_test only once N >= P (certificate valid\n"
        "  from N = 1024 on, gap tightening ~1/N); at smaller N the\n"
        "  empirical-Bayes ('phase1') centering and the subgamma_const=0\n"
        "  idealization leave the certificate heuristic, as flagged in\n"
        "  App. C."
    )
    add()

    return "\n".join(lines) + "\n"


def acceptance_checks(results, train_sizes, rel_err):
    """Section-8 acceptance checks. Failures are reported loudly (and
    returned) rather than aborting, so the figure and summary are always
    produced for inspection."""
    checks = []
    checks.append(("solver validation rel err < 1e-3", rel_err < 1e-3, ""))
    covfracs = [
        r[key]["coverage_fraction"]
        for n in train_sizes
        for r in results[n]
        for key in ("ellipse", "pac")
    ]
    checks.append(
        (
            "all fits coverage_fraction_ == 1.0",
            all(c == 1.0 for c in covfracs),
            f"min = {min(covfracs):.6f}",
        )
    )
    # strict containment is asserted per replicate inside fit_replicate
    checks.append(
        (
            "PAC bounds contain bare bounds at every test point",
            True,
            "asserted per replicate",
        )
    )
    mean_bounds = [
        float(np.mean([r["pac"]["bound"] for r in results[n]])) for n in train_sizes
    ]
    checks.append(
        (
            "mean bound_ monotone decreasing in N",
            all(b1 > b2 for b1, b2 in zip(mean_bounds, mean_bounds[1:])),
            "sequence: " + ", ".join(f"{b:+.2f}" for b in mean_bounds),
        )
    )
    lines = ["Acceptance checks (handoff section 8):"]
    for name, ok, note in checks:
        lines.append(
            f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({note})" if note else "")
        )
    all_ok = all(ok for _, ok, _ in checks)
    if not all_ok:
        lines.append(
            "  *** One or more acceptance checks FAILED — see notes in the "
            "summary before quoting numbers. ***"
        )
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    return text, all_ok


# --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument(
        "--quick",
        action="store_true",
        help="reduced smoke test (small pool, fewer N and replicates)",
    )
    ap.add_argument(
        "--pool-cache",
        default=None,
        metavar="NPZ",
        help="optional npz cache of the data pool (regenerated bit-"
        "identically from the master seed when absent)",
    )
    ap.add_argument(
        "--outdir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="output directory for figure and summary files",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="J",
        help="parallel workers over replicates (results are identical "
        "for any value; each replicate is independently seeded)",
    )
    ap.add_argument(
        "--state-cache",
        default=None,
        metavar="PKL",
        help="optional pickle of all fit results, for re-rendering the "
        "figure/summary without re-running the protocol",
    )
    ap.add_argument(
        "--render-only",
        action="store_true",
        help="load --state-cache and regenerate only the figure and " "summary files",
    )
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    t_start = time.time()
    if args.render_only:
        with open(args.state_cache, "rb") as fh:
            state = pickle.load(fh)
        (
            results,
            panel_a,
            y_test,
            train_sizes,
            sigma_rff,
            sigma_scan,
            rel_err,
            timing,
            appendix,
        ) = state
    else:
        if args.quick:
            m_pool, m_test = 1200, 400
            train_sizes, n_replicates = (64, 256), 3
            do_timing = do_appendix = False
        else:
            m_pool, m_test = M_POOL, M_TEST
            train_sizes, n_replicates = N_TRAIN_SIZES, N_REPLICATES
            do_timing = do_appendix = True

        rel_err = validate_solver()
        print(f"solver validation: rel L2 err {rel_err:.3e} (< 1e-3)", flush=True)

        if args.pool_cache is not None and os.path.exists(args.pool_cache):
            data = np.load(args.pool_cache)
            Z, y_flux = data["Z"], data["y_flux"]
            assert Z.shape[0] == m_pool, "cached pool size mismatch"
            assert int(data["master_seed"]) == MASTER_SEED, "cached pool seed mismatch"
        else:
            print(f"generating {m_pool}-sample pool ...", flush=True)
            Z, y_flux, y_sensor = generate_pool(m_pool)
            if args.pool_cache is not None:
                np.savez_compressed(
                    args.pool_cache,
                    Z=Z,
                    y_flux=y_flux,
                    y_sensor=y_sensor,
                    master_seed=MASTER_SEED,
                )
        print(f"pool ready ({time.time() - t_start:.0f}s)", flush=True)

        # dense test set: the last M_TEST samples, never trained on
        Z_pool, y_pool = Z[: m_pool - m_test], y_flux[: m_pool - m_test]
        Z_test, y_test = Z[m_pool - m_test :], y_flux[m_pool - m_test :]

        omega0, b = rff_draw(P_RFF)
        sigma_rff, sigma_scan = tune_sigma_rff(Z_pool, y_pool, omega0, b)
        print(f"tuned sigma_RFF = {sigma_rff}", flush=True)

        results, panel_a = run_protocol(
            Z_pool,
            y_pool,
            Z_test,
            y_test,
            omega0,
            b,
            sigma_rff,
            train_sizes,
            n_replicates,
            n_jobs=args.jobs,
        )

        timing = timing_run(Z_pool, y_pool, sigma_rff) if do_timing else None
        appendix = (
            appendix_runs(Z_pool, y_pool, Z_test, y_test, omega0, b, sigma_rff)
            if do_appendix
            else None
        )
        if args.state_cache is not None:
            with open(args.state_cache, "wb") as fh:
                pickle.dump(
                    (
                        results,
                        panel_a,
                        y_test,
                        train_sizes,
                        sigma_rff,
                        sigma_scan,
                        rel_err,
                        timing,
                        appendix,
                    ),
                    fh,
                )

    path_base = os.path.join(args.outdir, "darcy_flow")
    make_figure(results, panel_a, y_test, train_sizes, path_base)
    print(f"figure written to {path_base}.pdf/.png", flush=True)

    text = summarize(
        results, train_sizes, sigma_rff, sigma_scan, rel_err, timing, appendix
    )
    acc_text, acc_ok = acceptance_checks(results, train_sizes, rel_err)
    text += "\n" + acc_text + "\n"
    with open(path_base + "_summary.txt", "w") as fh:
        fh.write(text)
    with open(path_base + "_summary.json", "w") as fh:
        json.dump(
            {
                "sigma_rff": sigma_rff,
                "solver_rel_err": rel_err,
                "results": {
                    str(n): [
                        {k: v for k, v in r.items() if k != "panel_a"}
                        for r in results[n]
                    ]
                    for n in train_sizes
                },
                "timing": timing,
                "appendix": appendix,
                "acceptance_ok": acc_ok,
            },
            fh,
            indent=1,
        )

    print(f"\ntotal wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
