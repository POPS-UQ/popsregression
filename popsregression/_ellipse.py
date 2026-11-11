"""
POPS ellipsoid fitting engine.

Misspecification-aware regression with a uniform-ellipsoid parameter
posterior, fit by direct minimization of the empirical generalization error
of the exact projected-ball pushforward, with an optional closed-form
empirical-Bayes (Laplace) hyperposterior layer ("Ellipse+EB").

This module holds the internal fitting engine. The public estimator is
:class:`~popsregression.POPSEllipseRegression`, whose ``regularization``
parameter selects the bare fit, the empirical-Bayes layer or the pilot-split
PAC-Bayes construction. The Laplace layer here is a *diagnostic*: its
whitening, baseline, center and hyperprior are all estimated from the one
sample it is evaluated on, so its ``bound_`` is not a PAC bound.

The objective is written once, as the per-datum loss
:func:`~popsregression._projected_ball.jx_ball_loss`; its gradient and the
diagonal curvature of the Laplace layer come from automatic differentiation,
and the whole continuation over the barrier schedule is one compiled kernel.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ._base import (
    BaseEstimator,
    check_is_fitted,
    check_option,
    check_random_state,
    check_sample_weight,
    check_scalar,
    check_X,
    check_X_y,
)
from ._optimize import lbfgs
from ._pops import POPSRegression
from ._projected_ball import ball_constants, jx_ball_curvature, jx_ball_loss


def _nll(center, U, Z, y, b0, weights, delta2, rho, k, log_c):
    """Weighted smooth-barrier negative log projected-ball likelihood.

    ``sum_i w_i l(r_i, v_i)`` with residuals ``r = y - Z @ center`` and
    squared half-widths ``v_i = b0_i + ||U^T z_i||^2 + delta**2``. All
    operations are ``O(n_samples * n_dim * rank)``; no dense ``(n_dim,
    n_dim)`` matrix is formed.
    """
    H = Z @ U
    v = b0 + jnp.sum(H * H, axis=1) + delta2
    return weights @ jx_ball_loss(y - Z @ center, v, rho, k, log_c)


@partial(jax.jit, static_argnames=("optimize_center",))
def _fit_ellipsoid(
    center, U, Z, y, b0, weights, delta2, rhos, k, log_c, tol, max_iter, optimize_center
):
    """Continuation L-BFGS over the barrier schedule ``rhos``.

    Returns the optimized center and factor, the objective at the last
    stage and the number of iterations of every stage.
    """
    n_dim, rank = U.shape
    if optimize_center:
        x0 = jnp.concatenate([center, U.ravel()])

        def split(x):
            return x[:n_dim], x[n_dim:].reshape(n_dim, rank)

    else:
        x0 = U.ravel()

        def split(x):
            return center, x.reshape(n_dim, rank)

    def stage(x, rho):
        def loss(x):
            return _nll(*split(x), Z, y, b0, weights, delta2, rho, k, log_c)

        x, value, n_iter = lbfgs(
            jax.value_and_grad(loss), x, max_iter=max_iter, tol=tol
        )
        return x, (value, n_iter)

    x, (values, n_iter) = jax.lax.scan(stage, x0, rhos)
    center, U = split(x)
    return center, U, values[-1], n_iter


@jax.jit
def _hess_diag(center, U, Z, y, b0, weights, delta2, rho, k, log_c):
    """Exact diagonal Hessian of :func:`_nll` in ``(center, U)``.

    The per-datum loss depends on the parameters only through ``(r_i,
    v_i)``, so the diagonal is its 2x2 curvature chained through the
    Jacobian, plus the ``dl/dv * d2v/dU2`` term.
    """
    H = Z @ U
    v = b0 + jnp.sum(H * H, axis=1) + delta2
    a_rr, a_vv, g_v = jx_ball_curvature(y - Z @ center, v, rho, k, log_c)
    Z2 = Z * Z
    hd_center = Z2.T @ (weights * a_rr)
    hd_U = 4.0 * (Z2.T @ ((weights * a_vv)[:, None] * (H * H)))
    hd_U = hd_U + 2.0 * (Z2.T @ (weights * g_v))[:, None]
    return hd_center, hd_U


@jax.jit
def _whitening(Xc, ridge, mode_threshold):
    """Symmetric whitening transform of the centered design and its inverse."""
    G = Xc.T @ Xc / Xc.shape[0] + ridge * jnp.eye(Xc.shape[1])
    evals, evecs = jnp.linalg.eigh(G)
    evals = jnp.maximum(evals, mode_threshold * evals.max())
    W = (evecs / jnp.sqrt(evals)) @ evecs.T
    return W, (evecs * jnp.sqrt(evals)) @ evecs.T


class _EllipsoidPosterior(BaseEstimator):
    """Uniform-ellipsoid POPS posterior (internal engine).

    This class implements the ellipsoid posterior and its Ellipse+EB layer.
    It is not part of the public API: use
    :class:`~popsregression.POPSEllipseRegression`, which drives one engine
    for ``regularization=None`` or ``'empirical-bayes'`` and one engine per
    pilot split for ``regularization='PAC'``.

    Fits a linear model whose parameter posterior is uniform on an
    ellipsoid, ``theta = mu + L z`` with ``z`` uniform on the unit ball and
    ``B = L L^T``, by directly minimizing the empirical generalization
    error of the exact scalar pushforward (a projected-ball density) at
    each training point [1]_. The negative log likelihood contains a
    log-barrier enforcing the POPS covering condition
    ``|y_i - phi_i mu| < sqrt(phi_i^T B phi_i + delta**2)``, so the
    optimization is an interior-point method for POPS coverage, extending
    :class:`POPSRegression` [2]_.

    The ellipsoid is parameterized in whitened feature coordinates as
    ``B_t = B0_t + U U^T`` with a fixed baseline ``B0_t`` and a low-rank
    factor ``U``; all fit operations are O(n_samples * n_features * rank).

    With ``pac_bayes=True`` a hierarchical empirical-Bayes layer
    ("Ellipse+EB") is added in closed form (no sampling anywhere): a
    diagonal Gaussian (Laplace) hyperposterior centered on the fitted
    optimum, with a Gaussian hyperprior centered there too. The fit itself
    is unchanged; the layer only broadens the predictive. Its ``bound_``
    has the form of a PAC-Bayes right side but is a **diagnostic, not a
    certificate**: the whitening, the POPS baseline, the frozen center and
    the hyperprior center are all estimated from the same sample the bound
    is evaluated on, ``empirical_H_`` is a second-order approximation and
    the smooth barrier is not the exact compact-support log loss. A PAC
    bound requires the pilot-split protocol of
    ``POPSEllipseRegression(regularization='PAC')``.

    Parameters
    ----------
    rank : int, default=32
        Rank of the ellipsoid update ``U U^T``. The effective rank is
        ``min(rank, n_dim)`` where ``n_dim`` is the whitened parameter
        dimension (``n_features``, plus one if ``fit_intercept=True``).

    delta : float, default=1e-3
        Radius of an output-offset coordinate. The posterior is uniform on
        the ellipsoid ``{(theta, a) : (theta - mu)^T B^{-1} (theta - mu)
        + a**2 / delta**2 <= 1}`` and the model output is ``x @ theta + a``,
        so ``delta**2`` is added to every squared pushforward width and the
        pushforward is a projected ball of dimension ``n_dim + 1``. It is a
        parameter coordinate, not observation noise. ``delta=0`` removes
        the coordinate and is valid only if ``rho_schedule`` does not
        approach 0 (the barrier diverges otherwise); a warning is raised if
        both are ~0.

    baseline : {'pops', 'ridge', 'zero'}, default='pops'
        Fixed baseline ``B0_t``:

        - ``'pops'``: whitened covariance of the POPS hypercube posterior
          from a :class:`POPSRegression` pre-fit (also used as warm start).
        - ``'ridge'``: ``baseline_ridge * I`` in whitened coordinates.
        - ``'zero'``: no baseline.

    baseline_ridge : float, default=1e-6
        Scale ``lam_B`` of the ``'ridge'`` baseline.

    whiten_ridge : float, default=1e-8
        Ridge ``lam_w`` added to the feature second-moment matrix
        ``G = Phi^T Phi / N + lam_w I`` before whitening.

    mode_threshold : float, default=1e-8
        Relative eigenvalue floor for the whitening transform: eigenvalues
        of ``G`` below ``mode_threshold * max_eigenvalue`` are clipped, as
        in :class:`POPSRegression`.

    rho_schedule : tuple of float, default=(1e-1, 1e-2, 1e-3, 1e-4)
        Continuation schedule for the smooth log-barrier; L-BFGS is run to
        convergence at each stage.

    tol : float, default=1e-8
        Convergence tolerance of each L-BFGS stage.

    max_iter : int, default=500
        Maximum L-BFGS iterations per continuation stage.

    fit_intercept : bool, default=False
        If True, features and targets are centered and an intercept
        coordinate is appended to the whitened design (after whitening;
        the intercept column is never whitened).

    weights : array-like of shape (n_samples,), default=None
        Non-negative per-datum weights ``w_i`` of the empirical
        generalization error ``G_hat = mean_i(w_i * ell_i)``.

    optimize_center : bool, default=False
        If False (default), the ellipsoid center is frozen at the warm
        start (the POPS pre-fit coefficients for ``baseline='pops'``, so
        ``coef_`` matches the familiar Bayesian ridge mean) and only the
        widths are optimized; with ``pac_bayes=True`` the hyperposterior is
        then over the width parameters only. If True, the center is
        optimized jointly with the widths; its stationarity condition is a
        heteroscedastic weighted least squares under the fitted widths, so
        ``coef_`` then deliberately differs from a least-squares mean and
        the fit is tighter but less conservative at small N.

    random_state : int, RandomState instance or None, default=None
        Seed of the small random initialization of ``U``. For
        reproducibility the default ``None`` behaves like 0, making fits
        deterministic.

    pac_bayes : bool, default=False
        Enable the closed-form Ellipse+EB layer: diagonal Laplace
        hyperposterior covariance, KL and diagnostic bound components, and
        analytic hyperposterior spread in prediction.

    hyperprior_scale : float, default=1.0
        Relative variance of the isotropic Gaussian hyperprior
        ``N(psi_0, tau2 * I)`` centered on the fitted optimum ``psi_0``:
        ``tau2 = hyperprior_scale * max(||psi_0||^2 / d, 1e-12)`` with
        ``d`` the parameter count, so the default is independent of the
        units of ``y``. ``np.inf`` is allowed; ``kl_`` and ``bound_`` are
        then infinite.

    hess_floor : float, default=1e-12
        Floor applied to (possibly negative) diagonal Hessian entries
        before inversion; a warning is raised if it activates on more than
        1% of coordinates.

    bound_xi : float, default=0.05
        Confidence parameter of the diagnostic ``bound_``.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Ellipsoid center ``mu`` in original feature coordinates.

    intercept_ : float
        Intercept; 0.0 if ``fit_intercept=False``.

    center_whitened_ : ndarray of shape (n_dim,)
        Ellipsoid center ``c_t`` in whitened coordinates.

    U_ : ndarray of shape (n_dim, rank_)
        Low-rank ellipsoid factor in whitened coordinates.

    ellipsoid_B_ : ndarray of shape (n_dim, n_dim)
        Dense ellipsoid shape matrix ``B`` in original coordinates
        (augmented with the intercept coordinate if
        ``fit_intercept=True``). Computed lazily on first access. Note
        that the posterior covariance is ``B / (d + 2)``, not ``B``, with
        ``d`` the ball dimension (``n_dim + 1`` when ``delta > 0``).

    baseline_B0_ : ndarray of shape (n_dim, n_dim)
        Dense fixed baseline ``B0_t`` in whitened coordinates. Computed
        lazily on first access.

    objective_ : float
        Final empirical generalization error ``G_hat(psi*)`` (mean
        weighted negative log predictive, at the last continuation stage).

    coverage_fraction_ : float
        Fraction of training points with ``q_i > 0`` (inside the
        pushforward support) at the optimum.

    n_iter_ : int
        Total number of L-BFGS iterations over all continuation stages.

    stage_iterations_ : ndarray of shape (n_stages,)
        L-BFGS iterations of each continuation stage.

    tau2_ : float
        Effective (absolute) hyperprior variance (only if
        ``pac_bayes=True``).

    hyper_sigma_diag_ : ndarray of shape (n_dim * (1 + rank_),)
        Diagonal of the Laplace hyperposterior covariance ``Sigma_H``
        (only if ``pac_bayes=True``). Zero on the center block when
        ``optimize_center=False``.

    kl_ : float
        ``KL(pi_H || pi_0H)`` in closed form (only if ``pac_bayes=True``).

    empirical_H_ : float
        Second-order (delta-method) approximation of the
        hyperposterior-averaged empirical error (only if
        ``pac_bayes=True``).

    bound_, diagnostic_bound_ : float
        Empirical-Bayes diagnostic ``empirical_H_ + kl_/N -
        log(bound_xi)/N`` (only if ``pac_bayes=True``). **Not** a
        certificate.

    gamma_ : float
        Effective number of well-determined parameters
        ``sum_j hd_j / (hd_j + 1/tau2)`` (only if ``pac_bayes=True``).

    n_features_in_ : int
        Number of features seen during ``fit``.

    See Also
    --------
    popsregression.POPSEllipseRegression : The public estimator.

    References
    ----------
    .. [1] Swinburne, T.D. et al. (2026). Hierarchical PAC-Bayes
           ellipsoid posteriors for misspecified surrogate models.
           NeurIPS Sim2Science workshop.

    .. [2] Swinburne, T.D. and Perez, D. (2025).
           "Parameter uncertainties for imperfect surrogate models in the
           low-noise regime."
           Machine Learning: Science and Technology, 6, 015008.
           :doi:`10.1088/2632-2153/ad9fce`
    """

    def __init__(
        self,
        *,
        rank=32,
        delta=1e-3,
        baseline="pops",
        baseline_ridge=1e-6,
        whiten_ridge=1e-8,
        mode_threshold=1e-8,
        rho_schedule=(1e-1, 1e-2, 1e-3, 1e-4),
        tol=1e-8,
        max_iter=500,
        fit_intercept=False,
        weights=None,
        optimize_center=False,
        random_state=None,
        pac_bayes=False,
        hyperprior_scale=1.0,
        hess_floor=1e-12,
        bound_xi=0.05,
    ):
        self.rank = rank
        self.delta = delta
        self.baseline = baseline
        self.baseline_ridge = baseline_ridge
        self.whiten_ridge = whiten_ridge
        self.mode_threshold = mode_threshold
        self.rho_schedule = rho_schedule
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.weights = weights
        self.optimize_center = optimize_center
        self.random_state = random_state
        self.pac_bayes = pac_bayes
        self.hyperprior_scale = hyperprior_scale
        self.hess_floor = hess_floor
        self.bound_xi = bound_xi

    def _validate(self):
        check_scalar("rank", self.rank, low=1)
        check_scalar("delta", self.delta, low=0.0)
        check_option("baseline", self.baseline, {"pops", "ridge", "zero"})
        check_scalar("baseline_ridge", self.baseline_ridge, low=0.0)
        check_scalar("whiten_ridge", self.whiten_ridge, low=0.0, low_open=True)
        check_scalar("mode_threshold", self.mode_threshold, low=0.0, low_open=True)
        check_scalar("tol", self.tol, low=0.0, low_open=True)
        check_scalar("max_iter", self.max_iter, low=1)
        check_scalar("hyperprior_scale", self.hyperprior_scale, low=0.0, low_open=True)
        check_scalar("hess_floor", self.hess_floor, low=0.0, low_open=True)
        check_scalar("bound_xi", self.bound_xi, 0.0, 1.0, low_open=True, high_open=True)
        rho_schedule = np.atleast_1d(np.asarray(self.rho_schedule, dtype=float))
        if rho_schedule.ndim != 1 or rho_schedule.size == 0:
            raise ValueError("rho_schedule must be a non-empty 1d sequence.")
        if np.any(rho_schedule <= 0):
            raise ValueError("rho_schedule entries must be positive.")
        if self.delta == 0.0 and rho_schedule.min() <= 1e-8:
            warnings.warn(
                (
                    "delta=0 with rho_schedule reaching ~0 can make the "
                    "log-barrier diverge on exactly-covered points; use a "
                    "positive delta or a larger final rho."
                ),
                UserWarning,
            )
        return rho_schedule

    def fit(self, X, y):
        """Fit the ellipsoid posterior by continuation L-BFGS.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        rho_schedule = self._validate()
        X, y = check_X_y(X, y)
        n_samples, self.n_features_in_ = X.shape
        sample_weight = check_sample_weight(self.weights, n_samples)
        rng = check_random_state(0 if self.random_state is None else self.random_state)

        # --- Preprocess: center, whiten, append intercept coordinate ---
        self._with_intercept = bool(self.fit_intercept)
        if self._with_intercept:
            self._x_offset = X.mean(axis=0)
            self._y_offset = float(y.mean())
        else:
            self._x_offset = np.zeros(X.shape[1])
            self._y_offset = 0.0
        Xc = X - self._x_offset
        yc = y - self._y_offset
        W, sqrt_G = _whitening(Xc, float(self.whiten_ridge), float(self.mode_threshold))
        self._whiten_W, self._sqrt_G = np.asarray(W), np.asarray(sqrt_G)

        _, Z = self._whitened_design(X)
        n_dim = Z.shape[1]
        self._ball_dim = n_dim
        # The width floor is an output-offset coordinate of radius delta:
        # the posterior is uniform on an ellipsoid in (theta, offset), so
        # the pushforward is a projected ball of dimension n_dim + 1.
        self._projection_dim = n_dim + int(self.delta > 0)
        self.rank_ = rank = min(int(self.rank), n_dim)

        # --- Baseline B0, warm start and optimization ---
        b0, center = self._setup_baseline(Xc, yc, Z, sample_weight)
        U0 = 1e-3 * rng.randn(n_dim * rank).reshape(n_dim, rank)
        constants = ball_constants(self._projection_dim)
        data = (jnp.asarray(Z), yc, b0, sample_weight, float(self.delta) ** 2)
        center, U, value, stage_iterations = _fit_ellipsoid(
            center,
            U0,
            *data,
            rho_schedule,
            *constants,
            float(self.tol),
            int(self.max_iter),
            optimize_center=bool(self.optimize_center),
        )

        # --- Recover fitted attributes ---
        self.center_whitened_ = c_t = np.asarray(center)
        self.U_ = np.asarray(U)
        self.stage_iterations_ = np.asarray(stage_iterations)
        self.n_iter_ = int(self.stage_iterations_.sum())
        if self._with_intercept:
            self.coef_ = self._whiten_W @ c_t[:-1]
            self.intercept_ = float(
                self._y_offset - self._x_offset @ self.coef_ + c_t[-1]
            )
        else:
            self.coef_ = self._whiten_W @ c_t
            self.intercept_ = 0.0
        self.objective_ = float(value) / n_samples
        resid = yc - Z @ c_t
        v = b0 + np.sum((Z @ self.U_) ** 2, axis=1) + self.delta**2
        self.coverage_fraction_ = float(np.mean(resid * resid < v))

        self._pac_bayes_fitted = bool(self.pac_bayes)
        if self.pac_bayes:
            hess = _hess_diag(center, U, *data, rho_schedule[-1], *constants)
            self._finalize_pac_bayes(hess, n_samples)
        return self

    def _setup_baseline(self, Xc, yc, Z, sample_weight):
        """Baseline squared widths ``b0_i`` and whitened warm-start center.

        For ``baseline='pops'`` the POPS hypercube covariance is expressed
        as a low-rank factor ``F`` in (centered) design coordinates, so
        that ``b0_i = ||F^T x_i||^2`` never requires a dense B0.
        """
        self._baseline_factor = None
        if self.baseline == "pops":
            design = Xc
            if self._with_intercept:
                design = np.hstack([Xc, np.ones((Xc.shape[0], 1))])
            pops = POPSRegression(fit_intercept=False)
            mu0 = pops._fit_hypercube(design, yc, sample_weight)
            support = pops._hypercube_support
            low, high = pops._hypercube_bounds
            mid = 0.5 * (low + high)
            std = (high - low) / np.sqrt(12.0)
            # Covariance (about zero) of the hypercube posterior:
            # S diag(var) S^T + (S mid)(S mid)^T = F F^T.
            self._baseline_factor = np.hstack([support * std, (support @ mid)[:, None]])
            if self._with_intercept:
                center = np.concatenate([self._sqrt_G @ mu0[:-1], mu0[-1:]])
            else:
                center = self._sqrt_G @ mu0
        else:
            # Whitened (approximate) least-squares center: Z has
            # near-identity second-moment matrix by construction.
            center = Z.T @ (sample_weight * yc) / sample_weight.sum()
        return self._baseline_widths(Xc, Z), np.asarray(center, dtype=np.float64)

    def _finalize_pac_bayes(self, hess, n_samples):
        """Closed-form Laplace hyperposterior, KL and diagnostic bound.

        The hyperprior is centered on the fitted optimum, so the fit is
        unchanged and the hyperposterior only broadens the predictive. It
        covers the optimized coordinates only: with
        ``optimize_center=False`` the frozen center block gets zero
        variance and is excluded from the KL and effective-dof sums.
        """
        n_dim = self._ball_dim
        hd = np.concatenate([np.asarray(hess[0]), np.asarray(hess[1]).ravel()])
        n_clipped = int(np.sum(hd < self.hess_floor))
        if n_clipped > 0.01 * hd.size:
            warnings.warn(
                (
                    f"Hessian diagonal floor activated on {n_clipped}/{hd.size} "
                    "coordinates; the Laplace covariance may be unreliable. "
                    "Consider a larger hyperprior_scale or hess_floor."
                ),
                UserWarning,
            )
        hd = np.maximum(hd, self.hess_floor)
        free = slice(None) if self.optimize_center else slice(n_dim, None)
        psi = np.concatenate([self.center_whitened_, self.U_.ravel()])
        tau2 = float(self.hyperprior_scale) * max(float(psi @ psi) / psi.size, 1e-12)
        sigma = np.zeros_like(hd)
        sigma[free] = 1.0 / (hd[free] + 1.0 / tau2)
        self.tau2_ = tau2
        self.hyper_sigma_diag_ = sigma
        self.gamma_ = float(np.sum(hd[free] * sigma[free]))
        self.kl_ = 0.5 * float(
            np.sum(sigma[free] / tau2 - 1.0 + np.log(tau2 / sigma[free]))
        )
        # Second-order (delta-method) hyperposterior average of G_hat.
        self.empirical_H_ = (
            self.objective_ + 0.5 * float(np.sum(sigma * hd)) / n_samples
        )
        self.bound_ = self.empirical_H_ + (self.kl_ - np.log(self.bound_xi)) / n_samples
        self.diagnostic_bound_ = self.bound_
        self.certificate_status_ = "diagnostic_empirical_bayes"
        self._sigma_c = sigma[:n_dim]
        self._sigma_U = sigma[n_dim:].reshape(n_dim, -1)

    def _whitened_design(self, X):
        """Map raw features to whitened coordinates (plus intercept)."""
        Xc = X - self._x_offset
        Z = Xc @ self._whiten_W
        if self._with_intercept:
            Z = np.hstack([Z, np.ones((Z.shape[0], 1))])
        return Xc, Z

    def _baseline_widths(self, Xc, Z):
        """Baseline squared widths ``b0(x) = z^T B0 z``."""
        if self.baseline == "pops":
            D = Xc
            if self._with_intercept:
                D = np.hstack([Xc, np.ones((Xc.shape[0], 1))])
            DF = D @ self._baseline_factor
            return np.einsum("ij,ij->i", DF, DF)
        if self.baseline == "ridge":
            return self.baseline_ridge * np.einsum("ij,ij->i", Z, Z)
        return np.zeros(Z.shape[0])

    def _squared_widths(self, Xc, Z):
        """Pushforward squared widths ``s(x) = z^T (B0 + U U^T) z``."""
        return np.sum((Z @ self.U_) ** 2, axis=1) + self._baseline_widths(Xc, Z)

    def predict(self, X, return_std=False, return_bounds=False, return_bound_std=False):
        """Predict using the ellipsoid pushforward.

        The pushforward of the ellipsoid posterior at ``x`` is a
        projected-ball density centered at ``x @ coef_ + intercept_`` with
        squared support half-width ``v = x^T B x + delta**2``. The
        returned standard deviation is the predictive standard deviation
        of that density, ``sqrt(v / (d + 2))`` (``d = n_dim + 1`` when
        ``delta > 0``) — note this is the pushforward standard deviation,
        NOT the support half-width. For a model fitted without
        ``pac_bayes`` the bounds are the support of the fitted ellipsoid,
        ``mean +/- sqrt(v)`` (the ellipse max/min).

        If the model was fitted with ``pac_bayes=True``, the analytic
        hyperposterior spread enters, with no sampling anywhere:

        - ``y_std`` averages over the hyperposterior: the mean variance
          gains ``z^2 @ Sigma_c`` and the expected squared width gains
          ``dv = sum_m z^2 @ Sigma_U[:, m]``, i.e.
          ``std = sqrt((v + dv) / (d + 2) + z^2 @ Sigma_c)``.
        - ``y_bound_std`` is the hyperposterior standard deviation of the
          support-bound curves, propagated about the hyperposterior mean
          squared width ``v_mixed = v + dv``:
          ``Var[y_max] = Var[y_min] = z^2 @ Sigma_c
          + Var[v] / (4 v_mixed)``, where the exact Gaussian quadratic
          moments give ``Var[v] = 4 sum_m (z @ U_m)^2 s_m
          + 2 sum_m s_m^2`` and ``s_m = z^2 @ Sigma_U[:, m]``. For a
          model fitted without ``pac_bayes`` it is identically zero.
        - the returned ``bounds`` are the max/min over the ensemble of
          ellipses within the 2-sigma range of the hyperposterior,
          ``mean +/- (sqrt(v_mixed) + 2 * y_bound_std)`` — strictly
          broader than the hyperposterior-averaged support.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        return_std : bool, default=False
            If True, also return the predictive standard deviation of the
            pushforward density.

        return_bounds : bool, default=False
            If True, also return the max and min predictions: the support
            of the fitted ellipsoid pushforward, widened to the 2-sigma
            hyperposterior ensemble if fitted with ``pac_bayes=True``.

        return_bound_std : bool, default=False
            If True, also return the hyperposterior standard deviation of
            the support bounds (zero unless fitted with
            ``pac_bayes=True``).

        Returns
        -------
        y_mean : ndarray of shape (n_samples,)
            Predicted mean values.

        y_std : ndarray of shape (n_samples,)
            Predictive standard deviation. Only returned if
            ``return_std=True``.

        y_max : ndarray of shape (n_samples,)
            Upper support bound. Only returned if ``return_bounds=True``.

        y_min : ndarray of shape (n_samples,)
            Lower support bound. Only returned if ``return_bounds=True``.

        y_bound_std : ndarray of shape (n_samples,)
            Hyperposterior standard deviation of the support bounds. Only
            returned if ``return_bound_std=True``.
        """
        check_is_fitted(self)
        Xc, Z = self._whitened_design(check_X(X, self.n_features_in_))
        y_mean = Z @ self.center_whitened_ + self._y_offset
        if not (return_std or return_bounds or return_bound_std):
            return y_mean

        v = self._squared_widths(Xc, Z) + self.delta**2
        v_mixed = v
        mean_var = 0.0
        bound_var = np.zeros(Z.shape[0])
        if self._pac_bayes_fitted:
            Z2 = Z * Z
            sigma_u_proj = Z2 @ self._sigma_U
            v_mixed = v + np.sum(sigma_u_proj, axis=1)
            mean_var = Z2 @ self._sigma_c
            var_v = 4.0 * np.sum(
                (Z @ self.U_) ** 2 * sigma_u_proj, axis=1
            ) + 2.0 * np.sum(sigma_u_proj**2, axis=1)
            bound_var = mean_var + var_v / (4.0 * v_mixed)

        result = [y_mean]
        if return_std:
            result.append(np.sqrt(v_mixed / (self._projection_dim + 2.0) + mean_var))
        if return_bounds:
            half_width = np.sqrt(v_mixed) + 2.0 * np.sqrt(bound_var)
            result.extend([y_mean + half_width, y_mean - half_width])
        if return_bound_std:
            result.append(np.sqrt(bound_var))
        return tuple(result)

    @property
    def baseline_B0_(self):
        """Dense fixed baseline ``B0_t`` in whitened coordinates (lazy)."""
        check_is_fitted(self)
        n_dim = self._ball_dim
        if self.baseline == "zero":
            return np.zeros((n_dim, n_dim))
        if self.baseline == "ridge":
            return self.baseline_ridge * np.eye(n_dim)
        F = self._baseline_factor
        if self._with_intercept:
            Ft = np.vstack([self._sqrt_G @ F[:-1], F[-1:]])
        else:
            Ft = self._sqrt_G @ F
        return Ft @ Ft.T

    @property
    def ellipsoid_B_(self):
        """Dense ellipsoid shape matrix ``B`` in original coordinates (lazy).

        If ``fit_intercept=True`` the matrix is augmented: the last
        row/column corresponds to the intercept coordinate of the
        (centered) affine design ``[x - x_mean, 1]``. The posterior
        covariance of the parameters is ``B / (d + 2)`` (``d = n_dim + 1``
        when ``delta > 0``).
        """
        check_is_fitted(self)
        if self._with_intercept:
            A = np.vstack([self._whiten_W @ self.U_[:-1], self.U_[-1:]])
        else:
            A = self._whiten_W @ self.U_
        B = A @ A.T
        if self.baseline == "pops":
            F = self._baseline_factor
            B = B + F @ F.T
        elif self.baseline == "ridge":
            W2 = self._whiten_W @ self._whiten_W
            if self._with_intercept:
                n_dim = self._ball_dim
                W2_aug = np.zeros((n_dim, n_dim))
                W2_aug[:-1, :-1] = W2
                W2_aug[-1, -1] = 1.0
                W2 = W2_aug
            B = B + self.baseline_ridge * W2
        return B

    def sample(self, n_samples, random_state=None):
        """Draw parameter samples from the ellipsoid posterior.

        Samples are an affine map of uniform unit-ball draws,
        ``theta = mu + L z`` with ``L L^T = B``; with ``delta > 0`` the ball
        has one extra (output-offset) coordinate, which is dropped here, so
        these are draws of the parameter marginal. Matching the orientation
        of ``POPSRegression.posterior_samples_`` (parameters in rows). If
        ``fit_intercept=True`` an intercept row is appended.

        Parameters
        ----------
        n_samples : int
            Number of posterior draws.

        random_state : int, RandomState instance or None, default=None
            Seed of the ball draws.

        Returns
        -------
        samples : ndarray of shape (n_features (+ 1), n_samples)
            Posterior parameter samples (NOT perturbations: the center is
            included).
        """
        check_is_fitted(self)
        rng = check_random_state(random_state)
        n_dim = self._ball_dim
        d = self._projection_dim
        evals, evecs = np.linalg.eigh(self.baseline_B0_ + self.U_ @ self.U_.T)
        L = evecs * np.sqrt(np.maximum(evals, 0.0))
        g = rng.randn(d, n_samples)
        g /= np.linalg.norm(g, axis=0, keepdims=True)
        radius = rng.uniform(size=n_samples) ** (1.0 / d)
        # The first n_dim coordinates of the (n_dim + 1)-ball draw are the
        # parameter marginal; the last one is the output offset.
        theta_t = self.center_whitened_[:, None] + L @ (g[:n_dim] * radius)
        if self._with_intercept:
            theta_f = self._whiten_W @ theta_t[:-1]
            theta_i = theta_t[-1] + self._y_offset - self._x_offset @ theta_f
            return np.vstack([theta_f, theta_i])
        return self._whiten_W @ theta_t
