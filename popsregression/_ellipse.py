"""
POPS ellipsoid regression.

Misspecification-aware regression with a uniform-ellipsoid parameter
posterior, fit by direct minimization of the empirical generalization error
of the exact projected-ball pushforward, with an optional closed-form
PAC-Bayes (Laplace) hyperposterior layer.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, Options, StrOptions
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from ._pops import POPSRegression
from ._projected_ball import log_norm_constant, smooth_log


def _unpack(psi, n_dim):
    """Split a packed parameter vector into center and low-rank factor."""
    c = psi[:n_dim]
    U = psi[n_dim:].reshape(n_dim, -1)
    return c, U


def _ellipse_nll(psi, Z, y, b0, weights, delta, rho, psi0=None, prior_precision=0.0):
    """Weighted negative log projected-ball likelihood and exact gradient.

    Objective ``L = sum_i w_i * (0.5*log(v_i) - log(C_P) - k*L_rho(q_i))``
    with ``v_i = b0_i + ||U^T z_i||^2 + delta**2`` and
    ``q_i = 1 - r_i**2 / v_i``, optionally plus the Gaussian hyperprior
    ridge ``0.5 * prior_precision * ||psi - psi0||^2``.

    All operations are O(n_samples * n_dim * rank); no dense
    (n_dim, n_dim) matrix is formed.

    Parameters
    ----------
    psi : ndarray of shape (n_dim * (1 + rank),)
        Packed parameters ``(c, vec(U))`` in whitened coordinates.

    Z : ndarray of shape (n_samples, n_dim)
        Whitened design matrix.

    y : ndarray of shape (n_samples,)
        Centered targets.

    b0 : ndarray of shape (n_samples,)
        Baseline squared widths ``z_i^T B0 z_i``.

    weights : ndarray of shape (n_samples,)
        Non-negative per-datum weights.

    delta : float
        Aleatoric width floor; ``delta**2`` is added to squared widths.

    rho : float
        Continuation threshold of the smooth log-barrier.

    psi0 : ndarray of shape (n_dim * (1 + rank),), default=None
        Hyperprior center. Only used when ``prior_precision > 0``.

    prior_precision : float, default=0.0
        Ridge strength ``1 / tau2`` of the Gaussian hyperprior.

    Returns
    -------
    value : float
        Objective value.

    grad : ndarray of shape (n_dim * (1 + rank),)
        Exact gradient with respect to ``psi``.
    """
    n_dim = Z.shape[1]
    c, U = _unpack(psi, n_dim)
    k = 0.5 * (n_dim - 1)

    resid = y - Z @ c
    H = Z @ U
    s = b0 + np.einsum("ij,ij->i", H, H)
    v = s + delta * delta
    q = 1.0 - resid * resid / v

    log_q, d_log_q, _ = smooth_log(q, rho)
    ell = 0.5 * np.log(v) - log_norm_constant(n_dim) - k * log_q
    value = float(weights @ ell)

    g_r = weights * (2.0 * k * resid * d_log_q / v)
    g_s = weights * (0.5 / v - k * resid * resid * d_log_q / (v * v))
    grad_c = -(Z.T @ g_r)
    grad_U = 2.0 * (Z.T @ (g_s[:, None] * H))
    grad = np.concatenate([grad_c, grad_U.ravel()])

    if prior_precision > 0.0:
        dpsi = psi - psi0
        value += 0.5 * prior_precision * float(dpsi @ dpsi)
        grad += prior_precision * dpsi
    return value, grad


def _ellipse_nll_hess_diag(psi, Z, y, b0, weights, delta, rho):
    """Exact diagonal of the Hessian of the weighted negative log likelihood.

    The per-datum loss depends on ``psi`` only through ``(r_i, s_i)``, so
    the diagonal is a 2x2 inner Hessian chained through the (diagonal-free)
    Jacobian, plus the ``g_s * d2s/dU^2`` curvature term. The hyperprior
    ridge is NOT included.

    Parameters
    ----------
    See :func:`_ellipse_nll`.

    Returns
    -------
    hess_diag : ndarray of shape (n_dim * (1 + rank),)
        Diagonal Hessian entries, packed like ``psi``.
    """
    n_dim = Z.shape[1]
    c, U = _unpack(psi, n_dim)
    k = 0.5 * (n_dim - 1)

    resid = y - Z @ c
    H = Z @ U
    s = b0 + np.einsum("ij,ij->i", H, H)
    v = s + delta * delta
    q = 1.0 - resid * resid / v

    _, d1, d2 = smooth_log(q, rho)
    r2 = resid * resid
    v2 = v * v
    # d2 l / dr2, d2 l / ds2 and dl / ds of the per-datum loss; on the
    # exact-log branch d1 = 1/q, d2 = -1/q**2 recover the closed forms.
    a_rr = 2.0 * k * d1 / v - 4.0 * k * r2 * d2 / v2
    a_ss = -0.5 / v2 + 2.0 * k * d1 * r2 / (v2 * v) - k * d2 * r2 * r2 / (v2 * v2)
    g_s = 0.5 / v - k * r2 * d1 / v2

    Z2 = Z * Z
    hd_c = Z2.T @ (weights * a_rr)
    hd_U = 4.0 * (Z2.T @ ((weights * a_ss)[:, None] * (H * H)))
    hd_U += 2.0 * (Z2.T @ (weights * g_s))[:, None]
    return np.concatenate([hd_c, hd_U.ravel()])


class POPSRegressionEllipse(RegressorMixin, BaseEstimator):
    """Misspecification-aware regression with a uniform-ellipsoid posterior.

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

    With ``pac_bayes=True`` a hierarchical PAC-Bayes layer is added in
    closed form (no sampling anywhere): the Catoni/Gibbs hyperposterior is
    followed via its Laplace approximation, a diagonal Gaussian whose mode
    is the ridge-regularized optimum and whose covariance is a
    ridge-regularized inverse Hessian diagonal. The PAC bound holds for
    all hyperposteriors simultaneously, so evaluating its right-hand side
    at the Laplace Gaussian gives a rigorous bound for that Gaussian: the
    Laplace step costs tightness, never validity.

    Parameters
    ----------
    rank : int, default=32
        Rank of the ellipsoid update ``U U^T``. The effective rank is
        ``min(rank, n_dim)`` where ``n_dim`` is the whitened parameter
        dimension (``n_features``, plus one if ``fit_intercept=True``).

    delta : float, default=1e-3
        Aleatoric width floor. ``delta**2`` is added to every squared
        pushforward width (not to ``B``). ``delta=0`` is valid only if
        ``rho_schedule`` does not approach 0 (the barrier diverges
        otherwise); a warning is raised if both are ~0.

    baseline : {'pops', 'ridge', 'zero'}, default='pops'
        Fixed baseline ``B0_t``:

        - ``'pops'``: whitened covariance of the POPS hypercube posterior
          from a :class:`POPSRegression` pre-fit (also used as warm start
          and hyperprior center).
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
        Convergence tolerance passed to each L-BFGS stage (and used for
        the ``update_hyperprior`` outer loop).

    max_iter : int, default=500
        Maximum L-BFGS iterations per continuation stage.

    fit_intercept : bool, default=False
        If True, features and targets are centered and an intercept
        coordinate is appended to the whitened design (after whitening;
        the intercept column is never whitened).

    weights : array-like of shape (n_samples,), default=None
        Non-negative per-datum weights ``w_i`` of the empirical
        generalization error ``G_hat = mean_i(w_i * ell_i)``.

    optimize_center : bool, default=True
        If True, the ellipsoid center is optimized jointly with the
        widths; its stationarity condition is a heteroscedastic weighted
        least squares under the fitted widths (weights ``1/(q_i v_i)``),
        so ``coef_`` deliberately differs from an OLS/BayesianRidge mean.
        If False, the center is frozen at the warm start (the POPS
        pre-fit coefficients for ``baseline='pops'``) and only the
        widths are optimized; with ``pac_bayes=True`` the hyperposterior
        is then over the width parameters only.

    random_state : int, RandomState instance or None, default=None
        Seed of the small random initialization of ``U``. For
        reproducibility the default ``None`` behaves like 0, making fits
        deterministic.

    pac_bayes : bool, default=False
        Enable the closed-form PAC-Bayes layer: ridge-regularized MAP,
        diagonal Laplace hyperposterior covariance, KL and bound
        components. ``pac_bayes=False`` is the ``tau2 -> inf`` limit and
        leaves phase-1 results unchanged.

    hyperprior_scale : float, default=1.0
        Relative variance of the isotropic Gaussian hyperprior
        ``N(psi_0, tau2 * I)`` centered on the POPS warm start. The
        effective variance is scaled to the warm start,
        ``tau2 = hyperprior_scale * max(||psi_0||^2 / d, 1e-12)`` with
        ``d`` the parameter count, so the default is independent of the
        units of ``y`` (an absolute ``tau2`` would over-shrink data with
        large targets). The effective value is stored in ``tau2_``.
        ``np.inf`` is allowed and recovers the ``pac_bayes=False``
        optimum exactly (the prior ridge vanishes); in that improper
        limit ``kl_`` and ``bound_`` are infinite.

    update_hyperprior : bool, default=False
        If True (and ``pac_bayes=True``), update ``tau2`` by a
        Tipping/MacKay evidence iteration with Gamma hyper-hyperprior,
        following the conventions of
        :class:`sklearn.linear_model.BayesianRidge`.

    hh_lambda_1 : float, default=1e-6
        Shape parameter of the Gamma hyper-hyperprior over ``1/tau2``.

    hh_lambda_2 : float, default=1e-6
        Rate parameter of the Gamma hyper-hyperprior over ``1/tau2``.

    n_outer : int, default=5
        Maximum number of outer evidence iterations when
        ``update_hyperprior=True``.

    hess_floor : float, default=1e-12
        Floor applied to (possibly negative) diagonal Hessian entries
        before inversion; a warning is raised if it activates on more than
        1% of coordinates.

    bound_xi : float, default=0.05
        Confidence parameter ``xi`` of the PAC bound: ``bound_`` holds
        with probability at least ``1 - xi``.

    subgamma_const : float, default=0.0
        Optional user-supplied sub-gamma (variance/tail) constant added to
        ``bound_``. The default 0 corresponds to the near-deterministic
        (bounded-loss) idealization; supply the appropriate constant for
        your noise model to make the bound fully rigorous.

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
        that the posterior covariance is ``B / (n_dim + 2)``, not ``B``.

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
        Total number of L-BFGS iterations over all continuation stages
        (and outer evidence iterations).

    n_outer_iter_ : int
        Number of outer evidence iterations performed (1 unless
        ``pac_bayes=True`` and ``update_hyperprior=True``).

    tau2_ : float
        Final effective (absolute) hyperprior variance, i.e.
        ``hyperprior_scale`` times the warm-start scale, after any
        evidence updates (only if ``pac_bayes=True``).

    hyper_sigma_diag_ : ndarray of shape (n_dim * (1 + rank_),)
        Diagonal of the Laplace hyperposterior covariance ``Sigma_H``
        (only if ``pac_bayes=True``). Zero on the center block when
        ``optimize_center=False``.

    kl_ : float
        ``KL(pi_H || pi_0H)`` in closed form (only if ``pac_bayes=True``).

    empirical_H_ : float
        Second-order estimate of the hyperposterior-averaged empirical
        error ``H[pi_H]`` (only if ``pac_bayes=True``).

    bound_ : float
        PAC-Bayes bound ``empirical_H_ + kl_/N - log(bound_xi)/N +
        subgamma_const`` (only if ``pac_bayes=True``).

    gamma_ : float
        Effective number of well-determined parameters
        ``sum_j hd_j / (hd_j + 1/tau2)`` (only if ``pac_bayes=True``).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    POPSRegression : Sampling-based POPS hypercube posterior.

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

    Examples
    --------
    >>> import numpy as np
    >>> from popsregression import POPSRegressionEllipse
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(30, 3)
    >>> y = X @ np.array([1.0, -1.0, 0.5]) + 0.1 * np.tanh(3 * X[:, 0])
    >>> model = POPSRegressionEllipse(random_state=0)
    >>> model.fit(X, y)
    POPSRegressionEllipse(random_state=0)
    >>> y_pred, y_std = model.predict(X[:2], return_std=True)
    """

    _parameter_constraints: dict = {
        "rank": [Interval(Integral, 1, None, closed="left")],
        "delta": [Interval(Real, 0, None, closed="left")],
        "baseline": [StrOptions({"pops", "ridge", "zero"})],
        "baseline_ridge": [Interval(Real, 0, None, closed="left")],
        "whiten_ridge": [Interval(Real, 0, None, closed="neither")],
        "mode_threshold": [Interval(Real, 0, None, closed="neither")],
        "rho_schedule": ["array-like"],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "fit_intercept": ["boolean"],
        "weights": ["array-like", None],
        "optimize_center": ["boolean"],
        "random_state": ["random_state"],
        "pac_bayes": ["boolean"],
        "hyperprior_scale": [
            Interval(Real, 0, None, closed="neither"),
            Options(Real, {np.inf}),
        ],
        "update_hyperprior": ["boolean"],
        "hh_lambda_1": [Interval(Real, 0, None, closed="left")],
        "hh_lambda_2": [Interval(Real, 0, None, closed="left")],
        "n_outer": [Interval(Integral, 1, None, closed="left")],
        "hess_floor": [Interval(Real, 0, None, closed="neither")],
        "bound_xi": [Interval(Real, 0, 1, closed="neither")],
        "subgamma_const": [Interval(Real, 0, None, closed="left")],
    }

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
        optimize_center=True,
        random_state=None,
        pac_bayes=False,
        hyperprior_scale=1.0,
        update_hyperprior=False,
        hh_lambda_1=1e-6,
        hh_lambda_2=1e-6,
        n_outer=5,
        hess_floor=1e-12,
        bound_xi=0.05,
        subgamma_const=0.0,
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
        self.update_hyperprior = update_hyperprior
        self.hh_lambda_1 = hh_lambda_1
        self.hh_lambda_2 = hh_lambda_2
        self.n_outer = n_outer
        self.hess_floor = hess_floor
        self.bound_xi = bound_xi
        self.subgamma_const = subgamma_const

    @_fit_context(prefer_skip_nested_validation=True)
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
        X, y = validate_data(self, X, y, dtype=[np.float64, np.float32], y_numeric=True)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples = X.shape[0]

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

        if self.weights is not None:
            sample_weight = _check_sample_weight(
                self.weights, X, dtype=X.dtype, ensure_non_negative=True
            )
        else:
            sample_weight = np.ones(n_samples)

        rng = check_random_state(0 if self.random_state is None else self.random_state)

        # --- Preprocess: center, whiten, append intercept coordinate ---
        if self.fit_intercept:
            self._x_offset = X.mean(axis=0)
            self._y_offset = float(y.mean())
        else:
            self._x_offset = np.zeros(X.shape[1])
            self._y_offset = 0.0
        self._with_intercept = bool(self.fit_intercept)
        Xc = X - self._x_offset
        yc = y - self._y_offset

        G = Xc.T @ Xc / n_samples
        G[np.diag_indices_from(G)] += self.whiten_ridge
        evals, evecs = eigh(G)
        evals = np.maximum(evals, self.mode_threshold * evals.max())
        self._whiten_W = (evecs / np.sqrt(evals)) @ evecs.T
        sqrt_G = (evecs * np.sqrt(evals)) @ evecs.T

        Z = Xc @ self._whiten_W
        if self._with_intercept:
            Z = np.hstack([Z, np.ones((n_samples, 1))])
        n_dim = Z.shape[1]
        self._ball_dim = n_dim
        rank = min(self.rank, n_dim)
        self.rank_ = rank

        if self._with_intercept:
            design = np.hstack([Xc, np.ones((n_samples, 1))])
        else:
            design = Xc

        # --- Baseline B0 and warm start ---
        b0, c_init = self._setup_baseline(design, yc, Z, sample_weight, sqrt_G)
        if self.baseline == "pops":
            self._sqrt_G = sqrt_G

        psi0 = np.concatenate([c_init, np.zeros(n_dim * rank)])
        psi = np.concatenate([c_init, 1e-3 * rng.randn(n_dim * rank)])
        # Coordinates the optimizer (and hyperposterior) act on: all of
        # psi, or the width block only when the center is frozen.
        free = slice(None) if self.optimize_center else slice(n_dim, None)

        def objective(psi_free, *args):
            psi_full = psi_free
            if not self.optimize_center:
                psi_full = np.concatenate([c_init, psi_free])
            value, grad = _ellipse_nll(psi_full, *args)
            return value, grad[free]

        # --- Continuation L-BFGS, optional evidence outer loop ---
        # The hyperprior variance is scaled to the warm start so that
        # hyperprior_scale is independent of the units of y.
        scale2 = max(float(psi0 @ psi0) / psi0.size, 1e-12)
        tau2 = float(self.hyperprior_scale) * scale2
        update_mode = self.pac_bayes and self.update_hyperprior
        n_outer = self.n_outer if update_mode else 1
        self.n_iter_ = 0
        self.n_outer_iter_ = 0
        for outer in range(n_outer):
            prec = 1.0 / tau2 if self.pac_bayes else 0.0
            for rho in rho_schedule:
                res = minimize(
                    objective,
                    psi[free],
                    args=(Z, yc, b0, sample_weight, self.delta, rho, psi0, prec),
                    method="L-BFGS-B",
                    jac=True,
                    tol=self.tol,
                    options={"maxiter": self.max_iter},
                )
                psi[free] = res.x
                self.n_iter_ += int(res.nit)
            self.n_outer_iter_ += 1
            if not update_mode or outer == n_outer - 1:
                break
            hd = self._clipped_hess_diag(
                psi, Z, yc, b0, sample_weight, rho_schedule[-1], warn=False
            )[free]
            gamma = float(np.sum(hd / (hd + 1.0 / tau2)))
            dpsi = psi - psi0
            tau2_new = (float(dpsi @ dpsi) + 2.0 * self.hh_lambda_2) / (
                gamma + 2.0 * self.hh_lambda_1
            )
            rel_change = abs(tau2_new - tau2) / max(tau2, np.finfo(float).tiny)
            tau2 = tau2_new
            if rel_change < self.tol:
                break

        # --- Recover fitted attributes ---
        c_t, U = _unpack(psi, n_dim)
        self.center_whitened_ = c_t
        self.U_ = U
        self._psi0 = psi0
        if self._with_intercept:
            self.coef_ = self._whiten_W @ c_t[:-1]
            self.intercept_ = float(
                self._y_offset - self._x_offset @ self.coef_ + c_t[-1]
            )
        else:
            self.coef_ = self._whiten_W @ c_t
            self.intercept_ = 0.0

        rho_final = rho_schedule[-1]
        value, _ = _ellipse_nll(psi, Z, yc, b0, sample_weight, self.delta, rho_final)
        self.objective_ = value / n_samples
        resid = yc - Z @ c_t
        v = b0 + np.sum((Z @ U) ** 2, axis=1) + self.delta**2
        self.coverage_fraction_ = float(np.mean(resid * resid < v))

        if self.pac_bayes:
            self._finalize_pac_bayes(
                psi, psi0, Z, yc, b0, sample_weight, rho_final, tau2, n_samples, free
            )
        self._pac_bayes_fitted = bool(self.pac_bayes)
        return self

    def _setup_baseline(self, design, yc, Z, sample_weight, sqrt_G):
        """Baseline squared widths ``b0_i`` and whitened warm-start center.

        For ``baseline='pops'`` the POPS hypercube covariance is expressed
        as a low-rank factor ``F`` in (centered) design coordinates, so
        that ``b0_i = ||F^T x_i||^2`` never requires a dense B0.
        """
        self._baseline_factor = None
        if self.baseline == "pops":
            pops = POPSRegression(fit_intercept=False)
            pops.fit(design, yc, sample_weight=sample_weight)
            support = pops._hypercube_support
            low, high = pops._hypercube_bounds
            mid = 0.5 * (low + high)
            std = (high - low) / np.sqrt(12.0)
            # Covariance (about zero) of the hypercube posterior:
            # S diag(var) S^T + (S mid)(S mid)^T = F F^T.
            F = np.hstack([support * std, (support @ mid)[:, None]])
            self._baseline_factor = F
            b0 = np.einsum("ij,ij->i", design @ F, design @ F)
            mu0 = pops.coef_
            if self._with_intercept:
                c_init = np.concatenate([sqrt_G @ mu0[:-1], mu0[-1:]])
            else:
                c_init = sqrt_G @ mu0
        else:
            if self.baseline == "ridge":
                b0 = self.baseline_ridge * np.einsum("ij,ij->i", Z, Z)
            else:
                b0 = np.zeros(Z.shape[0])
            # Whitened (approximate) ridge/OLS center: Z has near-identity
            # second-moment matrix by construction.
            c_init = Z.T @ (sample_weight * yc) / sample_weight.sum()
        return b0, np.asarray(c_init, dtype=np.float64)

    def _clipped_hess_diag(self, psi, Z, yc, b0, sample_weight, rho, warn=True):
        """Diagonal Hessian of the unpenalized objective, floored."""
        hd = _ellipse_nll_hess_diag(psi, Z, yc, b0, sample_weight, self.delta, rho)
        n_clipped = int(np.sum(hd < self.hess_floor))
        if warn and n_clipped > 0.01 * hd.size:
            warnings.warn(
                (
                    f"Hessian diagonal floor activated on {n_clipped}/{hd.size} "
                    "coordinates; the Laplace covariance may be unreliable. "
                    "Consider a larger hyperprior_scale or hess_floor."
                ),
                UserWarning,
            )
        return np.maximum(hd, self.hess_floor)

    def _finalize_pac_bayes(
        self, psi, psi0, Z, yc, b0, sample_weight, rho, tau2, n_samples, free
    ):
        """Closed-form Laplace hyperposterior, KL and PAC bound components.

        The hyperposterior covers the optimized coordinates only: with
        ``optimize_center=False`` the frozen center block gets zero
        variance and is excluded from the KL and effective-dof sums.
        """
        n_dim = Z.shape[1]
        hd = self._clipped_hess_diag(psi, Z, yc, b0, sample_weight, rho)
        sigma = np.zeros_like(hd)
        sigma[free] = 1.0 / (hd[free] + 1.0 / tau2)
        dpsi = psi - psi0
        self.tau2_ = float(tau2)
        self.hyper_sigma_diag_ = sigma
        self.gamma_ = float(np.sum(hd[free] * sigma[free]))
        self.kl_ = 0.5 * float(
            np.sum(
                sigma[free] / tau2
                + dpsi[free] * dpsi[free] / tau2
                - 1.0
                + np.log(tau2 / sigma[free])
            )
        )
        # Second-order (delta-method) hyperposterior average of G_hat.
        self.empirical_H_ = (
            self.objective_ + 0.5 * float(np.sum(sigma * hd)) / n_samples
        )
        self.bound_ = (
            self.empirical_H_
            + self.kl_ / n_samples
            - np.log(self.bound_xi) / n_samples
            + self.subgamma_const
        )
        self._sigma_c = sigma[:n_dim]
        self._sigma_U = sigma[n_dim:].reshape(n_dim, -1)

    def _whitened_design(self, X):
        """Map raw features to whitened coordinates (plus intercept)."""
        Xc = X - self._x_offset
        Z = Xc @ self._whiten_W
        if self._with_intercept:
            Z = np.hstack([Z, np.ones((Z.shape[0], 1))])
        return Xc, Z

    def _squared_widths(self, Xc, Z):
        """Pushforward squared widths ``s(x) = z^T (B0 + U U^T) z``."""
        s = np.sum((Z @ self.U_) ** 2, axis=1)
        if self.baseline == "pops":
            D = Xc
            if self._with_intercept:
                D = np.hstack([Xc, np.ones((Xc.shape[0], 1))])
            DF = D @ self._baseline_factor
            s = s + np.einsum("ij,ij->i", DF, DF)
        elif self.baseline == "ridge":
            s = s + self.baseline_ridge * np.einsum("ij,ij->i", Z, Z)
        return s

    def predict(self, X, return_std=False, return_bounds=False):
        """Predict using the ellipsoid pushforward.

        The pushforward of the ellipsoid posterior at ``x`` is a
        projected-ball density centered at ``x @ coef_ + intercept_`` with
        squared support half-width ``v = x^T B x + delta**2``. The
        returned standard deviation is the predictive standard deviation
        of that density, ``sqrt(v / (n_dim + 2))`` — note this is the
        pushforward standard deviation, NOT the support half-width; the
        bounds are the support, ``mean +/- sqrt(v)``.

        If the model was fitted with ``pac_bayes=True``, the analytic
        (delta-method) hyperposterior spread is added: the mean variance
        gains ``z^T diag(Sigma_c) z`` and the expected squared width gains
        ``sum_m z^2 @ Sigma_U[:, m]``, i.e.
        ``std = sqrt((v + dv) / (n_dim + 2) + z^2 @ Sigma_c)`` and
        ``bounds = mean +/- sqrt(v + dv)`` with
        ``dv = sum_m z^2 @ Sigma_U[:, m]``. No sampling is used.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        return_std : bool, default=False
            If True, also return the predictive standard deviation of the
            pushforward density.

        return_bounds : bool, default=False
            If True, also return the max and min predictions (support of
            the pushforward density).

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
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)
        Xc, Z = self._whitened_design(np.asarray(X, dtype=np.float64))
        y_mean = Z @ self.center_whitened_ + self._y_offset
        if not (return_std or return_bounds):
            return y_mean

        v = self._squared_widths(Xc, Z) + self.delta**2
        mean_var = 0.0
        if self._pac_bayes_fitted:
            Z2 = Z * Z
            v = v + np.sum(Z2 @ self._sigma_U, axis=1)
            mean_var = Z2 @ self._sigma_c

        result = [y_mean]
        if return_std:
            result.append(np.sqrt(v / (self._ball_dim + 2.0) + mean_var))
        if return_bounds:
            half_width = np.sqrt(v)
            result.extend([y_mean + half_width, y_mean - half_width])
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
        covariance of the parameters is ``B / (n_dim + 2)``.
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
        ``theta = mu + L z`` with ``L L^T = B``, matching the orientation
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
        B_t = self.baseline_B0_ + self.U_ @ self.U_.T
        evals, evecs = eigh(B_t)
        L = evecs * np.sqrt(np.maximum(evals, 0.0))
        g = rng.randn(n_dim, n_samples)
        g /= np.linalg.norm(g, axis=0, keepdims=True)
        radius = rng.uniform(size=n_samples) ** (1.0 / n_dim)
        theta_t = self.center_whitened_[:, None] + L @ (g * radius)
        if self._with_intercept:
            theta_f = self._whiten_W @ theta_t[:-1]
            theta_i = theta_t[-1] + self._y_offset - self._x_offset @ theta_f
            return np.vstack([theta_f, theta_i])
        return self._whiten_W @ theta_t
