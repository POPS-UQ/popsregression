"""
POPS ellipse regression.

Misspecification-aware linear regression whose parameter posterior is uniform
on an ellipsoid. The ellipsoid is fitted by direct minimization of the
empirical zero-noise generalization error of its exact projected-ball
pushforward, and can be regularized by a hierarchical layer over the ellipsoid
hyperparameters ``Psi = (center, low-rank shape)``:

- ``regularization=None``: the bare fitted ellipsoid.
- ``regularization='empirical-bayes'``: the one-sample Laplace layer
  (Ellipse+EB). Its prior is centered on the fitted optimum, so its bound is a
  diagnostic only.
- ``regularization='PAC'``: the hyperparameter-level PAC-Bayes construction.
  A pilot split fixes any preprocessing, the whitening and a pilot
  ellipsoid; the hyperparameters are its log semi-axis lengths (and
  optionally center shifts), the hyperposterior is a Gaussian (Laplace
  approximation to the Gibbs hyperposterior) fitted on the independent
  certification split, and every term of the bound is evaluated exactly or
  bounded with a stated failure probability. The returned predictive is a
  finite mixture over stored hyperparameter draws, and the reported bound
  applies to that stored mixture.

The width floor ``delta`` is a parameter coordinate: the posterior is uniform
on an ellipsoid in ``(theta, a)``, where ``a`` is an output offset with
``|a| <= delta``. Scores, intervals and propagated draws therefore all come
from one parameter distribution. No observation noise enters anywhere.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import betainc, gammaln, logsumexp, xlogy
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context, clone
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import HasMethods, Interval, Options, StrOptions
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from ._ellipse import DiagnosticBoundWarning, _EllipsoidPosterior
from ._projected_ball import (
    floor_contaminated_loss_bounds,
    log_norm_constant,
    projected_ball_logpdf,
    smooth_log,
)


@dataclass(frozen=True)
class PACFoldBound:
    """PAC-Bayes right side of one pilot/certification fold.

    All terms refer to the floor-contaminated log loss of the fold's
    hierarchical predictive (App. ``sec:certificate`` of the paper).

    Attributes
    ----------
    lam : float
        Selected temperature ``lambda`` from the predeclared grid.
    n_units : int
        Number of independent certification units ``N_1`` (for grouped
        data, groups, not rows).
    empirical : float
        Monte Carlo estimate of ``E_{pi_H} G_hat(Psi)``.
    monte_carlo : float
        Empirical-Bernstein upper deviation of that estimate.
    kl : float
        Exact ``KL(pi_H || pi_0H)`` of the Gaussian hyperposterior.
    complexity : float
        ``'linear'``: ``(kl + log(|Lambda| |T| / xi)) / lam``. ``'kl'``:
        the whole gap ``raw - empirical - monte_carlo``.
    concentration : float
        ``'linear'``: Hoeffding term ``lam * R**2 / (8 * N_1)``; ``'kl'``:
        zero (the moment term is inside ``complexity``).
    raw : float
        Bound on ``E_{pi_H} L(Psi)``, the hyperposterior-averaged population
        loss, which also bounds the loss of the continuous mixture over the
        Gaussian hyperposterior (Jensen).
    mixture_empirical : float
        Empirical floor-contaminated log loss of the fold's mixture
        predictive; the gap to ``empirical`` is the remaining Jensen gap.
    outside_support_fraction : float
        Fraction of (draw, observation) pairs outside the compact support,
        where the loss is supplied by the floor.
    prior_std : float
        Selected hyperprior standard deviation of the log axis multipliers.
    inequality : str
        ``'linear'`` (Theorem 1 with the Hoeffding moment bound, the
        paper's form) or ``'kl'`` (Seeger--Maurer PAC-Bayes-kl with the
        exact Maurer moment constant ``xi(N_1)``).
    moment_constant : float
        ``'kl'``: ``xi(N_1) = E exp(N_1 kl(G_hat || G))`` for the worst
        (Bernoulli) loss, computed exactly (``<= 2 sqrt(N_1)``).
        ``'linear'``: 1.
    stored_mixture : float
        Bound on the population loss of the stored finite mixture of
        ``n_stored`` hyperparameter draws that ``predict*`` uses: the loss
        of the mixture is at most the draw average of ``L(Psi_k)``
        (Jensen), which exceeds its mean ``E_{pi_H} L`` by at most a
        Chernoff--kl deviation with its own failure probability.
    n_stored : int
        Number of stored hyperparameter draws of this fold.
    """

    lam: float
    n_units: int
    empirical: float
    monte_carlo: float
    kl: float
    complexity: float
    concentration: float
    raw: float
    mixture_empirical: float
    outside_support_fraction: float
    prior_std: float = 1.0
    inequality: str = "linear"
    moment_constant: float = 1.0
    stored_mixture: float = float("nan")
    n_stored: int = 0


@dataclass(frozen=True)
class PACCertificate:
    """Hyperparameter PAC-Bayes bound on the floor-contaminated log risk.

    With probability at least ``1 - failure_probability`` over the draw of
    the training sample and of the Monte Carlo draws (the
    pilot/certification split being independent of the data values), the
    population log risk of the floor-contaminated predictive
    ``(1 - beta) p(y | x) + beta / R_y`` on ``y_bounds`` is at most
    ``raw_bound``, where ``p`` is the stored finite mixture that
    ``predict``, ``predict_interval``, ``predict_logpdf`` and
    ``predict_cdf`` use. The bound says nothing directly about interval
    coverage.

    Attributes
    ----------
    raw_bound : float
        Bound for the returned (stored finite-mixture) predictive: the
        fold average of ``PACFoldBound.stored_mixture``.
    continuous_bound : float
        Fold average of ``PACFoldBound.raw``: the bound for the continuous
        mixture over each Gaussian hyperposterior, which ``predict*`` only
        approximates.
    trivial_bound : float
        ``log(R_y / beta)``, the loss ceiling; the bound is non-vacuous iff
        ``raw_bound < trivial_bound``.
    loss_lower, loss_upper : float
        Analytic range ``[a_beta, b_beta]`` of the per-unit loss.
    failure_probability : float
        Total failure probability: ``pac_failure_probability +
        mc_failure_probability + mixture_failure_probability``.
    pac_failure_probability, mc_failure_probability, mixture_failure_probability : float
        Budgets of the PAC-Bayes inequality, of the Monte Carlo estimate of
        the hyperposterior-averaged empirical loss, and of the step from the
        continuous to the stored finite mixture (each split evenly over
        folds).
    n_units : int
        Total number of independent units (groups) in the training sample.
    folds : tuple of PACFoldBound
        Per-fold decomposition.
    target : str
        The certified quantity.
    assumptions : tuple of str
        Conditions the caller asserts and the software cannot verify.
    """

    raw_bound: float
    trivial_bound: float
    loss_lower: float
    loss_upper: float
    failure_probability: float
    beta: float
    y_bounds: tuple
    min_half_width: float
    folds: tuple = field(default_factory=tuple)
    continuous_bound: float = float("nan")
    pac_failure_probability: float = float("nan")
    mc_failure_probability: float = float("nan")
    mixture_failure_probability: float = float("nan")
    n_units: int = 0
    target: str = "floor_contaminated_log_risk_of_stored_mixture"
    assumptions: tuple = (
        "training units are i.i.d. draws from the deployment law",
        (
            "y_bounds contains the population output support (declared by the "
            "user; the software cannot verify it)"
        ),
        (
            "any preprocessor depends on the pilot units only (enforced: it is "
            "cloned and fitted on each pilot split)"
        ),
    )

    @property
    def is_nonvacuous(self):
        """True iff the raw bound is below the trivial loss ceiling."""
        return bool(self.raw_bound < self.trivial_bound)

    @property
    def capped_bound(self):
        """``min(raw_bound, trivial_bound)``."""
        return float(min(self.raw_bound, self.trivial_bound))


def _to_original(engine, theta_t):
    """Whitened parameter vectors (columns) to original coordinates."""
    if engine._with_intercept:
        theta_f = engine._whiten_W @ theta_t[:-1]
        theta_i = theta_t[-1] + engine._y_offset - engine._x_offset @ theta_f
        return np.vstack([theta_f, theta_i])
    return engine._whiten_W @ theta_t


def _center_coefficients(engine, C):
    """Coefficients and intercepts of whitened centers ``C`` (one per row)."""
    if engine._with_intercept:
        coef = C[:, :-1] @ engine._whiten_W.T
        return coef, C[:, -1] + engine._y_offset - coef @ engine._x_offset
    return C @ engine._whiten_W.T, np.zeros(C.shape[0])


def _ball(rng, n_dim, n):
    g = rng.randn(n_dim, n)
    return g * (rng.uniform(size=n) ** (1.0 / n_dim) / np.linalg.norm(g, axis=0))


def _maurer_xi(n):
    """``xi(n) = sum_k C(n, k) (k/n)^k (1 - k/n)^(n - k)``, exactly.

    ``E exp(n kl(mean || mu)) <= xi(n)`` for ``n`` i.i.d. variables in
    ``[0, 1]`` with mean ``mu`` (Maurer 2004; the Bernoulli case is the
    worst and gives equality, whatever ``mu``). ``sqrt(n) <= xi(n) <= 2
    sqrt(n)``, with ``xi(1) = 2``.
    """
    n = int(n)
    k = np.arange(n + 1, dtype=float)
    log_terms = (
        gammaln(n + 1.0)
        - gammaln(k + 1.0)
        - gammaln(n - k + 1.0)
        + xlogy(k, k / n)
        + xlogy(n - k, 1.0 - k / n)
    )
    return float(np.exp(logsumexp(log_terms)))


class _Kernel:
    """Shared parts of the frozen and axis kernels.

    ``n_dim`` is the number of whitened parameter coordinates and
    ``ball_dim = n_dim + 1`` when the output-offset coordinate (radius
    ``delta``) is present: the pushforward of every input is then a
    projected ball of dimension ``ball_dim`` with squared half-width
    ``x^T B x + delta**2``.
    """

    def _setup(self, engine, preprocessor):
        self.engine = engine
        self.preprocessor = preprocessor
        self.n_dim = int(engine._ball_dim)
        self.delta = float(engine.delta)
        self.ball_dim = self.n_dim + int(self.delta > 0)

    def whitened(self, X):
        """Preprocess (if a fitted preprocessor is attached) and whiten."""
        if self.preprocessor is not None:
            X = np.asarray(self.preprocessor.transform(X), dtype=np.float64)
        return self.engine._whitened_design(X)

    def _split_ball(self, rng, n):
        """Unit-ball draws: parameter coordinates and scaled offsets."""
        u = _ball(rng, self.ball_dim, n)
        offset = self.delta * u[self.n_dim] if self.delta > 0 else np.zeros(n)
        return u[: self.n_dim], offset

    def sample_parameters(self, psi, rng):
        """One parameter vector (original coordinates) and offset per row."""
        theta_t, offset = self.sample_whitened(psi, rng)
        return _to_original(self.engine, theta_t), offset

    def sample_outputs(self, X, psi, rng):
        """Joint output draws ``x @ theta + a`` at ``X``, one column per row."""
        _, Z = self.whitened(X)
        theta_t, offset = self.sample_whitened(psi, rng)
        return Z @ theta_t + self.engine._y_offset + offset[None, :]


class _FrozenKernel(_Kernel):
    """Fitted ellipsoid with hyperparameters ``psi = (center, vec U)``.

    Wraps a fitted :class:`_EllipsoidPosterior`, whose centering, whitening,
    baseline and width floor are then held fixed. Used by the bare and
    empirical-Bayes fits, whose Laplace layer acts on ``(center, U)``.
    """

    def __init__(self, engine, preprocessor=None):
        self._setup(engine, preprocessor)
        self.rank = int(engine.rank_)
        self.psi_hat = np.concatenate([engine.center_whitened_, engine.U_.ravel()])

    def design(self, X):
        """Whitened design and baseline squared widths."""
        Xc, Z = self.whitened(X)
        return Z, self.engine._baseline_widths(Xc, Z)

    def pushforward(self, design, psi):
        """Pushforward means and support half-widths, shape (m, n)."""
        Z, b0 = design
        n_dim = self.n_dim
        C = psi[:, :n_dim]
        Us = psi[:, n_dim:].reshape(psi.shape[0], n_dim, self.rank)
        mean = C @ Z.T + self.engine._y_offset
        H = np.einsum("nd,mdr->mnr", Z, Us)
        s = b0[None, :] + np.einsum("mnr,mnr->mn", H, H) + self.delta**2
        return mean, np.sqrt(s)

    def coefficients(self, psi):
        return _center_coefficients(self.engine, psi[:, : self.n_dim])

    def sample_whitened(self, psi, rng):
        """One uniform draw of ``(theta, a)`` per hyperparameter row."""
        n_dim = self.n_dim
        B0 = self.engine.baseline_B0_
        out = np.empty((n_dim, psi.shape[0]))
        u, offset = self._split_ball(rng, psi.shape[0])
        for j, row in enumerate(psi):
            U = row[n_dim:].reshape(n_dim, self.rank)
            evals, evecs = eigh(B0 + U @ U.T)
            L = evecs * np.sqrt(np.maximum(evals, 0.0))
            out[:, j] = row[:n_dim] + L @ u[:, j]
        return out, offset


class _AxisKernel(_Kernel):
    """Pilot ellipsoid with uncertain principal-axis lengths (and center).

    With the pilot shape ``B_pilot = V diag(lam0) V^T`` and center ``c0`` in
    whitened coordinates, the hyperparameters ``psi = (eta, omega)`` in
    ``R^{2 P}`` define

    - center ``c(psi) = c0 + V (sqrt(lam0) * eta)``, a shift measured in
      pilot semi-axes, and
    - shape ``B(psi) = V diag(lam0 * exp(2 omega)) V^T``, i.e. every pilot
      semi-axis rescaled by ``exp(omega_k)``.

    The hyperparameter dimension is ``2 P`` (``P`` when the center is
    frozen) rather than the ``P (1 + r)`` of ``(center, U)``, so the KL cost
    of a concentrated hyperposterior grows with ``P`` only.
    """

    def __init__(self, engine, preprocessor=None, rel_floor=1e-8):
        self._setup(engine, preprocessor)
        B = engine.baseline_B0_ + engine.U_ @ engine.U_.T
        lam0, V = eigh(B)
        self.lam0 = np.maximum(lam0, rel_floor * max(lam0.max(), np.finfo(float).tiny))
        self.V = V
        self.c0 = engine.center_whitened_.copy()
        self.psi_hat = np.zeros(2 * self.n_dim)

    def design(self, X):
        """Pilot mean and the axis coordinates of each input.

        Returns ``(m0, A, E)`` with ``A = Z V sqrt(lam0)`` (center-shift
        sensitivities) and ``E = (Z V)**2 lam0`` (squared widths per axis),
        computed once and reused by every objective evaluation.
        """
        _, Z = self.whitened(X)
        W = Z @ self.V
        A = W * np.sqrt(self.lam0)
        return Z @ self.c0 + self.engine._y_offset, A, A * A

    def pushforward(self, design, psi):
        m0, A, E = design
        n = self.n_dim
        mean = m0[None, :] + psi[:, :n] @ A.T
        s = np.exp(np.clip(2.0 * psi[:, n:], -60.0, 60.0)) @ E.T
        return mean, np.sqrt(s + self.delta**2)

    def centers(self, psi):
        return self.c0[None, :] + (psi[:, : self.n_dim] * np.sqrt(self.lam0)) @ self.V.T

    def coefficients(self, psi):
        return _center_coefficients(self.engine, self.centers(psi))

    def sample_whitened(self, psi, rng):
        n = self.n_dim
        axes = np.sqrt(self.lam0[None, :] * np.exp(2.0 * psi[:, n:]))
        u, offset = self._split_ball(rng, psi.shape[0])
        return self.centers(psi).T + self.V @ (axes.T * u), offset

    def objective(self, psi, design, y, weights, rho, prior_precision):
        """Weighted smooth-barrier loss plus Gaussian prior, and gradient."""
        m0, A, E = design
        n = self.n_dim
        k = 0.5 * (self.ball_dim - 1)
        e2 = np.exp(np.clip(2.0 * psi[n:], -60.0, 60.0))
        r = y - m0 - A @ psi[:n]
        v = E @ e2 + self.delta**2
        q = 1.0 - r * r / v
        log_q, d1, _ = smooth_log(q, rho)
        ell = 0.5 * np.log(v) - log_norm_constant(self.ball_dim) - k * log_q
        g_r = weights * (2.0 * k * r * d1 / v)
        g_v = weights * (0.5 / v - k * r * r * d1 / (v * v))
        grad = np.concatenate([-(A.T @ g_r), 2.0 * e2 * (E.T @ g_v)])
        value = float(weights @ ell) + 0.5 * float(prior_precision @ (psi * psi))
        return value, grad + prior_precision * psi

    def hess_diag(self, psi, design, y, weights, rho):
        """Exact diagonal Hessian of the weighted loss (prior excluded)."""
        m0, A, E = design
        n = self.n_dim
        k = 0.5 * (self.ball_dim - 1)
        e2 = np.exp(np.clip(2.0 * psi[n:], -60.0, 60.0))
        r = y - m0 - A @ psi[:n]
        v = E @ e2 + self.delta**2
        q = 1.0 - r * r / v
        _, d1, d2 = smooth_log(q, rho)
        r2, v2 = r * r, v * v
        a_rr = 2.0 * k * d1 / v - 4.0 * k * r2 * d2 / v2
        a_vv = -0.5 / v2 + 2.0 * k * d1 * r2 / (v2 * v) - k * d2 * r2 * r2 / (v2 * v2)
        g_v = 0.5 / v - k * r2 * d1 / v2
        D = 2.0 * E * e2  # dv / domega
        hd_eta = (A * A).T @ (weights * a_rr)
        hd_omega = (D * D).T @ (weights * a_vv) + 2.0 * D.T @ (weights * g_v)
        return np.concatenate([hd_eta, hd_omega])


def _bernoulli_kl(q, p):
    """``kl(q || p)`` of Bernoulli laws, with ``0 log 0 = 0``."""
    if p <= 0.0 or p >= 1.0:
        return 0.0 if p == q else np.inf
    out = 0.0
    if q > 0.0:
        out += q * np.log(q / p)
    if q < 1.0:
        out += (1.0 - q) * np.log((1.0 - q) / (1.0 - p))
    return out


def _kl_inverse(q, budget, n_bisect=100):
    """``sup {p in [q, 1] : kl(q || p) <= budget}`` (upper bisection value)."""
    q = float(np.clip(q, 0.0, 1.0))
    lo, hi = q, 1.0
    for _ in range(n_bisect):
        p = 0.5 * (lo + hi)
        if _bernoulli_kl(q, p) > budget:
            hi = p
        else:
            lo = p
    return hi


def _empirical_bernstein(values, value_range, delta):
    """One-sided empirical Bernstein deviation (Maurer and Pontil 2009, Thm 4)."""
    n = values.size
    log_term = np.log(2.0 / delta)
    var = float(np.var(values, ddof=1)) if n > 1 else value_range**2
    return float(
        np.sqrt(2.0 * var * log_term / n)
        + 7.0 * value_range * log_term / (3.0 * max(n - 1, 1))
    )


class POPSEllipseRegression(RegressorMixin, BaseEstimator):
    """Uniform-ellipsoid POPS regression with optional hierarchical regularization.

    The parameter posterior is uniform on an ellipsoid
    ``{(theta, a) : (theta - mu)^T B^{-1} (theta - mu) + a**2 / delta**2 <= 1}``
    over the coefficients ``theta`` and an output offset ``a`` (the width
    floor, ``|a| <= delta``); the model output is ``x @ theta + a``. Its
    pushforward at ``x`` is a projected-ball density of dimension ``P + 1``
    with mean ``x @ mu`` and support half-width ``sqrt(x^T B x +
    delta**2)``, so the zero-noise empirical generalization error is
    analytic. The ellipsoid is
    fitted by minimizing it, starting from a :class:`POPSRegression`
    hypercube [1]_.

    A hierarchical layer over the ellipsoid hyperparameters
    ``Psi = (center, U)``, with ``B = B0 + U U^T`` in whitened coordinates,
    supplies a finite-data correction. The predictive is then a finite
    mixture of pushforwards over stored draws of ``Psi``; all predictions,
    intervals, densities and parameter draws refer to that one parameter
    distribution (no observation noise term).

    Parameters
    ----------
    regularization : {None, 'empirical-bayes', 'PAC'}, default=None
        - ``None``: the bare fitted ellipsoid.
        - ``'empirical-bayes'``: Ellipse+EB. A diagonal Laplace
          hyperposterior at inverse temperature ``N`` with a Gaussian
          hyperprior centered on the fitted optimum. The hyperprior, the
          whitening and the POPS baseline all use the training sample, so
          ``diagnostic_bound_`` is not a PAC bound.
        - ``'PAC'``: the hyperparameter PAC-Bayes construction. The units
          are split at random into a pilot and a certification half. The
          pilot fit fixes the centering, whitening, width floor and the
          pilot ellipsoid. The hyperparameters are the log lengths of its
          principal semi-axes (and, with ``optimize_center=True``, center
          shifts along them), with a fixed Gaussian hyperprior. On the
          certification half a Gaussian hyperposterior (the Laplace
          approximation to the Gibbs hyperposterior) is fitted for each
          temperature of a predeclared grid, and the PAC-Bayes right side of
          the floor-contaminated log loss is evaluated with an exact KL, the
          moment term of ``pac_inequality``, a union correction over the
          predeclared grids and an empirical-Bernstein bound on the Monte
          Carlo error. A last step (with its own failure probability) carries
          the bound from the continuous hyperposterior mixture to the stored
          finite mixture that ``predict*`` returns. With ``cross_fit=True``
          the roles of the halves are also swapped and the predictive is the
          equal mixture of both folds, whose bound is the average of the
          fold bounds. Requires ``y_bounds``.

    rank : int, default=32
        Rank of the ellipsoid update ``U U^T``; capped at the whitened
        dimension.

    delta : float, default=1e-3
        Radius of the output-offset coordinate ``a`` (in units of ``y``).
        It is part of the parameter distribution, so ``delta**2`` is added
        to every squared pushforward width, the projected-ball dimension is
        ``P + 1``, and :meth:`sample` (``return_offset=True``) and
        :meth:`sample_predictions` include it. For designs without a
        constant feature it adds a bounded constant shift to the function
        space. ``delta=0`` removes it. For ``regularization='PAC'`` the
        radius used is ``max(delta, min_half_width)``, which is then an
        analytic lower bound on every half-width over the whole input
        domain.

    baseline : {'pops', 'ridge', 'zero'}, default='pops'
        Fixed baseline ``B0`` of the ellipsoid shape.

    optimize_center : bool, default=False
        If True the ellipsoid center is optimized (and, with a hierarchical
        layer, uncertain) jointly with its shape; otherwise it is frozen at
        the POPS warm start.

    fit_intercept : bool, default=False
        Center the data and append an unwhitened intercept coordinate.

    preprocessor : transformer, default=None
        Optional scikit-learn transformer (for example a PCA projection)
        applied to ``X`` before the ellipsoid model. It is cloned and fitted
        on the training sample for ``None`` and ``'empirical-bayes'``, and on
        each **pilot split only** for ``'PAC'``, so the certification units
        never influence the features. With a preprocessor, :meth:`sample`
        is unavailable (the parameters of different folds live in
        different feature spaces); use :meth:`sample_predictions`.

    rho_schedule : tuple of float, default=(1e-1, 1e-2, 1e-3, 1e-4)
        Continuation schedule of the smooth log-barrier.

    tol : float, default=1e-8
        L-BFGS tolerance at each continuation stage.

    max_iter : int, default=500
        Maximum L-BFGS iterations per continuation stage.

    hyperprior_scale : float, default=1.0
        Relative variance of the empirical-Bayes Gaussian hyperprior,
        ``tau2 = hyperprior_scale * ||psi_0||^2 / d``, with ``psi_0`` the
        fitted ``(center, U)``.

    n_hyper_samples : int, default=1024
        Hyperparameter draws per fold stored to represent the hierarchical
        predictive mixture (fresh draws, independent of the Monte Carlo
        draws used in the bound). Ignored for ``regularization=None``. For
        ``'PAC'`` a larger value tightens the stored-mixture bound and slows
        prediction.

    pac_log_scale_std : float or array-like, default=(0.125, 0.25, 0.5, 1.0)
        Standard deviation of the ``'PAC'`` hyperprior on the log
        multipliers of the pilot ellipsoid's semi-axes. A sequence is a
        predeclared grid of hyperpriors: the bound is minimized over it
        with a ``log`` of its size added to the confidence term.

    pac_inequality : {'kl', 'linear'}, default='kl'
        PAC-Bayes inequality applied to the hyperparameter-level loss.
        ``'linear'`` is Theorem 1 of the paper with the Hoeffding moment
        bound of the bounded loss and a union over the temperature grid;
        ``'kl'`` is the Seeger--Maurer PAC-Bayes-kl inequality for the loss
        rescaled to ``[0, 1]``, ``kl(G_hat || G) <= (KL + log(xi(N_1) |T| /
        xi)) / N_1``, with Maurer's exact moment constant ``xi(N_1) <= 2
        sqrt(N_1)`` (valid for every ``N_1 >= 1``) and ``|T|`` the size of
        the hyperprior grid. It holds for every hyperposterior at once, so
        the temperature only indexes the Gibbs family and costs nothing.
        Both are instances of the same lifting (a PAC-Bayes inequality over
        hyperparameters followed by the Jensen step); ``'kl'`` is usually
        tighter when few units are available.

    pac_center_std : float, default=1.0
        Standard deviation of the ``'PAC'`` hyperprior on center shifts,
        in units of the pilot semi-axes (used with
        ``optimize_center=True``).

    y_bounds : tuple of float, default=None
        Known interval ``(y_lower, y_upper)`` containing the population
        output support. Required for ``'PAC'``; it must come from domain
        knowledge, not from sample extrema.

    floor_weight : float, default=0.02
        Weight ``beta`` of the uniform floor on ``y_bounds`` in the
        certified loss (``'PAC'`` only). The floor bounds the loss; it does
        not enter predictions, intervals or densities.

    min_half_width : float, default=None
        Radius of the output-offset coordinate used by ``'PAC'`` (if larger
        than ``delta``), hence a lower bound on every pushforward half-width
        anywhere in the input domain; ``None`` means ``0.01 * (y_upper -
        y_lower)``.

    pilot_fraction : float, default=0.5
        Fraction of independent units in the pilot split (``'PAC'``).

    cross_fit : bool, default=True
        Also fit the swapped fold and mix the two (``'PAC'``).

    lambda_fractions : array-like, default=None
        Temperature grid as fractions of the certification unit count,
        ``Lambda = N_1 * lambda_fractions``. ``None`` means
        ``2.0 ** -arange(11)`` (from ``N_1`` down to ``N_1 / 1024``).

    failure_probability : float, default=0.05
        PAC failure probability ``xi`` (split evenly over folds).

    mc_failure_probability : float, default=0.01
        Failure probability of the Monte Carlo evaluation of the
        hyperposterior-averaged empirical risk.

    mixture_failure_probability : float, default=0.01
        Failure probability of the step from the continuous hyperposterior
        mixture to the stored finite mixture of ``n_hyper_samples`` draws.

    n_bound_samples : int, default=4000
        Hyperparameter draws per fold for the certified Monte Carlo term.

    n_select_samples : int, default=256
        Draws per temperature used only to select ``lambda`` and the
        hyperprior width.

    random_state : int, RandomState instance or None, default=None
        Seed for the split, the initialization and all draws. ``None``
        behaves like 0.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Mean parameter vector of the (mixture) posterior (not set with a
        ``preprocessor``).

    intercept_ : float
        Mean intercept (not set with a ``preprocessor``).

    coverage_fraction_ : float
        Fraction of training points inside the bare fitted support
        (``None`` and ``'empirical-bayes'`` only).

    objective_ : float
        Final empirical objective of the fitted ellipsoid (``None`` and
        ``'empirical-bayes'`` only).

    diagnostic_bound_ : float
        Empirical-Bayes objective of the Laplace layer
        (``'empirical-bayes'`` only). Not a PAC bound.

    certificate_ : PACCertificate
        The bound and its decomposition (``'PAC'`` only).

    bound_ : float
        ``certificate_.raw_bound``, the bound for the stored mixture
        (``'PAC'`` only).

    hyperposteriors_ : list of tuple
        Per-fold Gaussian hyperposterior ``(mean, variance)`` over the axis
        hyperparameters ``(eta, omega)`` (``'PAC'`` only).

    components_ : list of tuple
        ``(weight, kernel, draws)`` per fold: the frozen pilot kernel (with
        its fitted preprocessor) and the stored hyperparameter draws that
        represent the predictive mixture.

    certificate_status_ : str
        ``'none'``, ``'diagnostic_empirical_bayes'`` or ``'pac_bound'``.

    n_iter_ : int
        Total L-BFGS iterations over all fits.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    See Also
    --------
    popsregression.POPSRegression : POPS hypercube and ensemble posteriors.

    References
    ----------
    .. [1] Swinburne, T.D. and Perez, D. (2025). "Parameter uncertainties for
           imperfect surrogate models in the low-noise regime." Machine
           Learning: Science and Technology, 6, 015008.

    Examples
    --------
    >>> import numpy as np
    >>> from popsregression import POPSEllipseRegression
    >>> rng = np.random.RandomState(0)
    >>> X = rng.uniform(-1, 1, size=(40, 3))
    >>> y = X @ np.array([1.0, -1.0, 0.5]) + 0.2 * np.sin(4 * X[:, 0])
    >>> model = POPSEllipseRegression(random_state=0).fit(X, y)
    >>> lo, hi = model.predict_interval(X[:2], level=0.9)
    >>> pac = POPSEllipseRegression(regularization="PAC", y_bounds=(-4, 4),
    ...                             random_state=0).fit(X, y)
    >>> bool(pac.certificate_.raw_bound > pac.certificate_.loss_lower)
    True
    """

    _parameter_constraints: dict = {
        "regularization": [StrOptions({"empirical-bayes", "PAC"}), None],
        "rank": [Interval(Integral, 1, None, closed="left")],
        "delta": [Interval(Real, 0, None, closed="left")],
        "baseline": [StrOptions({"pops", "ridge", "zero"})],
        "optimize_center": ["boolean"],
        "fit_intercept": ["boolean"],
        "preprocessor": [HasMethods(["fit", "transform"]), None],
        "rho_schedule": ["array-like"],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "hyperprior_scale": [
            Interval(Real, 0, None, closed="neither"),
            Options(Real, {np.inf}),
        ],
        "n_hyper_samples": [Interval(Integral, 1, None, closed="left")],
        "pac_log_scale_std": [Interval(Real, 0, None, closed="neither"), "array-like"],
        "pac_inequality": [StrOptions({"linear", "kl"})],
        "pac_center_std": [Interval(Real, 0, None, closed="neither")],
        "y_bounds": ["array-like", None],
        "floor_weight": [Interval(Real, 0, 1, closed="neither")],
        "min_half_width": [Interval(Real, 0, None, closed="neither"), None],
        "pilot_fraction": [Interval(Real, 0, 1, closed="neither")],
        "cross_fit": ["boolean"],
        "lambda_fractions": ["array-like", None],
        "failure_probability": [Interval(Real, 0, 1, closed="neither")],
        "mc_failure_probability": [Interval(Real, 0, 1, closed="neither")],
        "mixture_failure_probability": [Interval(Real, 0, 1, closed="neither")],
        "n_bound_samples": [Interval(Integral, 2, None, closed="left")],
        "n_select_samples": [Interval(Integral, 2, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        regularization=None,
        rank=32,
        delta=1e-3,
        baseline="pops",
        optimize_center=False,
        fit_intercept=False,
        preprocessor=None,
        rho_schedule=(1e-1, 1e-2, 1e-3, 1e-4),
        tol=1e-8,
        max_iter=500,
        hyperprior_scale=1.0,
        n_hyper_samples=1024,
        pac_log_scale_std=(0.125, 0.25, 0.5, 1.0),
        pac_center_std=1.0,
        pac_inequality="kl",
        y_bounds=None,
        floor_weight=0.02,
        min_half_width=None,
        pilot_fraction=0.5,
        cross_fit=True,
        lambda_fractions=None,
        failure_probability=0.05,
        mc_failure_probability=0.01,
        mixture_failure_probability=0.01,
        n_bound_samples=4000,
        n_select_samples=256,
        random_state=None,
    ):
        self.regularization = regularization
        self.rank = rank
        self.delta = delta
        self.baseline = baseline
        self.optimize_center = optimize_center
        self.fit_intercept = fit_intercept
        self.preprocessor = preprocessor
        self.rho_schedule = rho_schedule
        self.tol = tol
        self.max_iter = max_iter
        self.hyperprior_scale = hyperprior_scale
        self.n_hyper_samples = n_hyper_samples
        self.pac_log_scale_std = pac_log_scale_std
        self.pac_center_std = pac_center_std
        self.pac_inequality = pac_inequality
        self.y_bounds = y_bounds
        self.floor_weight = floor_weight
        self.min_half_width = min_half_width
        self.pilot_fraction = pilot_fraction
        self.cross_fit = cross_fit
        self.lambda_fractions = lambda_fractions
        self.failure_probability = failure_probability
        self.mc_failure_probability = mc_failure_probability
        self.mixture_failure_probability = mixture_failure_probability
        self.n_bound_samples = n_bound_samples
        self.n_select_samples = n_select_samples
        self.random_state = random_state

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def _engine(self, delta, pac_bayes, seed, weights=None):
        return _EllipsoidPosterior(
            rank=self.rank,
            delta=delta,
            baseline=self.baseline,
            rho_schedule=self.rho_schedule,
            tol=self.tol,
            max_iter=self.max_iter,
            fit_intercept=self.fit_intercept,
            weights=weights,
            optimize_center=self.optimize_center,
            random_state=seed,
            pac_bayes=pac_bayes,
            hyperprior_center="phase1",
            hyperprior_scale=self.hyperprior_scale,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, groups=None):
        """Fit the ellipsoid posterior and its regularization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Per-datum weights of the empirical objective. Not supported by
            ``regularization='PAC'``, whose loss is an average over
            independent units.

        groups : array-like of shape (n_samples,), default=None
            Independent-unit labels (``'PAC'`` only). Rows sharing a label
            are one unit: the split never separates them and their losses
            are averaged before the unit average, so ``N_1`` counts units.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = validate_data(self, X, y, dtype=[np.float64, np.float32], y_numeric=True)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        rng = check_random_state(0 if self.random_state is None else self.random_state)
        if self.regularization == "PAC":
            if sample_weight is not None:
                raise ValueError("regularization='PAC' does not support sample_weight.")
            self._fit_pac(X, y, groups, rng)
        else:
            self._fit_one_sample(X, y, sample_weight, rng)
        kernel = self.components_[0][1]
        self._n_coef = kernel.n_dim
        self._ball_dim = kernel.ball_dim
        if self.preprocessor is None:
            self._set_mean_coefficients()
        return self

    def _preprocess(self, X, y):
        """Clone and fit the preprocessor on ``(X, y)``; transformed ``X``."""
        if self.preprocessor is None:
            return None, X
        pre = clone(self.preprocessor).fit(X, y)
        return pre, np.asarray(pre.transform(X), dtype=np.float64)

    def _fit_one_sample(self, X, y, sample_weight, rng):
        weights = None
        if sample_weight is not None:
            weights = _check_sample_weight(sample_weight, X, ensure_non_negative=True)
        eb = self.regularization == "empirical-bayes"
        engine = self._engine(self.delta, eb, int(rng.randint(2**31 - 1)), weights)
        pre, Xt = self._preprocess(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DiagnosticBoundWarning)
            engine.fit(Xt, y)
        kernel = _FrozenKernel(engine, pre)
        self.n_iter_ = int(engine.n_iter_)
        self.coverage_fraction_ = engine.coverage_fraction_
        self.objective_ = engine.objective_
        self.rank_ = engine.rank_
        if eb:
            draws = kernel.psi_hat + np.sqrt(engine.hyper_sigma_diag_) * rng.randn(
                self.n_hyper_samples, kernel.psi_hat.size
            )
            self.diagnostic_bound_ = float(engine.bound_)
            self.kl_ = float(engine.kl_)
            self.hyper_sigma_diag_ = engine.hyper_sigma_diag_
            self.certificate_status_ = "diagnostic_empirical_bayes"
        else:
            draws = kernel.psi_hat[None, :]
            self.certificate_status_ = "none"
        self.components_ = [(1.0, kernel, draws)]

    def _pac_protocol(self):
        if self.y_bounds is None:
            raise ValueError(
                "regularization='PAC' requires y_bounds, a known interval "
                "containing the population output support."
            )
        y_bounds = tuple(float(v) for v in np.asarray(self.y_bounds).ravel())
        if len(y_bounds) != 2 or not y_bounds[0] < y_bounds[1]:
            raise ValueError(
                "y_bounds must be (y_lower, y_upper) with y_lower < y_upper."
            )
        width = y_bounds[1] - y_bounds[0]
        a_min = 0.01 * width if self.min_half_width is None else self.min_half_width
        fractions = (
            2.0 ** -np.arange(11)
            if self.lambda_fractions is None
            else np.asarray(self.lambda_fractions, dtype=float).ravel()
        )
        if fractions.size == 0 or np.any(fractions <= 0):
            raise ValueError("lambda_fractions must be non-empty and positive.")
        return y_bounds, float(max(self.delta, a_min)), fractions

    def _fit_pac(self, X, y, groups, rng):
        y_bounds, delta, fractions = self._pac_protocol()
        if np.any(y < y_bounds[0]) or np.any(y > y_bounds[1]):
            raise ValueError(
                "A training output lies outside y_bounds, contradicting the "
                "declared population support; outputs are never clipped."
            )
        beta = float(self.floor_weight)

        unit_labels = np.arange(y.size) if groups is None else np.asarray(groups)
        if unit_labels.shape[0] != y.size:
            raise ValueError("groups must have one entry per sample.")
        units, unit_index = np.unique(unit_labels, return_inverse=True)
        n_units = units.size
        n_pilot = int(np.floor(self.pilot_fraction * n_units))
        if n_pilot < 1 or n_pilot > n_units - 1:
            raise ValueError(
                f"pilot_fraction={self.pilot_fraction} leaves an empty split "
                f"for {n_units} independent units."
            )
        order = rng.permutation(n_units)
        halves = [order[:n_pilot], order[n_pilot:]]
        folds = [(halves[0], halves[1])]
        if self.cross_fit:
            folds.append((halves[1], halves[0]))
        xi = self.failure_probability / len(folds)
        xi_mc = self.mc_failure_probability / len(folds)
        xi_mix = self.mixture_failure_probability / len(folds)

        self.components_ = []
        self.hyperposteriors_ = []
        self.n_iter_ = 0
        fold_bounds = []
        for pilot_units, cert_units in folds:
            pilot_rows = np.isin(unit_index, pilot_units)
            cert_rows = ~pilot_rows
            engine = self._engine(delta, False, int(rng.randint(2**31 - 1)))
            # Preprocessing, whitening and the pilot ellipsoid see the pilot
            # units only.
            pre, X_pilot = self._preprocess(X[pilot_rows], y[pilot_rows])
            engine.fit(X_pilot, y[pilot_rows])
            self.n_iter_ += int(engine.n_iter_)
            kernel = _AxisKernel(engine, pre)
            loss_lower, loss_upper = floor_contaminated_loss_bounds(
                kernel.ball_dim, beta=beta, y_bounds=y_bounds, min_half_width=delta
            )
            _, local_index = np.unique(unit_index[cert_rows], return_inverse=True)
            fold, gaussian, draws = self._certify_fold(
                kernel,
                X[cert_rows],
                y[cert_rows],
                local_index,
                fractions,
                (xi, xi_mc, xi_mix),
                beta,
                y_bounds,
                (loss_lower, loss_upper),
                rng,
            )
            fold_bounds.append(fold)
            self.hyperposteriors_.append(gaussian)
            self.components_.append((1.0 / len(folds), kernel, draws))

        stored = float(np.mean([f.stored_mixture for f in fold_bounds]))
        self.certificate_ = PACCertificate(
            raw_bound=stored,
            trivial_bound=float(loss_upper),
            loss_lower=float(loss_lower),
            loss_upper=float(loss_upper),
            failure_probability=float(
                self.failure_probability
                + self.mc_failure_probability
                + self.mixture_failure_probability
            ),
            beta=beta,
            y_bounds=y_bounds,
            min_half_width=delta,
            folds=tuple(fold_bounds),
            continuous_bound=float(np.mean([f.raw for f in fold_bounds])),
            pac_failure_probability=float(self.failure_probability),
            mc_failure_probability=float(self.mc_failure_probability),
            mixture_failure_probability=float(self.mixture_failure_probability),
            n_units=int(n_units),
        )
        self.bound_ = stored
        self.certificate_status_ = "pac_bound"
        self.rank_ = self.components_[0][1].engine.rank_

    def _certify_fold(
        self,
        kernel,
        X1,
        y1,
        unit_index,
        fractions,
        budgets,
        beta,
        y_bounds,
        loss_limits,
        rng,
    ):
        """Gaussian hyperposterior and PAC right side on one certification half.

        The hyperprior ``N(0, diag(tau^2))`` over the pilot-anchored axis
        hyperparameters is fixed before the certification half is seen:
        ``pac_log_scale_std`` for the log axis multipliers and
        ``pac_center_std`` (in pilot semi-axes) for the center shifts.
        ``budgets = (xi, xi_mc, xi_mix)`` are this fold's failure
        probabilities for the PAC-Bayes inequality, the Monte Carlo term and
        the stored-mixture step.
        """
        xi, xi_mc, xi_mix = budgets
        n_units = int(unit_index.max()) + 1
        counts = np.bincount(unit_index, minlength=n_units)
        row_weights = 1.0 / counts[unit_index]
        design = kernel.design(X1)
        n_dim = kernel.n_dim
        ball_dim = kernel.ball_dim
        free = np.ones(2 * n_dim, dtype=bool)
        if not self.optimize_center:
            free[:n_dim] = False
        prior_stds = np.atleast_1d(np.asarray(self.pac_log_scale_std, dtype=float))
        lambdas = n_units * fractions
        loss_lower, loss_upper = loss_limits
        loss_range = loss_upper - loss_lower
        if self.pac_inequality == "linear":
            moment = 1.0
            log_card = np.log(lambdas.size * prior_stds.size / xi)
        else:
            moment = _maurer_xi(n_units)
            log_card = np.log(prior_stds.size * moment / xi)

        def right_side(empirical_upper, kl, lam):
            """PAC right side and its (complexity, concentration) split."""
            if self.pac_inequality == "linear":
                complexity = (kl + log_card) / lam
                concentration = lam * loss_range**2 / (8.0 * n_units)
                return (
                    empirical_upper + complexity + concentration,
                    complexity,
                    concentration,
                )
            q = (empirical_upper - loss_lower) / loss_range
            total = loss_lower + loss_range * _kl_inverse(q, (kl + log_card) / n_units)
            return total, total - empirical_upper, 0.0

        rho_schedule = np.atleast_1d(np.asarray(self.rho_schedule, dtype=float))
        averaging = csr_matrix(
            (row_weights, (np.arange(y1.size), unit_index)), shape=(y1.size, n_units)
        )
        log_floor = np.log(beta) - np.log(y_bounds[1] - y_bounds[0])

        def unit_losses(psi):
            losses, outside = [], 0
            for start in range(0, psi.shape[0], 512):
                mean, half = kernel.pushforward(design, psi[start : start + 512])
                log_ball = projected_ball_logpdf(y1[None, :] - mean, half, ball_dim)
                outside += int(np.sum(np.isneginf(log_ball)))
                log_p = np.logaddexp(np.log1p(-beta) + log_ball, log_floor)
                losses.append(np.asarray((averaging.T @ -log_p.T).T))
            return np.vstack(losses), outside / (psi.shape[0] * y1.size)

        def hyperposterior(lam, tau2):
            # Laplace approximation to the Gibbs hyperposterior at
            # temperature lam: mode of lam * G_hat + ||psi||^2 / (2 tau^2)
            # (sum form: sum_rows w l + (N_1 / lam) ||psi||^2 / (2 tau^2)).
            prec = np.where(free, n_units / (lam * tau2), 0.0)
            psi = np.zeros(2 * n_dim)
            for rho in rho_schedule:

                def objective(p_free):
                    full = psi.copy()
                    full[free] = p_free
                    value, grad = kernel.objective(
                        full, design, y1, row_weights, rho, prec
                    )
                    return value, grad[free]

                res = minimize(
                    objective,
                    psi[free],
                    jac=True,
                    method="L-BFGS-B",
                    tol=self.tol,
                    options={"maxiter": self.max_iter},
                )
                psi[free] = res.x
                self.n_iter_ += int(res.nit)
            hd = kernel.hess_diag(psi, design, y1, row_weights, rho_schedule[-1])
            var = np.zeros_like(psi)
            var[free] = 1.0 / (
                (lam / n_units) * np.maximum(hd[free], 0.0) + 1.0 / tau2[free]
            )
            m, v, t = psi[free], var[free], tau2[free]
            kl = 0.5 * float(np.sum(v / t + m * m / t - 1.0 + np.log(t / v)))
            return psi, var, kl

        def draw(mean, var, n):
            return mean + np.sqrt(var) * rng.randn(n, mean.size)

        # Select (prior, lambda) on a small independent Monte Carlo sample;
        # the log-cardinality term makes any data-dependent choice valid.
        candidates = []
        for prior_std in prior_stds:
            tau2 = np.full(2 * n_dim, prior_std**2)
            tau2[:n_dim] = float(self.pac_center_std) ** 2
            for lam in lambdas:
                mean, var, kl = hyperposterior(lam, tau2)
                losses, _ = unit_losses(draw(mean, var, self.n_select_samples))
                estimate = right_side(losses.mean(), kl, lam)[0]
                candidates.append((estimate, lam, prior_std, mean, var, kl))
        _, lam, prior_std, mean, var, kl = min(candidates, key=lambda c: c[0])

        # Certified evaluation on fresh draws.
        losses, outside = unit_losses(draw(mean, var, self.n_bound_samples))
        per_draw = losses.mean(axis=1)
        empirical = float(per_draw.mean())
        mc = _empirical_bernstein(per_draw, loss_range, xi_mc)
        raw, complexity, concentration = right_side(empirical + mc, kl, lam)
        mixture = float(
            np.mean(-(logsumexp(-losses, axis=0) - np.log(losses.shape[0])))
        )
        # Stored finite mixture of K fresh draws: its loss is at most the
        # draw average of L(Psi_k) (Jensen, pointwise in y), an average of K
        # i.i.d. values in [a, b] with mean E_q L <= raw; the Chernoff-kl
        # tail bound (valid for [0, 1] variables) gives the deviation.
        n_stored = int(self.n_hyper_samples)
        q_raw = float(np.clip((raw - loss_lower) / loss_range, 0.0, 1.0))
        stored = loss_lower + loss_range * _kl_inverse(
            q_raw, np.log(1.0 / xi_mix) / n_stored
        )
        fold = PACFoldBound(
            lam=float(lam),
            n_units=n_units,
            empirical=empirical,
            monte_carlo=mc,
            kl=float(kl),
            complexity=float(complexity),
            concentration=float(concentration),
            raw=float(raw),
            mixture_empirical=mixture,
            outside_support_fraction=float(outside),
            prior_std=float(prior_std),
            inequality=self.pac_inequality,
            moment_constant=float(moment),
            stored_mixture=float(stored),
            n_stored=n_stored,
        )
        return fold, (mean, var), draw(mean, var, n_stored)

    def _set_mean_coefficients(self):
        coef, intercept = 0.0, 0.0
        for weight, kernel, draws in self.components_:
            c, i = kernel.coefficients(draws)
            coef = coef + weight * c.mean(axis=0)
            intercept = intercept + weight * float(i.mean())
        self.coef_ = np.asarray(coef)
        self.intercept_ = float(intercept)

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------

    def _mixture(self, X):
        """Weights, means and half-widths of every pushforward component."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)
        X = np.asarray(X, dtype=np.float64)
        weights, means, halves = [], [], []
        for weight, kernel, draws in self.components_:
            mean, half = kernel.pushforward(kernel.design(X), draws)
            weights.append(np.full(draws.shape[0], weight / draws.shape[0]))
            means.append(mean)
            halves.append(half)
        return np.concatenate(weights), np.vstack(means), np.vstack(halves)

    def predict(self, X, return_std=False, return_bounds=False):
        """Predict the mean of the parameter-only predictive.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        return_std : bool, default=False
            Also return the predictive standard deviation.

        return_bounds : bool, default=False
            Also return the upper and lower support bounds of the predictive
            mixture: the exact support of the bare ellipsoid, or the
            envelope of the supports of the stored hyperparameter draws.

        Returns
        -------
        y_mean : ndarray of shape (n_samples,)

        y_std : ndarray of shape (n_samples,)
            Only if ``return_std=True``.

        y_max, y_min : ndarray of shape (n_samples,)
            Only if ``return_bounds=True``.
        """
        w, mean, half = self._mixture(X)
        y_mean = w @ mean
        if not (return_std or return_bounds):
            return y_mean
        result = [y_mean]
        if return_std:
            second = w @ (half**2 / (self._ball_dim + 2.0) + mean**2)
            result.append(np.sqrt(np.maximum(second - y_mean**2, 0.0)))
        if return_bounds:
            result.extend([(mean + half).max(axis=0), (mean - half).min(axis=0)])
        return tuple(result)

    def predict_logpdf(self, X, y):
        """Log density of the parameter-only predictive at ``y``.

        Equal to ``-inf`` where ``y`` is outside the support of every
        pushforward component.
        """
        w, mean, half = self._mixture(X)
        y = np.asarray(y, dtype=float).ravel()
        with np.errstate(divide="ignore"):
            log_w = np.log(w)[:, None]
        comp = projected_ball_logpdf(y[None, :] - mean, half, self._ball_dim)
        return logsumexp(log_w + comp, axis=0)

    def predict_cdf(self, X, y):
        """Cumulative distribution function of the predictive at ``y``."""
        w, mean, half = self._mixture(X)
        y = np.asarray(y, dtype=float).ravel()
        return w @ self._component_cdf(y[None, :], mean, half)

    def _component_cdf(self, y, mean, half):
        a = 0.5 * (self._ball_dim + 1)
        u = np.clip(0.5 * ((y - mean) / half + 1.0), 0.0, 1.0)
        return betainc(a, a, u)

    def predict_interval(self, X, level=0.9545, n_bisect=60, xtol=1e-10):
        """Central interval of the parameter-only predictive.

        The quantiles are exact (to the root-finding tolerance) for the
        returned predictive, which is the stored finite mixture of
        hyperparameter draws; for ``'empirical-bayes'`` and ``'PAC'`` that
        mixture is a Monte Carlo representation of the continuous
        hyperposterior mixture, whose quantiles are not computed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        level : float, default=0.9545
            Central probability of the interval.

        n_bisect : int, default=60
            Maximum number of safeguarded Newton steps on the mixture CDF
            (each falls back to bisection when Newton would leave the
            current bracket).

        xtol : float, default=1e-10
            Stop when every quantile has moved by less than ``xtol`` times
            the width of its initial bracket.

        Returns
        -------
        lower, upper : ndarray of shape (n_samples,)
        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1).")
        w, mean, half = self._mixture(X)
        lo0 = (mean - half).min(axis=0)
        hi0 = (mean + half).max(axis=0)
        scale = np.maximum(hi0 - lo0, np.finfo(float).tiny)

        def quantile(prob):
            lo, hi = lo0.copy(), hi0.copy()
            x = np.clip(w @ mean, lo, hi)
            for _ in range(n_bisect):
                F = w @ self._component_cdf(x[None, :], mean, half)
                f = w @ np.exp(
                    projected_ball_logpdf(x[None, :] - mean, half, self._ball_dim)
                )
                below = F < prob
                lo = np.where(below, x, lo)
                hi = np.where(below, hi, x)
                with np.errstate(divide="ignore", invalid="ignore"):
                    newton = x - (F - prob) / f
                inside = (f > 0) & (newton > lo) & (newton < hi)
                x_new = np.where(inside, newton, 0.5 * (lo + hi))
                done = np.max(np.abs(x_new - x) / scale) < xtol
                x = x_new
                if done:
                    break
            return x

        tail = 0.5 * (1.0 - level)
        return quantile(tail), quantile(1.0 - tail)

    def _assign_draws(self, n_samples, rng):
        """Fold and stored hyperparameter draw of each sample."""
        weights = np.array([c[0] for c in self.components_])
        which = rng.choice(len(self.components_), size=n_samples, p=weights)
        for k, (_, kernel, draws) in enumerate(self.components_):
            idx = np.flatnonzero(which == k)
            if idx.size:
                yield idx, kernel, draws[rng.randint(draws.shape[0], size=idx.size)]

    def sample(self, n_samples, random_state=None, return_offset=False):
        """Draw parameter vectors from the (hierarchical) posterior.

        Each draw selects a fold and a stored hyperparameter draw, then a
        uniform point of that ellipsoid in ``(theta, a)``; for propagation
        reuse one draw for every input of a field. The same distribution
        gives every score and interval of this estimator.

        Parameters
        ----------
        n_samples : int
            Number of parameter vectors.

        random_state : int, RandomState instance or None, default=None
            Seed of the draws.

        return_offset : bool, default=False
            Also return the output offsets ``a`` (``|a| <= delta``), so that
            each draw's function is ``x -> x @ theta (+ intercept) + a``.
            Without them the draws are the coefficient marginal.

        Returns
        -------
        samples : ndarray of shape (n_features (+ 1), n_samples)
            Parameter vectors in original coordinates, with an intercept row
            appended if ``fit_intercept=True``.

        offsets : ndarray of shape (n_samples,)
            Only if ``return_offset=True``.
        """
        check_is_fitted(self)
        if self.preprocessor is not None:
            raise ValueError(
                "sample() is unavailable with a preprocessor: the parameters of "
                "each fold live in its own feature space. Use "
                "sample_predictions(X, n_samples)."
            )
        rng = check_random_state(random_state)
        out = np.empty((self._n_coef, n_samples))
        offsets = np.empty(n_samples)
        for idx, kernel, psi in self._assign_draws(n_samples, rng):
            out[:, idx], offsets[idx] = kernel.sample_parameters(psi, rng)
        return (out, offsets) if return_offset else out

    def sample_predictions(self, X, n_samples, random_state=None):
        """Joint draws of the predictive at the rows of ``X``.

        Each column is one parameter draw ``(theta, a)`` (see :meth:`sample`)
        evaluated at every row, so columns are coherent functions of the
        input and the marginal at each row is the scored predictive.

        Parameters
        ----------
        X : array-like of shape (n_points, n_features)
            Inputs.

        n_samples : int
            Number of draws.

        random_state : int, RandomState instance or None, default=None
            Seed of the draws.

        Returns
        -------
        draws : ndarray of shape (n_points, n_samples)
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)
        X = np.asarray(X, dtype=np.float64)
        rng = check_random_state(random_state)
        out = np.empty((X.shape[0], n_samples))
        for idx, kernel, psi in self._assign_draws(n_samples, rng):
            out[:, idx] = kernel.sample_outputs(X, psi, rng)
        return out
