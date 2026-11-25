"""
Low-noise predictive-risk objectives: PACm-Bayes and PAC^2_T-Bayes.

Independent NumPy/SciPy implementations of two misspecification-aware
learning objectives for a Gaussian working likelihood
``p(y | x, theta) = N(y; x.theta, sigma^2)`` on a linear model, used to
compare their behaviour with POPS as the likelihood width ``sigma``
decreases:

- **PACm-Bayes** (Morningstar, Alemi and Dillon, AISTATS 2022, Eq. 13):
  for a Gaussian variational posterior ``q`` and ``m`` joint draws
  ``theta_1..theta_m ~ q`` shared across the data,

  .. code-block:: text

      PACm = E_q[ -(1/n) sum_i log( (1/m) sum_j p(y_i | x_i, theta_j) ) ]
             + KL(q || prior) / (beta n).

- **PAC^2_T-Bayes** (Masegosa, NeurIPS 2020, arXiv:1912.08335): the
  first-order expected log loss minus a second-order variance term with
  the tighter Taylor weight ``h`` (their Eqs. 4-5 and the stable form of
  their Appendix C, "numerically stable version of the PAC^2-Variational
  learning"),

  .. code-block:: text

      PAC2_T = (1/n) sum_i E_{theta, theta' ~ q}[ -ln p_i(theta)
               - h(alpha_i) V_i(theta, theta') ] + KL(q || prior) / n,
      V_i    = exp(2 a - 2 M) - exp(a + b - 2 M),
      a      = ln p_i(theta),  b = ln p_i(theta'),  M = max(a, b) + 0.1,
      alpha_i = ln(exp(a - M) + exp(b - M)) - ln 2,
      h(alpha) = alpha / (1 - e^alpha)^2 + 1 / (e^alpha (1 - e^alpha)),

  with ``theta`` and ``theta'`` two *independent* draws from ``q`` (the
  tandem loss of their Lemma for Theorem 3, ``rho(theta, theta') =
  rho(theta) rho(theta')``), the data term evaluated at the first draw
  only, KL weight ``1 / n`` (``c = 1``; the pair KL is ``2 KL`` and the
  bound is taken at ``lambda = 2 c n``), and ``M`` the regression choice
  of their Appendix C (the larger of the two sampled log densities plus
  ``epsilon = 0.1``). ``objective='pac2'`` is the untightened Eq. 4, whose
  weight is ``1 / (2 max p^2)``, i.e. ``h = 1/2`` in these normalized
  units; this matches the published code (github.com/PGM-Lab/PAC2BAYES,
  ``var = 0.5 * mean(hmax * (...))`` with ``hmax = 1`` for PAC^2 and
  ``hmax = 2 h`` for PAC^2_T). The appendix's stable form writes ``h = 1``
  for PAC^2, which drops that factor 1/2; the PAC^2_T weight is the same
  in the paper, the code and here. The ensemble variant (their Theorem 4 /
  Appendix C.3) replaces ``q`` by ``E`` equally weighted particles.

  The published code draws **one** independent pair per optimizer step
  (fresh at every Adam step) and applies a stop-gradient to ``M`` and
  ``h``. Here the objective values are exactly those of the stable form,
  but because L-BFGS needs a gradient consistent with the evaluated
  function, ``M`` and ``h`` are differentiated exactly rather than held
  fixed.

**Fixed draws versus the expectation.** The expectations over ``q`` are
replaced by sample averages over ``n_groups`` independent groups of
reparameterized draws ``theta = mu + L eps``, with the standard normal
``eps`` drawn once at the start of the fit and then held fixed (a
"sample-average approximation"). The objective is then an ordinary
deterministic function of ``(mu, L)``, which L-BFGS minimizes with exact
gradients; its minimizer approaches the minimizer of the expectation as
``n_groups`` grows. What ``m`` means differs between the two objectives:

- PACm: ``m`` is part of the *objective itself* (the log of an ``m``-draw
  average is not the average of logs), so each group is one ``m``-tuple
  and ``n_groups`` only controls how well the expectation over tuples is
  approximated.
- PAC^2_T: the objective is a plain average over independent pairs, so a
  group of ``S`` pairs is just ``S`` more pairs. ``n_samples`` (``S``) and
  ``n_groups`` together only set the Monte Carlo precision
  (``n_groups * S`` pairs); the expected objective does not depend on
  ``S``. ``S = 1`` is the published one-pair-per-step convention.

The ``sigma^{-2}`` divergence is present both for the fixed draws and for
the expectation. For PACm with ``m`` fixed and the nearest of the ``m``
residuals ``r_min`` nonzero, ``-ln((1/m) sum_j p_j) >= r_min^2 / (2
sigma^2) + ln(sigma sqrt(2 pi))``, and on misspecified data no Gaussian
``q`` makes ``E_q[r_min^2]`` vanish for every datum, so the expected
objective also grows like ``sigma^{-2}``; a larger finite ``m`` lowers the
coefficient and delays the onset but does not remove it. For PAC^2_T the
data term is the single-model log loss ``E_q[r^2] / (2 sigma^2) + ...``,
while ``alpha`` lies in ``[-0.1 - ln 2, -0.1]`` so ``h V`` is bounded by a
constant: the second-order correction cannot cancel the divergence. With
``m = 1`` (and for ``'elbo'``) the expectation of the Gaussian log loss
under a Gaussian ``q`` is evaluated in closed form instead of by draws. A
single fixed draw would let ``q`` inflate its covariance while the sampled
parameter stays put, which is why several groups are needed.

:meth:`LowNoiseObjective.evaluate_objective` re-evaluates the same
objective at the fitted ``q`` on *fresh* independent draws, to check that
the fitted value is not an artefact of the particular fixed draws.

Low-noise convention: the likelihood width ``sigma`` is a fitting device
only. Every prediction, interval and predictive score uses the
**parameter-only** predictive, the pushforward of ``q`` (a Gaussian
``N(x.mu, x.T Sigma x)``) or of the particle ensemble (a discrete
distribution with no density), unless ``include_sigma=True`` is requested
explicitly.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.utils import check_random_state

OBJECTIVES = ("pacm", "pac2", "pac2t", "pac2t_ensemble", "elbo")
_M_OFFSET = 0.1


def taylor_weight(alpha, derivative=False):
    """``h(alpha)`` of Masegosa (2020), Eq. 5 / C.13, for ``alpha < 0``.

    With ``derivative=True`` also return ``dh/dalpha``.
    """
    alpha = np.asarray(alpha, dtype=float)
    u = np.exp(alpha)
    h = alpha / (1.0 - u) ** 2 + 1.0 / (u * (1.0 - u))
    if not derivative:
        return h
    dh = 2.0 / (1.0 - u) ** 2 + 2.0 * alpha * u / (1.0 - u) ** 3 - 1.0 / (u * (1.0 - u))
    return h, dh


def _termination_reason(res):
    """Why L-BFGS-B stopped: 'converged', 'max_iter', 'max_fun',
    'line_search' or 'other' (read from scipy's message)."""
    if res.success:
        return "converged"
    message = str(res.message).upper()
    if "ITERATIONS REACHED LIMIT" in message:
        return "max_iter"
    if "EVALUATIONS EXCEEDS LIMIT" in message:
        return "max_fun"
    # scipy >= 1.15 reports a failed line search as "ABNORMAL: " (status 2);
    # older versions as "ABNORMAL_TERMINATION_IN_LNSRCH".
    if "ABNORMAL" in message or "LNSRCH" in message or "LINE SEARCH" in message:
        return "line_search"
    return "other"


def _tril_indices(p):
    return np.tril_indices(p, k=-1)


class _GaussianFamily:
    """Full-covariance Gaussian ``q = N(mu, L L^T)`` packed as a vector."""

    def __init__(self, p):
        self.p = p
        self.rows, self.cols = _tril_indices(p)
        self.size = 2 * p + self.rows.size

    def unpack(self, params):
        p = self.p
        mu = params[:p]
        log_diag = params[p : 2 * p]
        L = np.zeros((p, p))
        L[np.diag_indices(p)] = np.exp(log_diag)
        L[self.rows, self.cols] = params[2 * p :]
        return mu, L

    def pack_grad(self, g_mu, g_L, params):
        p = self.p
        log_diag = params[p : 2 * p]
        return np.concatenate(
            [g_mu, np.diag(g_L) * np.exp(log_diag), g_L[self.rows, self.cols]]
        )

    def kl(self, mu, L, prior_var):
        """``KL(N(mu, L L^T) || N(0, prior_var I))`` and its gradients."""
        p = self.p
        diag = np.diag(L)
        value = 0.5 * (
            np.sum(L * L) / prior_var
            + mu @ mu / prior_var
            - p
            + p * np.log(prior_var)
            - 2.0 * np.sum(np.log(diag))
        )
        g_mu = mu / prior_var
        g_L = L / prior_var
        g_L[np.diag_indices(p)] -= 1.0 / diag
        return value, g_mu, g_L


class LowNoiseObjective:
    """PACm / PAC^2_T variational or ensemble learning for a linear model.

    Parameters
    ----------
    objective : {'pacm', 'pac2t', 'pac2', 'pac2t_ensemble', 'elbo'}
        ``'elbo'`` is PACm with ``m = 1`` (ordinary variational Bayes).

    sigma : float, default=0.1
        Likelihood width in units of the training-target standard deviation
        (with ``standardize=True`` features and target are divided by their
        root-mean-square, never centered, so no implicit intercept is added
        to the shared design matrix). ``sigma_`` records the original-unit
        value.

    fit_sigma : bool, default=False
        Learn ``log sigma`` jointly (the fitted-residual-scale variant);
        ``sigma`` is then the initial value.

    n_samples : int, default=16
        ``m`` (PACm), the number of ``(theta, theta')`` pairs per group
        (PAC^2 / PAC^2_T; this only sets the number of Monte Carlo pairs,
        see the module notes) or the number of particles ``E`` (ensemble).

    n_groups : int, default=32
        Number of independent draw groups averaged in the sample-average
        approximation of the variational objectives (ignored by the
        ensemble and by closed-form ``m = 1``).

    prior_scale : float, default=10.0
        Prior standard deviation of every coefficient, ``N(0, prior_scale^2
        I)`` in standardized units; also the ensemble's Gaussian prior.

    beta : float, default=1.0
        PACm's KL temperature (Eq. 13); ``beta = 1`` for every objective
        unless changed.

    max_iter : int, default=20000
        L-BFGS-B iteration limit (scipy ``maxiter``).

    max_fun : int or None, default=None
        L-BFGS-B limit on objective-and-gradient evaluations (scipy
        ``maxfun``, a separate limit whose scipy default is 15000). ``None``
        uses ``2 * max_iter``, so that the iteration limit is normally the
        one that binds; the value used is recorded in ``max_fun_``.

    tol : float, default=1e-9
        Projected-gradient tolerance (scipy ``gtol``).

    random_state : int, default=0
        Seed of the fixed reparameterization draws and particle jitter.

    Attributes
    ----------
    converged_ : bool
        L-BFGS-B reported success (``res.success``).

    termination_status_, termination_message_ : int, str
        scipy's ``res.status`` (0 converged, 1 an iteration or evaluation
        limit was reached, 2 other, e.g. a failed line search) and
        ``res.message``.

    termination_ : str
        One of ``'converged'``, ``'max_iter'``, ``'max_fun'``,
        ``'line_search'`` (scipy's "ABNORMAL" stop: the line search could
        not lower the objective) or ``'other'``, read from the message.

    n_iter_, n_fev_, n_evaluations_, n_nonfinite_ : int
        Iterations (``res.nit``), evaluations reported by scipy
        (``res.nfev``), evaluations counted here, and evaluations returning
        a non-finite value or gradient (those return a large finite value).

    max_iter_, max_fun_ : int
        The iteration and evaluation limits actually passed to L-BFGS-B.

    params_ : ndarray
        The returned iterate, kept whether or not the run converged.

    objective_value_ : float
        Final objective (per datum, standardized units) on the fixed
        training draws.

    objective_data_, objective_kl_ : float
        Its two parts: the per-datum data term and ``KL / (beta n)`` (for
        the ensemble, its Gaussian prior term).

    objective_variance_ : float
        For PAC^2 / PAC^2_T (and the ensemble), the per-datum second-order
        correction ``mean(h V)`` already subtracted inside the data term;
        0 otherwise.

    fit_time_ : float
        Seconds spent optimizing.

    sigma_ : float
        Final likelihood width in original units.
    """

    def __init__(
        self,
        objective="pacm",
        *,
        sigma=0.1,
        fit_sigma=False,
        n_samples=16,
        n_groups=32,
        prior_scale=10.0,
        beta=1.0,
        max_iter=20000,
        max_fun=None,
        tol=1e-9,
        standardize=True,
        random_state=0,
    ):
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}.")
        self.objective = objective
        self.sigma = sigma
        self.fit_sigma = fit_sigma
        self.n_samples = n_samples
        self.n_groups = n_groups
        self.prior_scale = prior_scale
        self.beta = beta
        self.max_iter = max_iter
        self.max_fun = max_fun
        self.tol = tol
        self.standardize = standardize
        self.random_state = random_state

    # -- preprocessing --------------------------------------------------

    def _standardize_fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if self.standardize:
            # Scale by the root-mean-square, never center: centering would add
            # an implicit intercept that the other methods, fitted on the same
            # design matrix, do not have.
            rms = np.sqrt(np.mean(X * X, axis=0))
            self.x_scale_ = np.where(rms > 0, rms, 1.0)
            self.x_mean_ = np.zeros_like(self.x_scale_)
            self.y_mean_ = 0.0
            self.y_scale_ = float(np.sqrt(np.mean(y * y))) or 1.0
            # ``sigma`` is specified in units of the target standard deviation.
            self._sigma_unit = float(y.std()) / self.y_scale_ if y.std() > 0 else 1.0
        else:
            self.x_mean_, self.x_scale_ = np.zeros(X.shape[1]), np.ones(X.shape[1])
            self.y_mean_, self.y_scale_ = 0.0, 1.0
            self._sigma_unit = 1.0
        return (X - self.x_mean_) / self.x_scale_, (y - self.y_mean_) / self.y_scale_

    def _transform(self, X):
        return (np.asarray(X, dtype=float) - self.x_mean_) / self.x_scale_

    # -- per-datum log likelihood and its derivative ---------------------

    @staticmethod
    def _loglik(Z, ys, Theta, sigma):
        """``a[i, j] = ln N(y_i; z_i.theta_j, sigma^2)`` and residuals."""
        R = ys[:, None] - Z @ Theta
        a = -0.5 * np.log(2.0 * np.pi * sigma**2) - 0.5 * R * R / sigma**2
        return a, R

    # -- objectives: value and gradient w.r.t. a[i, j] and log sigma ------

    def _per_datum(self, a, kind):
        """Return objective value (per datum) and ``dF/da`` matrix.

        The value excludes the prior/KL term; ``a`` is (n, m) for PACm and
        the ensemble, or (n, 2, S) for the pair objectives.
        """
        n = a.shape[0]
        if kind == "pacm":
            # a is (n, K, m): log-mean-exp within each m-tuple, mean over
            # the K groups and the data.
            K, m = a.shape[1], a.shape[2]
            lse = logsumexp(a, axis=2)
            value = -np.mean(lse - np.log(m))
            dF = -np.exp(a - lse[:, :, None]) / (n * K)
            return value, dF
        if kind in ("pac2", "pac2t"):
            A, B = a[:, 0, :], a[:, 1, :]
            S = A.shape[1]
            a_max = A >= B
            M = np.maximum(A, B) + _M_OFFSET
            ea, eb = np.exp(A - M), np.exp(B - M)
            V = ea * ea - ea * eb
            dV_dA, dV_dB, dV_dM = 2.0 * ea * ea - ea * eb, -ea * eb, -2.0 * V
            if kind == "pac2t":
                alpha = np.log(ea + eb) - np.log(2.0)
                h, dh = taylor_weight(alpha, derivative=True)
                w_a = ea / (ea + eb)
                dalpha_dA, dalpha_dB, dalpha_dM = w_a, 1.0 - w_a, -1.0
            else:
                # Eq. 4: weight 1 / (2 max p^2), i.e. 1/2 in units of e^M
                # (the published code's 0.5 * var with hmax = 1).
                h, dh = np.full_like(A, 0.5), np.zeros_like(A)
                dalpha_dA = dalpha_dB = dalpha_dM = 0.0
            value = np.mean(-A - h * V)
            # Exact derivative through M = max(A, B) + offset and h(alpha).
            dF_dM = -(h * dV_dM + V * dh * dalpha_dM)
            dA = -1.0 - h * dV_dA - V * dh * dalpha_dA + np.where(a_max, dF_dM, 0.0)
            dB = -h * dV_dB - V * dh * dalpha_dB + np.where(a_max, 0.0, dF_dM)
            return value, np.stack([dA, dB], axis=1) / (n * S)
        # Ensemble of E equally weighted particles.
        E = a.shape[1]
        argmax = a.argmax(axis=1)
        M = a.max(axis=1, keepdims=True) + _M_OFFSET
        e = np.exp(a - M)
        s = e.sum(axis=1, keepdims=True)
        alpha = np.log(s[:, 0]) - np.log(E)
        h, dh = taylor_weight(alpha, derivative=True)
        h, dh = h[:, None], dh[:, None]
        var = np.mean(e * e, axis=1, keepdims=True) - (s / E) ** 2
        value = np.mean(-a.mean(axis=1) - (h * var)[:, 0])
        dvar_da = 2.0 * e * e / E - 2.0 * e * s / E**2
        dF = -1.0 / E - h * dvar_da - var * dh * (e / s)
        # Through M (only the arg-max particle): dvar/dM = -2 var,
        # dalpha/dM = -1.
        dF_dM = -(h * (-2.0 * var) + var * dh * (-1.0))
        dF[np.arange(n), argmax] += dF_dM[:, 0]
        return value, dF / n

    def _objective(self, params, Z, ys, eps, parts=False):
        """Objective value and gradient, or with ``parts=True`` a dict of
        its data term, KL (or prior) term and second-order correction."""
        n, p = Z.shape
        kind = self.objective
        if kind == "pac2t_ensemble":
            E = self.n_samples
            Theta = params[: p * E].reshape(p, E)
            log_sigma = (
                params[p * E]
                if self.fit_sigma
                else np.log(self.sigma * self._sigma_unit)
            )
            sigma = np.exp(log_sigma)
            a, R = self._loglik(Z, ys, Theta, sigma)
            value, dF = self._per_datum(a, kind)
            prior_var = self.prior_scale**2
            prior = 0.5 * np.sum(Theta * Theta) / prior_var / (n * E)
            if parts:
                first = float(np.mean(-a))
                return dict(data=value, kl=prior, variance=first - value)
            value += prior
            dTheta = Z.T @ (dF * R) / sigma**2 + Theta / prior_var / (n * E)
            grad = [dTheta.ravel()]
            if self.fit_sigma:
                grad.append(np.array([np.sum(dF * (R * R / sigma**2 - 1.0))]))
            return value, np.concatenate(grad)

        fam = self._family
        mu, L = fam.unpack(params[: fam.size])
        log_sigma = (
            params[fam.size]
            if self.fit_sigma
            else np.log(self.sigma * self._sigma_unit)
        )
        sigma = np.exp(log_sigma)
        if eps is None:
            # Closed-form E_q[-ln p] for the Gaussian likelihood: the
            # expected squared residual is r_i^2 + ||L^T z_i||^2.
            r = ys - Z @ mu
            ZL = Z @ L
            spread = np.einsum("ij,ij->i", ZL, ZL)
            value = np.mean(
                0.5 * np.log(2.0 * np.pi * sigma**2)
                + 0.5 * (r * r + spread) / sigma**2
            )
            g_mu = -(Z.T @ r) / sigma**2 / n
            g_L = np.tril(Z.T @ ZL) / sigma**2 / n
            g_sigma = np.mean(1.0 - (r * r + spread) / sigma**2)
        else:
            eps_flat = eps.reshape(p, -1)
            Theta = mu[:, None] + L @ eps_flat
            a, R = self._loglik(Z, ys, Theta, sigma)
            first = np.nan
            if kind in ("pac2", "pac2t"):
                S = eps_flat.shape[1] // 2
                value, dF = self._per_datum(a.reshape(n, 2, S), kind)
                first = float(np.mean(-a[:, :S]))
            else:
                # eps is (p, groups, m): one m-tuple per group.
                value, dF = self._per_datum(a.reshape(n, eps.shape[1], -1), "pacm")
            dF = dF.reshape(n, -1)
            D = Z.T @ (dF * R) / sigma**2  # (p, n_draws): dF/dtheta_j
            g_mu = D.sum(axis=1)
            g_L = np.tril(D @ eps_flat.T)
            g_sigma = np.sum(dF * (R * R / sigma**2 - 1.0))
        kl, kl_mu, kl_L = fam.kl(mu, L, self.prior_scale**2)
        scale = 1.0 / (self.beta * n)
        if parts:
            variance = first - value if kind in ("pac2", "pac2t") else 0.0
            return dict(data=float(value), kl=float(scale * kl), variance=variance)
        value += scale * kl
        g_mu += scale * kl_mu
        g_L += scale * np.tril(kl_L)
        grad = [fam.pack_grad(g_mu, g_L, params[: fam.size])]
        if self.fit_sigma:
            grad.append(np.array([g_sigma]))
        return value, np.concatenate(grad)

    # -- fitting ---------------------------------------------------------

    def fit(self, X, y):
        """Optimize the objective by L-BFGS-B with analytic gradients."""
        t0 = time.perf_counter()
        Z, ys = self._standardize_fit(X, y)
        n, p = Z.shape
        rng = check_random_state(self.random_state)
        ridge = np.linalg.solve(Z.T @ Z + 1e-6 * np.eye(p), Z.T @ ys)
        kind = self.objective
        if kind == "pac2t_ensemble":
            E = self.n_samples
            Theta0 = ridge[:, None] + 0.1 * rng.randn(p, E)
            x0 = [Theta0.ravel()]
            eps = None
        else:
            self._family = _GaussianFamily(p)
            K = self.n_groups
            if kind in ("pac2", "pac2t"):
                eps = rng.randn(p, 2, K * self.n_samples)
            elif kind == "elbo" or self.n_samples == 1:
                eps = None  # closed form
            else:
                eps = rng.randn(p, K, self.n_samples)
            self._eps = eps
            x0 = [
                np.concatenate(
                    [ridge, np.log(0.1) * np.ones(p), np.zeros(p * (p - 1) // 2)]
                )
            ]
        if self.fit_sigma:
            x0.append(np.array([np.log(self.sigma * self._sigma_unit)]))
        x0 = np.concatenate(x0)

        self.n_evaluations_ = 0
        self.n_nonfinite_ = 0

        def fun(params):
            self.n_evaluations_ += 1
            with np.errstate(over="ignore", invalid="ignore"):
                value, grad = self._objective(params, Z, ys, eps)
            if not (np.isfinite(value) and np.all(np.isfinite(grad))):
                self.n_nonfinite_ += 1
                return 1e300, np.zeros_like(params)
            return value, grad

        self.max_iter_ = int(self.max_iter)
        self.max_fun_ = int(2 * self.max_iter if self.max_fun is None else self.max_fun)
        res = minimize(
            fun,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": self.max_iter_,
                "maxfun": self.max_fun_,
                "gtol": self.tol,
                "ftol": 1e-14,
            },
        )
        self.converged_ = bool(res.success)
        self.termination_status_ = int(res.status)
        self.termination_message_ = self.message_ = str(res.message)
        self.termination_ = _termination_reason(res)
        self.n_iter_ = int(res.nit)
        self.n_fev_ = int(res.nfev)
        self.objective_value_ = float(res.fun)
        params = self.params_ = np.array(res.x, dtype=float)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            split = self._objective(params, Z, ys, eps, parts=True)
        self.objective_data_ = float(split["data"])
        self.objective_kl_ = float(split["kl"])
        self.objective_variance_ = float(split["variance"])
        if kind == "pac2t_ensemble":
            E = self.n_samples
            self.particles_ = params[: p * E].reshape(p, E)
            log_sigma = (
                params[p * E]
                if self.fit_sigma
                else np.log(self.sigma * self._sigma_unit)
            )
        else:
            self.mu_, self.L_ = self._family.unpack(params[: self._family.size])
            log_sigma = (
                params[self._family.size]
                if self.fit_sigma
                else np.log(self.sigma * self._sigma_unit)
            )
        self.sigma_standardized_ = float(np.exp(log_sigma))
        self.sigma_ = self.y_scale_ * self.sigma_standardized_
        self.fit_time_ = time.perf_counter() - t0
        return self

    # -- independent re-evaluation -----------------------------------------

    def _fresh_draws(self, p, n_groups, rng):
        kind = self.objective
        if kind == "pac2t_ensemble":
            return None
        if kind in ("pac2", "pac2t"):
            return rng.randn(p, 2, n_groups * self.n_samples)
        m = 1 if kind == "elbo" else self.n_samples
        return rng.randn(p, n_groups, m)

    def evaluate_objective(self, X, y, n_groups=256, random_state=None):
        """Re-evaluate the fitted objective on fresh Monte Carlo draws.

        The fitted ``q`` (and ``sigma``) are held fixed and the expectation
        over ``q`` is re-estimated with ``n_groups`` new independent groups
        (fresh ``m``-tuples for PACm, including ``m = 1``; fresh
        ``(theta, theta')`` pairs, ``n_groups * n_samples`` of them, for
        PAC^2 / PAC^2_T). The particle ensemble has no draws and returns its
        deterministic objective.

        Returns
        -------
        data, kl, total : float
            Per-datum data term, ``KL / (beta n)`` term and their sum, in
            the same standardized units as ``objective_value_``.
        """
        Z = self._transform(X)
        ys = (np.asarray(y, dtype=float).ravel() - self.y_mean_) / self.y_scale_
        rng = check_random_state(random_state)
        eps = self._fresh_draws(Z.shape[1], n_groups, rng)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            split = self._objective(self.params_, Z, ys, eps, parts=True)
        data, kl = float(split["data"]), float(split["kl"])
        return data, kl, data + kl

    # -- prediction ------------------------------------------------------

    def _components(self, X, include_sigma=False):
        """Locations and scales (original units) of the predictive mixture.

        Parameter-only by default; ``include_sigma=True`` adds the
        likelihood width. The ensemble's parameter-only predictive is a
        discrete mixture of its particle predictions (scale 0).
        """
        Z = self._transform(X)
        s2 = self.sigma_standardized_**2 if include_sigma else 0.0
        if self.objective == "pac2t_ensemble":
            loc = Z @ self.particles_
            scale = np.full_like(loc, np.sqrt(s2))
        else:
            loc = (Z @ self.mu_)[:, None]
            ZL = Z @ self.L_
            scale = np.sqrt(s2 + np.einsum("ij,ij->i", ZL, ZL))[:, None]
        return self.y_mean_ + self.y_scale_ * loc, self.y_scale_ * scale

    def predict_logpdf(self, X, y, include_sigma=False):
        """Log density of the parameter-only predictive (``N(x.mu, x.T
        Sigma x)``), or of the full predictive with ``include_sigma=True``.
        The particle ensemble has no parameter-only density and raises
        unless ``include_sigma=True``."""
        if self.objective == "pac2t_ensemble" and not include_sigma:
            raise ValueError(
                "The particle ensemble's parameter-only predictive is discrete "
                "and has no density; use predict_interval or include_sigma=True."
            )
        loc, scale = self._components(X, include_sigma)
        y = np.asarray(y, dtype=float)[:, None]
        return logsumexp(norm.logpdf(y, loc, scale), axis=1) - np.log(loc.shape[1])

    def predict(self, X, return_std=False, include_sigma=False):
        loc, scale = self._components(X, include_sigma)
        mean = loc.mean(axis=1)
        if not return_std:
            return mean
        second = np.mean(scale**2 + loc**2, axis=1)
        return mean, np.sqrt(np.maximum(second - mean**2, 0.0))

    def predict_cdf(self, X, y, include_sigma=False):
        loc, scale = self._components(X, include_sigma)
        y = np.asarray(y, dtype=float)[:, None]
        if np.all(scale == 0.0):
            return np.mean(loc <= y, axis=1)
        return norm.cdf(y, loc, scale).mean(axis=1)

    def predict_interval(self, X, level=0.9545, include_sigma=False, n_bisect=100):
        """Exact central interval of the predictive (mixture) distribution;
        particle quantiles for the discrete ensemble predictive."""
        loc, scale = self._components(X, include_sigma)
        probs = [0.5 * (1 - level), 0.5 * (1 + level)]
        if np.all(scale == 0.0):
            return tuple(np.quantile(loc, probs, axis=1))
        lo = (loc - 8.0 * scale).min(axis=1)
        hi = (loc + 8.0 * scale).max(axis=1)

        def quantile(prob):
            left, right = lo.copy(), hi.copy()
            for _ in range(n_bisect):
                mid = 0.5 * (left + right)
                below = self.predict_cdf(X, mid, include_sigma) < prob
                left = np.where(below, mid, left)
                right = np.where(below, right, mid)
            return 0.5 * (left + right)

        return quantile(probs[0]), quantile(probs[1])
