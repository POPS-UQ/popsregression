"""
Predictive variational inference (PVI), following the published code.

Lai, Linero and Yao, "Predictive variational inference: Learn the
predictively optimal posterior distribution" (arXiv:2410.14843, ICML 2026);
reference implementation https://github.com/lll6924/pvi. This module ports,
for a linear model ``y ~ N(theta . x, sigma^2)``, exactly the pieces of that
code used for the log score:

- objective ``PACMVIBasic``: at every step ``s`` fresh reparameterized draws
  ``theta_j ~ q`` and ``sum_i [logsumexp_j log p(y_i | theta_j) - log s]``;
- regularizer ``KLPrior``: ``mean_j [log p_prior(theta_j) - log q(theta_j)]``
  on an independent set of ``s`` draws, weighted by ``lamb``;
- families ``Basic`` (mean-field Gaussian, parameters ``(loc, log scale)``)
  and ``BasicFullRank`` (Cholesky factor with log diagonal floored at 1e-2),
  both initialized at zero;
- the training loop of ``run/run.py``: gradient ascent with the gradient
  divided by ``n``, rescaled to norm 100 when its (undivided) norm exceeds
  100, steps with NaN gradients skipped, and the learning rate divided by 10
  after half of the iterations; optimizers ``sgd`` (default), ``nesterov``
  (mass 0.9) and ``rmsprop`` (``jax.example_libraries.optimizers.
  rmsprop_momentum`` with its defaults). The last iterate is returned.

Only the gradient computation differs from the JAX original: the gradients
of the same Monte Carlo objectives are written out analytically. Features
and target are divided by their root-mean-square (never centered, so no
implicit intercept is added to the shared design matrix) so that the
published learning rates apply; ``sigma`` is given in units of the target
standard deviation, and the prior is ``N(0, prior_scale^2 I)`` on the
rescaled coefficients.

With the Monte Carlo log score this objective is the PACm objective of
Morningstar et al. (2022) with KL weight ``lamb`` (the published class is
named ``PACMVIBasic``). Predictions, intervals and densities are
**parameter-only**, ``N(x . mu, x^T Sigma x)``, following the paper's
convention that no residual-noise term enters any reported prediction.
"""

import time

import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import norm


class PredictiveVI:
    """Published PVI (log score) for a linear-Gaussian model.

    Parameters
    ----------
    sigma : float, default=1e-2
        Likelihood width in units of the training-target standard deviation.

    lamb : float, default=0.0
        Strength of the ``KLPrior`` regularizer (the published default).

    s : int, default=1
        Number of ``theta`` draws per step in the objective and regularizer
        (the published default).

    posterior : {'Basic', 'BasicFullRank'}, default='Basic'
        Variational family.

    iterations : int, default=1000
        Training iterations (the published default).

    learning_rate : float, default=1e-3
        Learning rate (the published default), divided by 10 after half of
        the iterations.

    optimizer : {'sgd', 'nesterov', 'rmsprop'}, default='sgd'

    prior_scale : float, default=10.0
        Prior standard deviation of every rescaled coefficient.

    standardize : bool, default=True

    random_state : int, default=0
        Seed of the draws.

    Attributes
    ----------
    mu_, L_ : ndarray
        Mean and Cholesky factor of ``q`` in rescaled coordinates.

    n_skipped_ : int
        Steps skipped because of NaN gradients.

    fit_time_ : float
    """

    def __init__(
        self,
        *,
        sigma=1e-2,
        lamb=0.0,
        s=1,
        posterior="Basic",
        iterations=1000,
        learning_rate=1e-3,
        optimizer="sgd",
        prior_scale=10.0,
        standardize=True,
        random_state=0,
    ):
        self.sigma = sigma
        self.lamb = lamb
        self.s = s
        self.posterior = posterior
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.prior_scale = prior_scale
        self.standardize = standardize
        self.random_state = random_state

    # -- preprocessing -----------------------------------------------------

    def _standardize_fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if self.standardize:
            # Scale by the root-mean-square, never center: centering would add
            # an implicit intercept that the other methods, fitted on the same
            # design matrix, do not have.
            rms = np.sqrt(np.mean(X * X, axis=0))
            self.x_scale_ = np.where(rms > 0, rms, 1.0)
            self.y_scale_ = float(np.sqrt(np.mean(y * y))) or 1.0
            self._sigma_unit = float(y.std()) / self.y_scale_ if y.std() > 0 else 1.0
        else:
            self.x_scale_ = np.ones(X.shape[1])
            self.y_scale_ = 1.0
            self._sigma_unit = 1.0
        self.x_mean_ = np.zeros_like(self.x_scale_)
        self.y_mean_ = 0.0
        return X / self.x_scale_, y / self.y_scale_

    def _transform(self, X):
        return np.asarray(X, dtype=float) / self.x_scale_

    # -- variational family --------------------------------------------------

    def _n_params(self, p):
        if self.posterior == "Basic":
            return 2 * p
        return p + p * (p + 1) // 2

    def _factor(self, params, p):
        """Location, Cholesky factor and the log-diagonal activity mask."""
        loc = params[:p]
        if self.posterior == "Basic":
            return loc, np.diag(np.exp(params[p:])), None
        scale = params[p:]
        L = np.zeros((p, p))
        rows, cols = np.tril_indices(p, k=-1)
        L[rows, cols] = scale[p:]
        diag = np.exp(scale[:p])
        active = diag > 1e-2
        L[np.diag_indices(p)] = np.maximum(1e-2, diag)
        return loc, L, active

    def _theta_grad_to_params(self, g_theta, u, params, p):
        """Chain ``d/dtheta_j`` (s, p) through ``theta_j = loc + L u_j``."""
        g_loc = g_theta.sum(axis=0)
        if self.posterior == "Basic":
            scale = np.exp(params[p:])
            return np.concatenate([g_loc, (g_theta * u).sum(axis=0) * scale])
        _, L, active = self._factor(params, p)
        G = g_theta.T @ u  # d/dL_ab = sum_j g_ja u_jb
        diag = np.where(active, np.diag(G) * np.diag(L), 0.0)
        rows, cols = np.tril_indices(p, k=-1)
        return np.concatenate([g_loc, diag, G[rows, cols]])

    # -- the two published objectives and their gradients -------------------

    def _width(self):
        return self.sigma * getattr(self, "_sigma_unit", 1.0)

    def _log_score(self, params, Z, ys, u):
        """``PACMVIBasic`` value and gradient for fixed standard draws ``u``."""
        p = Z.shape[1]
        loc, L, _ = self._factor(params, p)
        theta = loc[None, :] + u @ L.T  # (s, p)
        resid = ys[:, None] - Z @ theta.T  # (n, s)
        s2 = self._width() ** 2
        lls = -0.5 * np.log(2 * np.pi * s2) - 0.5 * resid**2 / s2
        value = float(np.sum(logsumexp(lls, axis=1) - np.log(u.shape[0])))
        w = softmax(lls, axis=1)  # (n, s)
        g_theta = (w * resid / s2).T @ Z  # (s, p)
        return value, self._theta_grad_to_params(g_theta, u, params, p)

    def _kl_prior(self, params, u):
        """``KLPrior``: ``mean_j log p_prior(theta_j) - log q(theta_j)``."""
        p = u.shape[1]
        loc, L, active = self._factor(params, p)
        theta = loc[None, :] + u @ L.T
        tau2 = self.prior_scale**2
        log_prior = -0.5 * np.sum(theta**2, axis=1) / tau2 - 0.5 * p * np.log(
            2 * np.pi * tau2
        )
        log_q = (
            -0.5 * np.sum(u**2, axis=1)
            - np.sum(np.log(np.diag(L)))
            - 0.5 * p * np.log(2 * np.pi)
        )
        value = float(np.mean(log_prior - log_q))
        grad = self._theta_grad_to_params(-theta / tau2, u, params, p) / u.shape[0]
        # -log q contributes +d/dparams sum_k log L_kk (the path term cancels).
        if self.posterior == "Basic":
            grad[p:] += 1.0
        else:
            grad[p : 2 * p] += np.where(active, 1.0, 0.0)
        return value, grad

    # -- training loop of run/run.py -----------------------------------------

    def fit(self, X, y):
        start = time.perf_counter()
        Z, ys = self._standardize_fit(X, y)
        n, p = Z.shape
        rng = np.random.RandomState(self.random_state)
        params = np.zeros(self._n_params(p))
        state = {"v": np.zeros_like(params), "m": np.zeros_like(params)}
        self.n_skipped_ = 0
        self.values_ = []
        for step in range(self.iterations):
            lr = self.learning_rate
            if step >= self.iterations // 2:
                lr = self.learning_rate / 10
            value, grads = self._log_score(params, Z, ys, rng.randn(self.s, p))
            if self.lamb != 0:
                v2, g2 = self._kl_prior(params, rng.randn(self.s, p))
                value, grads = value + self.lamb * v2, grads + self.lamb * g2
            norm_ = np.linalg.norm(grads)
            grads = grads / n
            if norm_ > 100:
                grads = grads / norm_ * 100
            if np.any(np.isnan(grads)):
                self.n_skipped_ += 1
                continue
            params = self._ascend(params, grads, lr, state)
            self.values_.append(value)
        self.mu_, self.L_, _ = self._factor(params, p)
        self.params_ = params
        self.converged_ = bool(np.all(np.isfinite(params)))
        self.n_iter_ = self.iterations
        self.fit_time_ = time.perf_counter() - start
        return self

    def _ascend(self, x, g, lr, state):
        """One ascent step: the jax.example_libraries update applied to -g."""
        if self.optimizer == "sgd":
            return x + lr * g
        if self.optimizer == "nesterov":
            state["v"] = 0.9 * state["v"] - g
            return x - lr * (0.9 * state["v"] - g)
        if self.optimizer == "rmsprop":
            state["v"] = 0.9 * state["v"] + 0.1 * g * g
            state["m"] = 0.9 * state["m"] - lr * g / np.sqrt(state["v"] + 1e-8)
            return x - state["m"]
        raise ValueError(f"unknown optimizer {self.optimizer!r}")

    # -- parameter-only prediction ------------------------------------------

    def _loc_scale(self, X):
        Z = self._transform(X)
        ZL = Z @ self.L_
        loc = self.y_scale_ * (Z @ self.mu_)
        return loc, self.y_scale_ * np.sqrt(np.einsum("ij,ij->i", ZL, ZL))

    def predict(self, X, return_std=False):
        loc, scale = self._loc_scale(X)
        return (loc, scale) if return_std else loc

    def predict_logpdf(self, X, y):
        loc, scale = self._loc_scale(X)
        return norm.logpdf(np.asarray(y, dtype=float), loc, scale)

    def predict_cdf(self, X, y):
        loc, scale = self._loc_scale(X)
        return norm.cdf(np.asarray(y, dtype=float), loc, scale)

    def predict_interval(self, X, level=0.9545):
        loc, scale = self._loc_scale(X)
        z = norm.ppf(0.5 * (1.0 + level))
        return loc - z * scale, loc + z * scale

    def sample_parameters(self, n_samples, random_state=None):
        """Coefficient draws in original feature units (rows: features) and
        zero offsets, ``y = X @ coef + offset``."""
        rng = np.random.RandomState(random_state)
        theta = self.mu_[:, None] + self.L_ @ rng.randn(self.mu_.size, n_samples)
        return self.y_scale_ * theta / self.x_scale_[:, None], np.zeros(n_samples)
