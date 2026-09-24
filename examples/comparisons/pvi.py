"""
Predictive variational inference (PVI) for a linear-Gaussian model.

Lai, Linero and Yao, "Predictive variational inference: Learn the
predictively optimal posterior distribution" (arXiv:2410.14843), fit a
parameter distribution ``q_phi`` by maximizing the log score of its
posterior predictive on the training data, with an optional KL
regularization towards the prior,

    max_phi  sum_i log int p(y_i | x_i, theta) q_phi(theta) dtheta
             - lam * KL(q_phi || p_prior).

This is a direct minimization of the *empirical* generalization error
``G_hat`` of the paper, with the KL penalty acting at the parameter level.
For a Gaussian likelihood of width ``sigma`` and ``q = N(m, S)`` the
predictive integral is exact, ``N(y_i | m.x_i, x_i^T S x_i + sigma^2)``, so
this implementation has no Monte Carlo error: it is the ``M -> infinity``
limit of the reparameterization estimator used by Lai et al., i.e. the most
favourable version of PVI for this model class.

As in App. B.5 of Lai et al. the regularization strength ``lam`` can be
chosen by K-fold cross-validation of the predictive log score; the fold
scores use the training likelihood width ``sigma``. Predictions, intervals
and densities are **parameter-only**, ``N(m.x, x^T S x)``, following the
paper's convention that no residual-noise term enters any reported
prediction.
"""

import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class PredictiveVI:
    """Full-rank Gaussian PVI with an exact predictive and analytic gradients.

    Parameters
    ----------
    sigma : float, default=1e-2
        Likelihood width in standardized target units.

    lam : float or 'cv', default='cv'
        KL regularization strength. ``'cv'`` selects it from ``lam_grid`` by
        ``cv_folds``-fold cross-validated predictive log score.

    lam_grid : sequence of float, default=(0.0, 0.01, 0.1, 1.0, 10.0)
        Candidate strengths for ``lam='cv'``.

    cv_folds : int, default=5
        Number of folds (capped at ``n_samples``).

    prior_scale : float, default=10.0
        Prior standard deviation of every standardized coefficient.

    max_iter, tol : int, float
        L-BFGS-B budget and gradient tolerance.

    standardize : bool, default=True
        Center/scale features (constant columns are kept as they are) and
        target.

    random_state : int, default=0
        Seed of the cross-validation fold assignment.

    Attributes
    ----------
    lam_ : float
        Selected regularization strength.

    mu_, L_ : ndarray
        Mean and Cholesky factor of ``q`` in standardized coordinates.

    cv_scores_ : dict
        Mean held-out log score per candidate ``lam`` (``lam='cv'`` only).

    converged_, n_iter_, fit_time_ : bool, int, float
    """

    def __init__(
        self,
        *,
        sigma=1e-2,
        lam="cv",
        lam_grid=(0.0, 0.01, 0.1, 1.0, 10.0),
        cv_folds=5,
        prior_scale=10.0,
        max_iter=5000,
        tol=1e-10,
        standardize=True,
        random_state=0,
    ):
        self.sigma = sigma
        self.lam = lam
        self.lam_grid = lam_grid
        self.cv_folds = cv_folds
        self.prior_scale = prior_scale
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.random_state = random_state

    # -- preprocessing -----------------------------------------------------

    def _standardize_fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if self.standardize:
            self.x_mean_ = X.mean(axis=0)
            self.x_scale_ = X.std(axis=0)
            constant = self.x_scale_ <= 1e-12 * np.maximum(np.abs(self.x_mean_), 1.0)
            self.x_mean_[constant] = 0.0
            self.x_scale_[constant] = 1.0
            self.y_mean_ = float(y.mean())
            self.y_scale_ = float(y.std()) if y.std() > 0 else 1.0
        else:
            self.x_mean_, self.x_scale_ = np.zeros(X.shape[1]), np.ones(X.shape[1])
            self.y_mean_, self.y_scale_ = 0.0, 1.0
        return (X - self.x_mean_) / self.x_scale_, (y - self.y_mean_) / self.y_scale_

    def _transform(self, X):
        return (np.asarray(X, dtype=float) - self.x_mean_) / self.x_scale_

    # -- objective ------------------------------------------------------------

    @staticmethod
    def _unpack(params, p):
        mu = params[:p]
        L = np.zeros((p, p))
        rows, cols = np.tril_indices(p)
        L[rows, cols] = params[p:]
        diag = np.diag_indices(p)
        L[diag] = np.exp(L[diag])
        return mu, L

    def objective(self, params, Z, ys, lam):
        """Negative PVI objective per datum and its gradient.

        ``(1/n) [ -sum_i log N(y_i; z_i.mu, z_i^T S z_i + sigma^2)
        + lam * KL(N(mu, S) || N(0, prior_scale^2 I)) ]`` with ``S = L L^T``.
        """
        n, p = Z.shape
        mu, L = self._unpack(params, p)
        s2 = self.sigma**2
        ZL = Z @ L
        v = np.einsum("ij,ij->i", ZL, ZL) + s2
        r = ys - Z @ mu
        nll = 0.5 * np.sum(np.log(2.0 * np.pi * v) + r * r / v)
        g_mu = -(Z.T @ (r / v))
        g_v = 0.5 / v - 0.5 * r * r / (v * v)
        g_L = 2.0 * (Z.T @ (g_v[:, None] * ZL))

        tau2 = self.prior_scale**2
        diag = np.diag(L)
        kl = 0.5 * (
            np.sum(L * L) / tau2
            + mu @ mu / tau2
            - p
            + p * np.log(tau2)
            - 2.0 * np.sum(np.log(diag))
        )
        g_mu = g_mu + lam * mu / tau2
        g_L = g_L + lam * L / tau2
        g_L[np.diag_indices(p)] -= lam / diag

        rows, cols = np.tril_indices(p)
        g_tril = g_L[rows, cols]
        on_diag = rows == cols
        g_tril[on_diag] *= L[rows[on_diag], cols[on_diag]]  # chain rule, log-diag
        value = (nll + lam * kl) / n
        return value, np.concatenate([g_mu, g_tril]) / n

    def _optimize(self, Z, ys, lam):
        n, p = Z.shape
        mu0 = np.linalg.lstsq(Z.T @ Z + 1e-6 * np.eye(p), Z.T @ ys, rcond=None)[0]
        rows, cols = np.tril_indices(p)
        tril0 = np.where(rows == cols, np.log(0.1), 0.0)
        res = minimize(
            self.objective,
            np.concatenate([mu0, tril0]),
            args=(Z, ys, lam),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )
        mu, L = self._unpack(res.x, p)
        return mu, L, res

    def _log_score(self, mu, L, Z, ys):
        ZL = Z @ L
        v = np.einsum("ij,ij->i", ZL, ZL) + self.sigma**2
        return norm.logpdf(ys, Z @ mu, np.sqrt(v))

    def fit(self, X, y):
        start = time.perf_counter()
        Z, ys = self._standardize_fit(X, y)
        n = ys.size
        if self.lam == "cv":
            k = int(min(self.cv_folds, n))
            folds = np.random.RandomState(self.random_state).permutation(n) % k
            self.cv_scores_ = {}
            for lam in self.lam_grid:
                scores = []
                for f in range(k):
                    train, test = folds != f, folds == f
                    mu, L, _ = self._optimize(Z[train], ys[train], lam)
                    scores.append(self._log_score(mu, L, Z[test], ys[test]))
                self.cv_scores_[float(lam)] = float(np.mean(np.concatenate(scores)))
            self.lam_ = max(self.cv_scores_, key=self.cv_scores_.get)
        else:
            self.lam_ = float(self.lam)
        self.mu_, self.L_, res = self._optimize(Z, ys, self.lam_)
        self.converged_ = bool(res.success)
        self.n_iter_ = int(res.nit)
        self.objective_value_ = float(res.fun)
        self.fit_time_ = time.perf_counter() - start
        return self

    # -- parameter-only prediction ------------------------------------------

    def _loc_scale(self, X):
        Z = self._transform(X)
        ZL = Z @ self.L_
        loc = self.y_mean_ + self.y_scale_ * (Z @ self.mu_)
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
        the matching intercept offsets, ``y = X @ coef + offset``."""
        rng = np.random.RandomState(random_state)
        theta = self.mu_[:, None] + self.L_ @ rng.randn(self.mu_.size, n_samples)
        coef = self.y_scale_ * theta / self.x_scale_[:, None]
        offset = self.y_mean_ - self.x_mean_ @ coef
        return coef, offset
