"""
Standalone Bayesian subset-regression stacking.

The external comparator for POPS: a cross-validated mixture of ordinary
Bayesian linear regressions with proper normal--inverse-gamma priors,
following the predictive-density stacking of Yao, Vehtari, Simpson and
Gelman (2018, Section 3.2). Each component is a *model specification* (a
predeclared subset of the reference feature columns) whose regression
parameters and residual variance are integrated analytically, giving
Student-t predictive densities. Nothing here uses a POPS pre-fit, pointwise
corrections, a covering constraint, an ellipsoid, a PAC prior or a uniform
density floor; the only shared code is the generic convex simplex solver of
:mod:`popsregression._predictive_stacking`.

Low-noise convention: by default every prediction, interval and predictive
score uses the **parameter-only** predictive, the pushforward of the
parameter posterior ``x.theta`` (a Student-t whose scale carries only
``x.T solve(Lambda_n, x)``), exactly as :class:`~popsregression.POPSRegression`
never uses ``alpha_``. The learned residual scale is fitted and reported
(``residual_scale_``) but does not enter predictions unless
``include_residual=True`` is requested explicitly.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import logsumexp
from scipy.stats import t as student_t
from sklearn.model_selection import GroupKFold, KFold
from sklearn.utils import check_random_state

from .stacking_weights import mixture_log_loss, solve_stacking_weights


class BayesianLinearRegression:
    """Conjugate normal--inverse-gamma Bayesian linear regression.

    With design ``X`` (a column subset of the reference features),
    ``theta | sigma2 ~ N(0, sigma2 * inv(Lambda0))`` with
    ``Lambda0 = prior_precision * I`` and ``sigma2 ~ InvGamma(a0, b0)``,
    the posterior is conjugate,

    .. code-block:: text

        Lambda_n = Lambda0 + X.T @ X
        m_n      = solve(Lambda_n, X.T @ y)
        a_n      = a0 + n_rows / 2
        b_n      = b0 + (||y - X m_n||^2 + prior_precision ||m_n||^2) / 2

    (the residual form of ``y.T y - m_n.T Lambda_n m_n``, which is stable
    when ``Lambda_n`` is ill-conditioned). The parameter-only predictive at
    ``x`` (the default everywhere) is Student-t with ``2 a_n`` degrees of
    freedom, location ``x.T m_n`` and squared scale ``(b_n / a_n) x.T
    solve(Lambda_n, x)``; the full posterior predictive adds the residual
    variance, ``(b_n / a_n) (1 + x.T solve(Lambda_n, x))``, and is only
    used when ``include_residual=True``. With ``standardize=True`` features
    and targets are divided by their root-mean-square on the fitting data,
    never centered, so no implicit intercept is added to the design shared
    with the other methods; predictive densities in original units include
    the target Jacobian.

    Parameters
    ----------
    columns : array-like of int, default=None
        Columns of the reference design defining this component; all
        columns if None.

    prior_precision : float, default=1e-2
        ``Lambda0 = prior_precision * I`` (in standardized units).

    a0, b0 : float, default=2.0, 1e-2
        Inverse-gamma prior of the residual variance in standardized target
        units (prior mean ``b0 / (a0 - 1)``, i.e. a prior residual standard
        deviation of 10% of the target scale). Weakly informative but
        proper; the fitted residual scale stays positive on deterministic
        data.

    standardize : bool, default=True
        Standardize non-constant features and the targets on the fitting
        data (fold-internal when used inside cross-validation).
    """

    def __init__(
        self, columns=None, *, prior_precision=1e-2, a0=2.0, b0=1e-2, standardize=True
    ):
        self.columns = columns
        self.prior_precision = prior_precision
        self.a0 = a0
        self.b0 = b0
        self.standardize = standardize

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        cols = slice(None) if self.columns is None else np.asarray(self.columns, int)
        return X[:, cols]

    def fit(self, X, y):
        """Fit the conjugate posterior."""
        if not (self.prior_precision > 0 and self.a0 > 0 and self.b0 > 0):
            raise ValueError("prior_precision, a0 and b0 must be positive.")
        Xs = self._design(X)
        y = np.asarray(y, dtype=float).ravel()
        n, p = Xs.shape
        if n == 0 or y.shape[0] != n:
            raise ValueError("X and y must be non-empty with matching rows.")
        if self.standardize:
            # Scale by the root-mean-square, never center: centering would add
            # an implicit intercept that the other methods, fitted on the same
            # design matrix, do not have.
            rms = np.sqrt(np.mean(Xs * Xs, axis=0))
            self.x_scale_ = np.where(rms > 0, rms, 1.0)
            self.x_mean_ = np.zeros_like(self.x_scale_)
            self.y_mean_ = 0.0
            self.y_scale_ = float(np.sqrt(np.mean(y * y))) or 1.0
        else:
            self.x_mean_ = np.zeros(p)
            self.x_scale_ = np.ones(p)
            self.y_mean_, self.y_scale_ = 0.0, 1.0
        Z = (Xs - self.x_mean_) / self.x_scale_
        ys = (y - self.y_mean_) / self.y_scale_

        Lambda_n = Z.T @ Z
        Lambda_n[np.diag_indices_from(Lambda_n)] += self.prior_precision
        self._chol = cho_factor(Lambda_n, lower=True)
        self.m_n_ = cho_solve(self._chol, Z.T @ ys)
        resid = ys - Z @ self.m_n_
        self.a_n_ = self.a0 + 0.5 * n
        self.b_n_ = self.b0 + 0.5 * (
            float(resid @ resid) + self.prior_precision * float(self.m_n_ @ self.m_n_)
        )
        self.df_ = 2.0 * self.a_n_
        self.n_samples_ = n
        # Posterior-mean coefficients in original units.
        coef_s = self.m_n_ / self.x_scale_
        self.coef_ = self.y_scale_ * coef_s
        self.intercept_ = self.y_mean_ - self.y_scale_ * float(coef_s @ self.x_mean_)
        self.residual_scale_ = self.y_scale_ * np.sqrt(
            self.b_n_ / (self.a_n_ - 1.0) if self.a_n_ > 1.0 else self.b_n_ / self.a_n_
        )
        return self

    def _predictive(self, X):
        """Location, full scale and parameter-only scale in original units."""
        Z = (self._design(X) - self.x_mean_) / self.x_scale_
        loc = Z @ self.m_n_
        quad = np.einsum("ij,ij->i", Z, cho_solve(self._chol, Z.T).T)
        s2_param = (self.b_n_ / self.a_n_) * quad
        s2_full = (self.b_n_ / self.a_n_) * (1.0 + quad)
        return (
            self.y_mean_ + self.y_scale_ * loc,
            self.y_scale_ * np.sqrt(s2_full),
            self.y_scale_ * np.sqrt(s2_param),
        )

    def _scale(self, X, include_residual):
        loc, scale_full, scale_param = self._predictive(X)
        return loc, (scale_full if include_residual else scale_param)

    def predict_logpdf(self, X, y, include_residual=False):
        """Log Student-t predictive density in original units
        (parameter-only unless ``include_residual=True``)."""
        loc, scale = self._scale(X, include_residual)
        y = np.asarray(y, dtype=float)
        return student_t.logpdf((y - loc) / scale, self.df_) - np.log(scale)

    def predict_cdf(self, X, y, include_residual=False):
        loc, scale = self._scale(X, include_residual)
        return student_t.cdf((np.asarray(y, dtype=float) - loc) / scale, self.df_)

    def predict(self, X, return_std=False, include_residual=False):
        """Predictive mean and standard deviation.

        The default is the parameter-only (location) uncertainty;
        ``include_residual=True`` adds the learned residual variance.
        """
        loc, scale, scale_param = self._predictive(X)
        if not return_std:
            return loc
        var_factor = self.df_ / (self.df_ - 2.0) if self.df_ > 2.0 else np.inf
        s = scale if include_residual else scale_param
        return loc, s * np.sqrt(var_factor)

    def sample(self, n_samples, random_state=None):
        """Joint posterior draws ``(sigma2, theta)`` in original units.

        Returns
        -------
        coef : ndarray of shape (n_reference_features, n_samples)
            Coefficients embedded in the reference feature space (zero on
            columns this component does not use).

        intercept : ndarray of shape (n_samples,)
            Intercept correction from the standardization.

        sigma2 : ndarray of shape (n_samples,)
            Residual variance draws in original units.
        """
        rng = check_random_state(random_state)
        p = self.m_n_.size
        sigma2 = self.b_n_ / rng.gamma(self.a_n_, 1.0, size=n_samples)
        z = rng.randn(p, n_samples) * np.sqrt(sigma2)[None, :]
        L = np.tril(self._chol[0]) if self._chol[1] else np.triu(self._chol[0]).T
        theta_s = self.m_n_[:, None] + solve_triangular(L.T, z, lower=False)
        coef_s = theta_s / self.x_scale_[:, None]
        coef = self.y_scale_ * coef_s
        intercept = self.y_mean_ - self.y_scale_ * (self.x_mean_ @ coef_s)
        return coef, intercept, self.y_scale_**2 * sigma2


class BayesianStacking:
    """Cross-validated stacking of Bayesian subset regressions.

    Parameters
    ----------
    components : list of array-like of int
        Predeclared column subsets of the reference design, one per
        component (``None`` entries mean all columns). Include the full
        reference surrogate; no component may exceed it.

    cv : {'loo'} or int, default='loo'
        Exact leave-one-out refits (``'loo'``) or the number of folds.
        With ``group_ids`` the folds are grouped (``'loo'`` then means
        leave-one-group-out): a simulator case or structure is indivisible.

    include_residual : bool, default=False
        If False (the low-noise convention), the out-of-fold scores, the
        stacking weights and every prediction use the parameter-only
        predictive of each component; the residual scales are still fitted
        and reported. If True the full posterior predictive is used
        throughout.

    prior_precision, a0, b0, standardize
        Component priors and fold-internal preprocessing, see
        :class:`BayesianLinearRegression`.

    random_state : int, default=0
        Seed of the fold shuffling and of parameter draws.

    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
        Stacking weights (global, nonnegative, summing to one).

    weights_by_rule_ : dict
        ``'stacking'``, ``'best'`` (point mass on the best
        cross-validated component) and ``'uniform'``.

    oof_log_density_ : ndarray of shape (n_samples, n_components)
        Out-of-fold log predictive densities.

    cv_scores_ : ndarray of shape (n_components,)
        Out-of-fold mean negative log density per component (grouped
        averaging when ``group_ids`` were given).

    mixture_cv_scores_ : dict
        The same score of the three mixtures.

    solver_status_ : dict
        Stacking solver provenance (method, success, KKT residual).

    components_ : list of BayesianLinearRegression
        Components refitted on the full training sample.

    residual_scales_ : ndarray of shape (n_components,)
        Posterior-mean residual scales of the refitted components.

    fit_time_ : float
        Seconds spent in cross-validation, weight solving and refitting.
    """

    def __init__(
        self,
        components,
        *,
        cv="loo",
        include_residual=False,
        prior_precision=1e-2,
        a0=2.0,
        b0=1e-2,
        standardize=True,
        random_state=0,
    ):
        self.components = components
        self.cv = cv
        self.include_residual = include_residual
        self.prior_precision = prior_precision
        self.a0 = a0
        self.b0 = b0
        self.standardize = standardize
        self.random_state = random_state

    def _make_component(self, columns):
        return BayesianLinearRegression(
            columns,
            prior_precision=self.prior_precision,
            a0=self.a0,
            b0=self.b0,
            standardize=self.standardize,
        )

    def _folds(self, n, group_ids):
        if group_ids is None:
            if self.cv == "loo":
                return list(KFold(n_splits=n).split(np.zeros(n)))
            return list(
                KFold(
                    n_splits=int(self.cv), shuffle=True, random_state=self.random_state
                ).split(np.zeros(n))
            )
        groups = np.asarray(group_ids)
        n_groups = np.unique(groups).size
        n_splits = n_groups if self.cv == "loo" else min(int(self.cv), n_groups)
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(n), groups=groups))

    def fit(self, X, y, group_ids=None):
        """Cross-validate, solve the stacking weights and refit."""
        t0 = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = y.shape[0]
        if X.ndim != 2 or X.shape[0] != n or n < 2:
            raise ValueError("X must be 2d with one row per entry of y (n >= 2).")
        n_ref = X.shape[1]
        for cols in self.components:
            if cols is not None and np.asarray(cols).max() >= n_ref:
                raise ValueError("A component references a column outside X.")
        K = len(self.components)
        oof = np.full((n, K), np.nan)
        for train, test in self._folds(n, group_ids):
            for k, cols in enumerate(self.components):
                model = self._make_component(cols).fit(X[train], y[train])
                oof[test, k] = model.predict_logpdf(
                    X[test], y[test], self.include_residual
                )
        if not np.all(np.isfinite(oof)):
            raise RuntimeError("Non-finite out-of-fold log densities.")
        if group_ids is None:
            row_weights = np.full(n, 1.0 / n)
        else:
            _, inverse, counts = np.unique(
                np.asarray(group_ids), return_inverse=True, return_counts=True
            )
            inverse = inverse.ravel()
            row_weights = 1.0 / (counts.size * counts[inverse])
        solution = solve_stacking_weights(oof, row_weights)
        self.oof_log_density_ = oof
        self.row_weights_ = row_weights
        self.weights_ = solution.weights
        self.solver_status_ = solution.status
        self.cv_scores_ = -(row_weights @ oof)
        best = np.zeros(K)
        best[int(np.argmin(self.cv_scores_))] = 1.0
        self.weights_by_rule_ = {
            "stacking": self.weights_,
            "best": best,
            "uniform": np.full(K, 1.0 / K),
        }
        self.mixture_cv_scores_ = {
            name: mixture_log_loss(q, oof, row_weights)[0]
            for name, q in self.weights_by_rule_.items()
        }
        self.components_ = [
            self._make_component(cols).fit(X, y) for cols in self.components
        ]
        self.residual_scales_ = np.array([c.residual_scale_ for c in self.components_])
        self.n_features_in_ = n_ref
        self.fit_time_ = time.perf_counter() - t0
        return self

    def _weights(self, weights):
        if weights is None:
            return self.weights_
        if isinstance(weights, str):
            return self.weights_by_rule_[weights]
        q = np.asarray(weights, dtype=float)
        if (
            q.shape != self.weights_.shape
            or np.any(q < 0)
            or not np.isclose(q.sum(), 1)
        ):
            raise ValueError("weights must be a probability vector over components.")
        return q

    def _residual(self, include_residual):
        return self.include_residual if include_residual is None else include_residual

    def _component_logpdf(self, X, y, include_residual):
        return np.column_stack(
            [c.predict_logpdf(X, y, include_residual) for c in self.components_]
        )

    def predict_logpdf(self, X, y, weights=None, include_residual=None):
        """Log density of the stacked predictive mixture (parameter-only by
        default, see ``include_residual``)."""
        q = self._weights(weights)
        log_p = self._component_logpdf(X, y, self._residual(include_residual))
        with np.errstate(divide="ignore"):
            return logsumexp(log_p + np.log(q), axis=1)

    def predict(self, X, return_std=False, weights=None, include_residual=None):
        """Mean and standard deviation of the stacked mixture."""
        q = self._weights(weights)
        include_residual = self._residual(include_residual)
        means, stds = [], []
        for c in self.components_:
            m, s = c.predict(X, return_std=True, include_residual=include_residual)
            means.append(m)
            stds.append(s)
        means, stds = np.array(means).T, np.array(stds).T
        mean = means @ q
        if not return_std:
            return mean
        second = (stds**2 + means**2) @ q
        return mean, np.sqrt(np.maximum(second - mean**2, 0.0))

    def predict_cdf(self, X, y, weights=None, include_residual=None):
        q = self._weights(weights)
        include_residual = self._residual(include_residual)
        return (
            np.column_stack(
                [c.predict_cdf(X, y, include_residual) for c in self.components_]
            )
            @ q
        )

    def predict_interval(
        self, X, level=0.9545, weights=None, include_residual=None, n_bisect=100
    ):
        """Exact central interval of the mixture (a t mixture is not t)."""
        q = self._weights(weights)
        include_residual = self._residual(include_residual)
        locs, scales = [], []
        for c in self.components_:
            loc, scale = c._scale(X, include_residual)
            locs.append(loc)
            scales.append(scale * student_t.ppf(1.0 - 1e-9, c.df_))
        locs, scales = np.array(locs), np.array(scales)
        lo, hi = (locs - scales).min(axis=0), (locs + scales).max(axis=0)

        def quantile(prob):
            left, right = lo.copy(), hi.copy()
            for _ in range(n_bisect):
                mid = 0.5 * (left + right)
                below = self.predict_cdf(X, mid, q, include_residual) < prob
                left = np.where(below, mid, left)
                right = np.where(below, right, mid)
            return 0.5 * (left + right)

        return quantile(0.5 * (1.0 - level)), quantile(0.5 * (1.0 + level))

    def sample_parameters(self, n_samples, weights=None, random_state=None):
        """One component and one parameter vector per draw.

        Returns
        -------
        coef : ndarray of shape (n_reference_features, n_samples)
            Coefficients embedded in the reference feature space.

        intercept : ndarray of shape (n_samples,)

        component : ndarray of shape (n_samples,)

        sigma2 : ndarray of shape (n_samples,)
            Residual variance draws; a scalar residual variance is
            pointwise predictive noise, not a coherent spatial field.
        """
        q = self._weights(weights)
        rng = check_random_state(
            self.random_state if random_state is None else random_state
        )
        component = rng.choice(q.size, size=n_samples, p=q)
        coef = np.zeros((self.n_features_in_, n_samples))
        intercept = np.zeros(n_samples)
        sigma2 = np.zeros(n_samples)
        for k in np.unique(component):
            sel = component == k
            c = self.components_[k]
            sub_coef, sub_int, sub_s2 = c.sample(int(sel.sum()), rng)
            cols = (
                np.arange(self.n_features_in_)
                if self.components[k] is None
                else np.asarray(self.components[k], int)
            )
            coef[np.ix_(cols, np.flatnonzero(sel))] = sub_coef
            intercept[sel] = sub_int
            sigma2[sel] = sub_s2
        return coef, intercept, component, sigma2
