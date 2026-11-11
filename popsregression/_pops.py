"""
POPS (Pointwise Optimal Parameter Sets) Regression.

Bayesian regression for low-noise data accounting for model misspecification.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

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
from ._bayes import as_nan, bayesian_ridge, scaled_covariance

_RESAMPLING = {"latin": qmc.LatinHypercube, "sobol": qmc.Sobol, "halton": qmc.Halton}


@jax.jit
def pointwise_corrections(X, y, weights, fit, leverage_regularization):
    """Pointwise optimal parameter corrections with regularized leverage.

    The correction of training point ``i`` is the smallest parameter move, in
    the metric ``A = X^T W X + lambda / alpha`` of the mean fit, after which
    the model passes through ``y_i``. In closed form it is ``e_i A^-1 x_i /
    h_i`` with residual ``e_i`` and leverage ``h_i = x_i^T A^-1 x_i``, which
    diverges for points of vanishing leverage. Asking instead for the
    minimizer of ``w_i (x_i @ d - e_i)**2 / eps + d^T A d`` gives

        d_i = e_i A^-1 x_i * w_i / (w_i h_i + eps),

    which is bounded: ``||d_i||_A <= |e_i| sqrt(w_i / eps) / 2``. ``eps`` is
    ``leverage_regularization`` times the mean (weighted) leverage, so the
    regularization is dimensionless and only acts on points far less
    informative than the typical one.

    Parameters
    ----------
    X, y, weights : arrays of ``n_samples`` rows
        Design, targets and non-negative sample weights.
    fit : RidgeFit
        The mean fit.
    leverage_regularization : float
        Regularization, in units of the mean leverage.

    Returns
    -------
    corrections : array of shape (n_samples, n_modes)
        Corrections in the right singular basis ``fit.V`` of the design (an
        orthonormal basis, so row norms are the Euclidean ``||d_i||``).
    errors : array of shape (n_samples,)
        Residuals of the mean fit.
    leverage : array of shape (n_samples,)
        Weighted leverages ``w_i h_i``.
    """
    T = X @ fit.V
    inverse_gram = 1.0 / (fit.S**2 + fit.lam / fit.alpha)
    errors = y - X @ fit.coef
    leverage = weights * jnp.sum(T * T * inverse_gram, axis=1)
    active = weights > 0
    mean_leverage = jnp.sum(jnp.where(active, leverage, 0.0)) / jnp.sum(active)
    denominator = leverage + leverage_regularization * mean_leverage
    scale = jnp.where(denominator > 0, weights * errors / denominator, 0.0)
    return (scale[:, None] * T) * inverse_gram, errors, leverage


@jax.jit
def _select_and_bound(corrections, active, threshold, clipping):
    """Prune negligible corrections, then fit the PCA-aligned box.

    Points are kept when their Euclidean correction norm ``||d_i||`` is at
    least ``threshold`` times the root-mean-square norm: these are the rows
    that determine the singular decomposition of the correction matrix.

    Returns the selection mask, the eigenvalues and eigenvectors of the
    (pruned) second-moment matrix and the box bounds along every eigenvector.
    """
    norm2 = jnp.sum(corrections**2, axis=1)
    mean_norm2 = jnp.sum(jnp.where(active, norm2, 0.0)) / jnp.sum(active)
    keep = active & (norm2 >= threshold**2 * mean_norm2)
    keep = jnp.where(jnp.any(keep), keep, active)
    kept = jnp.where(keep[:, None], corrections, 0.0)
    eigenvalues, eigenvectors = jnp.linalg.eigh(kept.T @ kept)
    projected = jnp.where(keep[:, None], corrections @ eigenvectors, jnp.nan)
    low = jnp.nanpercentile(projected, clipping, axis=0)
    high = jnp.nanpercentile(projected, 100.0 - clipping, axis=0)
    return keep, eigenvalues, eigenvectors, low, high


class POPSRegression(BaseEstimator):
    """Bayesian regression for low-noise data with misspecification uncertainty.

    Fits a linear model by Bayesian ridge regression, then estimates weight
    uncertainties accounting for model misspecification using the POPS
    (Pointwise Optimal Parameter Sets) algorithm [1]_. Unlike standard
    Bayesian regression, the aleatoric noise precision ``alpha_`` is not
    used for predictions, as it should be negligible in the low-noise regime.

    Standard Bayesian regression can only estimate epistemic and aleatoric
    uncertainties. In the low-noise limit, the weight uncertainties
    ``sigma_`` are significantly underestimated as they only account for
    epistemic uncertainties that decay with increasing data. POPS corrects
    this by estimating misspecification uncertainty from pointwise optimal
    parameter sets.

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of evidence iterations of the Bayesian ridge fit.

    tol : float, default=1e-3
        Convergence threshold. Stop the evidence iterations once the
        coefficient vector has converged.

    alpha_1 : float, default=1e-6
        Shape parameter for the Gamma distribution prior over ``alpha_``.

    alpha_2 : float, default=1e-6
        Inverse scale (rate) parameter for the Gamma distribution prior
        over ``alpha_``.

    lambda_1 : float, default=1e-6
        Shape parameter for the Gamma distribution prior over ``lambda_``.

    lambda_2 : float, default=1e-6
        Inverse scale (rate) parameter for the Gamma distribution prior
        over ``lambda_``.

    alpha_init : float, default=None
        Initial value for ``alpha_`` (precision of the noise). If None,
        ``alpha_init`` is ``1 / Var(y)``.

    lambda_init : float, default=None
        Initial value for ``lambda_`` (precision of the weights). If None,
        ``lambda_init`` is 1.

    fit_intercept : bool, default=False
        Whether to fit an intercept. If True, a constant column is appended
        to X (rather than centering) so that the intercept participates in
        the POPS posterior estimation.

    mode_threshold : float, default=1e-8
        Eigenvalue threshold (relative to max) for determining the effective
        dimensionality of the POPS posterior. Eigenvalues below
        ``mode_threshold * max_eigenvalue`` are discarded.

    resample_density : float, default=1.0
        Number of resampled points per training point. The actual number of
        samples is ``max(100, int(resample_density * n_samples))``.

    resampling_method : {'uniform', 'sobol', 'latin', 'halton'}, \
            default='uniform'
        Quasi-random sampling method for generating points within the
        POPS hypercube posterior.

    percentile_clipping : float, default=0.0
        Percentile to clip from each end when determining hypercube bounds.
        The hypercube spans the ``[percentile_clipping,
        100 - percentile_clipping]`` range. Should be between 0 and 50.

    leverage_regularization : float, default=1e-3
        Regularization of the leverage in the pointwise corrections, in
        units of the mean leverage of the training points. The correction
        of point ``i`` is ``e_i A^-1 x_i / (h_i + eps)`` with ``eps =
        leverage_regularization * mean(h)``: the exact pointwise fit
        (``eps = 0``) divides by the leverage ``h_i`` alone and is unstable
        for points the model is barely sensitive to. With ``eps > 0`` the
        correction is the minimizer of ``(x_i @ d - e_i)**2 / eps + d^T A
        d``, so the pointwise fit is satisfied up to the relative residual
        ``eps / (h_i + eps)``: about ``leverage_regularization`` for a
        typical point, and a large fraction only for points of leverage
        a thousand times below the mean at the default. ``0.0`` recovers
        the exact pointwise fit. The corrections of low-leverage points can
        dominate the posterior, so on designs whose leverages span many
        orders of magnitude this parameter sets its width: see
        ``leverage_``.

    minimum_relative_correction : float, default=1e-2
        Relative threshold for training point selection, in units of the
        root-mean-square Euclidean norm of the pointwise corrections. Only
        training points with ``||d_i|| >= minimum_relative_correction *
        rms(||d||)`` enter the singular decomposition and bounds of the
        POPS posterior; the others are too small to affect either. Use
        ``0.0`` to keep every training point. If no point passes the
        threshold, all points are used.

    posterior : {'hypercube', 'ensemble'}, default='hypercube'
        Form of the POPS parameter posterior:

        - ``'hypercube'``: fit an axis-aligned box in PCA space (default).
        - ``'ensemble'``: use raw pointwise corrections directly.

        The uniform-ellipsoid posterior, with its empirical-Bayes and PAC
        regularizations, is :class:`~popsregression.POPSEllipseRegression`.

    random_state : int, RandomState instance or None, default=None
        Seed for the posterior resampling. ``None`` uses the global NumPy
        random state for ``'uniform'`` resampling and fresh entropy for the
        quasi-random methods.

    minimum_relative_error : float, default='deprecated'
        Deprecated. Training points used to be selected by the size of
        their residual; they are now selected by the size of their
        pointwise correction through ``minimum_relative_correction``, to
        which a value passed here is forwarded with a
        :class:`FutureWarning`.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficients of the regression model (posterior mean).

    intercept_ : float
        Independent term in the decision function. Set to 0.0 if
        ``fit_intercept=False``.

    alpha_ : float
        Estimated precision of the noise. Not used for prediction.

    lambda_ : float
        Estimated precision of the weights.

    sigma_ : ndarray of shape (n_features, n_features)
        Estimated epistemic variance-covariance matrix of the weights. If
        ``fit_intercept=True`` the last row and column are the intercept.

    misspecification_sigma_ : ndarray of shape (n_features, n_features)
        Estimated misspecification variance-covariance matrix from POPS.

    posterior_samples_ : ndarray of shape (n_features, n_posterior_samples)
        Samples from the POPS posterior, representing plausible weight
        perturbations.

    pointwise_corrections_ : ndarray of shape (n_samples, n_features)
        Pointwise corrections of every training point (lazy).

    leverage_ : ndarray of shape (n_samples,)
        Leverage of every training point in the mean fit.

    selected_ : ndarray of shape (n_samples,), dtype=bool
        Training points that entered the POPS posterior.

    n_iter_ : int
        The actual number of iterations to reach convergence.

    n_features_in_ : int
        Number of features seen during ``fit``.

    See Also
    --------
    popsregression.POPSEllipseRegression : Uniform-ellipsoid POPS posterior
        with an exact predictive density.

    References
    ----------
    .. [1] Swinburne, T.D. and Perez, D. (2025).
           "Parameter uncertainties for imperfect surrogate models in the
           low-noise regime."
           Machine Learning: Science and Technology, 6, 015008.
           :doi:`10.1088/2632-2153/ad9fce`

    Examples
    --------
    >>> import numpy as np
    >>> from popsregression import POPSRegression
    >>> rng = np.random.RandomState(0)
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.dot(X, np.array([1, 2])) + 0.01 * rng.randn(4)
    >>> reg = POPSRegression()
    >>> reg.fit(X, y)
    POPSRegression()
    >>> reg.predict(np.array([[3, 5]]))  # doctest: +ELLIPSIS
    array([...])
    """

    def __init__(
        self,
        *,
        max_iter=300,
        tol=1.0e-3,
        alpha_1=1.0e-6,
        alpha_2=1.0e-6,
        lambda_1=1.0e-6,
        lambda_2=1.0e-6,
        alpha_init=None,
        lambda_init=None,
        fit_intercept=False,
        mode_threshold=1.0e-8,
        resample_density=1.0,
        resampling_method="uniform",
        percentile_clipping=0.0,
        leverage_regularization=1.0e-3,
        minimum_relative_correction=1.0e-2,
        posterior="hypercube",
        random_state=None,
        minimum_relative_error="deprecated",
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.fit_intercept = fit_intercept
        self.mode_threshold = mode_threshold
        self.resample_density = resample_density
        self.resampling_method = resampling_method
        self.percentile_clipping = percentile_clipping
        self.leverage_regularization = leverage_regularization
        self.minimum_relative_correction = minimum_relative_correction
        self.posterior = posterior
        self.random_state = random_state
        self.minimum_relative_error = minimum_relative_error

    def _validate(self):
        check_scalar("max_iter", self.max_iter, low=1)
        check_scalar("tol", self.tol, low=0.0)
        for name in ("alpha_1", "alpha_2", "lambda_1", "lambda_2"):
            check_scalar(name, getattr(self, name), low=0.0)
        check_scalar("mode_threshold", self.mode_threshold, low=0.0, low_open=True)
        check_scalar("resample_density", self.resample_density, low=0.0, low_open=True)
        check_option(
            "resampling_method", self.resampling_method, {"uniform", *_RESAMPLING}
        )
        check_scalar(
            "percentile_clipping", self.percentile_clipping, low=0.0, high=50.0
        )
        check_scalar("leverage_regularization", self.leverage_regularization, low=0.0)
        check_option("posterior", self.posterior, {"hypercube", "ensemble"})
        threshold = self.minimum_relative_correction
        if not isinstance(self.minimum_relative_error, str):
            warnings.warn(
                (
                    "'minimum_relative_error' is deprecated: training points are "
                    "now selected by the norm of their pointwise correction. Its "
                    "value is forwarded to 'minimum_relative_correction'."
                ),
                FutureWarning,
                stacklevel=3,
            )
            threshold = self.minimum_relative_error
        return check_scalar("minimum_relative_correction", threshold, low=0.0)

    def fit(self, X, y, sample_weight=None):
        """Fit the POPS regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. They weight the mean fit
            and the leverage regularization; points of zero weight do not
            contribute to the POPS posterior.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        weights = check_sample_weight(sample_weight, X.shape[0])
        self.n_features_in_ = X.shape[1]
        if self.fit_intercept:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        coef = self._fit_hypercube(X, y, weights)
        if self.fit_intercept:
            self.coef_, self.intercept_ = coef[:-1], float(coef[-1])
        else:
            self.coef_, self.intercept_ = coef, 0.0

        if self.posterior == "ensemble":
            samples = self.pointwise_corrections_[self.selected_].T
            sigma = samples @ samples.T / samples.shape[1]
        else:
            samples, sigma = self._sample_hypercube()
        self.posterior_samples_, self.misspecification_sigma_ = samples, sigma
        return self

    def _fit_hypercube(self, X, y, weights):
        """Mean fit, pointwise corrections and the PCA-aligned box.

        ``X`` is the full design (any intercept column included). Returns
        the coefficients of that design; no posterior samples are drawn.
        """
        threshold = self._validate()
        hyper = jnp.array([self.alpha_1, self.alpha_2, self.lambda_1, self.lambda_2])
        fit = bayesian_ridge(
            X,
            y,
            weights,
            as_nan(self.alpha_init),
            as_nan(self.lambda_init),
            hyper,
            float(self.tol),
            int(self.max_iter),
        )
        corrections, errors, leverage = pointwise_corrections(
            X, y, weights, fit, float(self.leverage_regularization)
        )
        keep, eigenvalues, eigenvectors, low, high = _select_and_bound(
            corrections, weights > 0, float(threshold), float(self.percentile_clipping)
        )

        self.alpha_ = float(fit.alpha)
        self.lambda_ = float(fit.lam)
        self.n_iter_ = int(fit.n_iter)
        self.sigma_ = np.asarray(scaled_covariance(fit)) / self.alpha_
        self.leverage_ = np.asarray(leverage)
        self.selected_ = np.asarray(keep)
        self._basis = np.asarray(fit.V)
        self._corrections = corrections

        eigenvalues = np.asarray(eigenvalues)
        modes = eigenvalues > self.mode_threshold * eigenvalues.max()
        self._hypercube_support = self._basis @ np.asarray(eigenvectors)[:, modes]
        self._hypercube_bounds = [np.asarray(low)[modes], np.asarray(high)[modes]]
        return np.asarray(fit.coef)

    @property
    def pointwise_corrections_(self):
        """Pointwise corrections of every training point, one per row."""
        check_is_fitted(self, "_corrections")
        return np.asarray(self._corrections) @ self._basis.T

    def _sample_hypercube(self, size=None, resampling_method=None):
        """Sample points from the fitted POPS hypercube.

        Parameters
        ----------
        size : int, optional
            Number of samples. If None, determined by ``resample_density``.

        resampling_method : str, optional
            Override the instance's resampling method.

        Returns
        -------
        samples : ndarray of shape (n_features, n_samples)
            Resampled weight perturbations.

        sigma : ndarray of shape (n_features, n_features)
            Misspecification covariance estimated from the samples.
        """
        method = (
            self.resampling_method if resampling_method is None else resampling_method
        )
        low, high = self._hypercube_bounds
        if size is None:
            size = int(self.resample_density * self.selected_.size)
        n_resample = max(size, 100)

        # random_state=None keeps NumPy's global state for 'uniform' and the
        # sampler's own entropy for the quasi-random methods.
        seed = self.random_state
        if method == "uniform":
            unit = check_random_state(seed).uniform(size=(low.size, n_resample))
        else:
            if method == "sobol":
                n_resample = 2 ** int(np.log2(n_resample))
            unit = _RESAMPLING[method](d=low.size, seed=seed).random(n_resample).T

        samples = self._hypercube_support @ (
            low[:, None] + (high - low)[:, None] * unit
        )
        return samples, samples @ samples.T / samples.shape[1]

    def _design(self, X):
        X = check_X(X, self.n_features_in_)
        if self.fit_intercept:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return X

    def predict(
        self,
        X,
        return_std=False,
        return_bounds=False,
        return_epistemic_std=False,
    ):
        """Predict using the POPS regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        return_std : bool, default=False
            If True, return the combined (misspecification + epistemic)
            standard deviation.

        return_bounds : bool, default=False
            If True, return the max and min predictions over the POPS
            posterior samples.

        return_epistemic_std : bool, default=False
            If True, return the epistemic-only standard deviation
            (from ``sigma_``, excluding misspecification).

        Returns
        -------
        y_mean : ndarray of shape (n_samples,)
            Predicted mean values.

        y_std : ndarray of shape (n_samples,)
            Combined standard deviation. Only returned if
            ``return_std=True``.

        y_max : ndarray of shape (n_samples,)
            Upper bound from posterior samples. Only returned if
            ``return_bounds=True``.

        y_min : ndarray of shape (n_samples,)
            Lower bound from posterior samples. Only returned if
            ``return_bounds=True``.

        y_epistemic_std : ndarray of shape (n_samples,)
            Epistemic-only standard deviation. Only returned if
            ``return_epistemic_std=True``.
        """
        check_is_fitted(self)
        X = self._design(X)
        y_mean = X[:, : self.n_features_in_] @ self.coef_ + self.intercept_
        result = [y_mean]

        if return_std or return_epistemic_std:
            epistemic_var = np.einsum("ij,ij->i", X @ self.sigma_, X)
        if return_std:
            misspecification_var = np.einsum(
                "ij,ij->i", X @ self.misspecification_sigma_, X
            )
            result.append(np.sqrt(misspecification_var + epistemic_var))
        if return_bounds:
            upper, lower = _sample_extrema(X, self.posterior_samples_)
            result.extend([y_mean + upper, y_mean + lower])
        if return_epistemic_std:
            result.append(np.sqrt(epistemic_var))

        if len(result) == 1:
            return result[0]
        return tuple(result)


def _sample_extrema(X, samples, block=4096):
    """Row-wise max and min of ``X @ samples``, in blocks of posterior samples."""
    upper = np.full(X.shape[0], -np.inf)
    lower = np.full(X.shape[0], np.inf)
    for start in range(0, samples.shape[1], block):
        values = X @ samples[:, start : start + block]
        np.maximum(upper, values.max(axis=1), out=upper)
        np.minimum(lower, values.min(axis=1), out=lower)
    return upper, lower
