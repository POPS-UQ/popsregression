"""
POPS (Pointwise Optimal Parameter Sets) Regression.

Bayesian regression for low-noise data accounting for model misspecification.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from numbers import Real

import numpy as np
from scipy.linalg import eigh
from scipy.stats import qmc
from sklearn.base import _fit_context
from sklearn.linear_model._base import _preprocess_data
from sklearn.linear_model._bayes import BayesianRidge
from sklearn.utils._param_validation import Hidden, Interval, StrOptions
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)


class POPSRegression(BayesianRidge):
    """Bayesian regression for low-noise data with misspecification uncertainty.

    Fits a linear model using BayesianRidge, then estimates weight
    uncertainties accounting for model misspecification using the POPS
    (Pointwise Optimal Parameter Sets) algorithm [1]_. Unlike standard
    Bayesian regression, the aleatoric noise precision ``alpha_`` is not
    used for predictions, as it should be negligible in the low-noise regime.

    Standard Bayesian regression can only estimate epistemic and aleatoric
    uncertainties. In the low-noise limit, weight uncertainties (``sigma_``
    in :class:`BayesianRidge`) are significantly underestimated as they only
    account for epistemic uncertainties that decay with increasing data.
    POPS corrects this by estimating misspecification uncertainty from
    pointwise optimal parameter sets.

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations for the BayesianRidge convergence loop.

    tol : float, default=1e-3
        Convergence threshold. Stop the algorithm if the coefficient vector
        has converged.

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

    compute_score : bool, default=False
        If True, compute the log marginal likelihood at each step.

    fit_intercept : bool, default=False
        Whether to fit an intercept. If True, a constant column is appended
        to X (rather than centering) so that the intercept participates in
        the POPS posterior estimation.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    verbose : bool, default=False
        Verbose mode when fitting the model.

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

    minimum_relative_error : float, default=0.01
        Relative residual threshold for training point selection, in units
        of the root-mean-square error of the mean fit. Only training points
        whose residual satisfies ``|y - X @ coef_| >= minimum_relative_error
        * rmse`` contribute to the POPS posterior; a value of ``0.01`` thus
        discards points fit a hundred times better than the typical training
        point. Larger values accelerate fitting by focusing on the points the
        model fits worst, which are the ones that carry misspecification
        information. Use ``0.0`` to keep every training point. If no point
        passes the threshold, all points are used.

    posterior : {'hypercube', 'ensemble'}, default='hypercube'
        Form of the POPS parameter posterior:

        - ``'hypercube'``: fit an axis-aligned box in PCA space (default).
        - ``'ensemble'``: use raw pointwise corrections directly.

    leverage_percentile : float, default='deprecated'
        Deprecated. Training points used to be selected by leverage score
        percentile; they are now selected by residual magnitude through
        ``minimum_relative_error``. Passing this parameter raises a
        :class:`FutureWarning` and has no effect.

        .. deprecated:: 0.5
            ``leverage_percentile`` is deprecated and will be removed in
            0.7. Use ``minimum_relative_error`` instead.

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
        Estimated epistemic variance-covariance matrix of the weights.

    misspecification_sigma_ : ndarray of shape (n_features, n_features)
        Estimated misspecification variance-covariance matrix from POPS.

    posterior_samples_ : ndarray of shape (n_features, n_posterior_samples)
        Samples from the POPS posterior, representing plausible weight
        perturbations.

    scores_ : ndarray of shape (n_iter_,)
        Value of the log marginal likelihood at each iteration.
        Only available if ``compute_score=True``.

    n_iter_ : int
        The actual number of iterations to reach convergence.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    sklearn.linear_model.BayesianRidge : Bayesian ridge regression without
        misspecification correction.
    sklearn.linear_model.ARDRegression : Bayesian ARD regression.

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

    _parameter_constraints: dict = {
        **BayesianRidge._parameter_constraints,
        "mode_threshold": [Interval(Real, 0, None, closed="neither")],
        "resample_density": [Interval(Real, 0, None, closed="neither")],
        "resampling_method": [StrOptions({"uniform", "sobol", "latin", "halton"})],
        "percentile_clipping": [Interval(Real, 0, 50.0, closed="both")],
        "minimum_relative_error": [Interval(Real, 0.0, None, closed="left")],
        "posterior": [StrOptions({"hypercube", "ensemble"})],
        "leverage_percentile": [
            Interval(Real, 0.0, 100.0, closed="left"),
            Hidden(StrOptions({"deprecated"})),
        ],
    }

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
        compute_score=False,
        fit_intercept=False,
        copy_X=True,
        verbose=False,
        mode_threshold=1.0e-8,
        resample_density=1.0,
        resampling_method="uniform",
        percentile_clipping=0.0,
        minimum_relative_error=1.0e-2,
        posterior="hypercube",
        leverage_percentile="deprecated",
    ):
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            alpha_init=alpha_init,
            lambda_init=lambda_init,
            compute_score=compute_score,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            verbose=verbose,
        )
        self.mode_threshold = mode_threshold
        self.resample_density = resample_density
        self.resampling_method = resampling_method
        self.percentile_clipping = percentile_clipping
        self.minimum_relative_error = minimum_relative_error
        self.posterior = posterior
        self.leverage_percentile = leverage_percentile

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the POPS regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if not isinstance(self.leverage_percentile, str):
            warnings.warn(
                (
                    "'leverage_percentile' was deprecated in 0.5 and will be "
                    "removed in 0.7. Training points are now selected by residual "
                    "magnitude: use 'minimum_relative_error' instead. The value "
                    "passed to 'leverage_percentile' is ignored."
                ),
                FutureWarning,
            )

        pops_fit_intercept = self.fit_intercept
        if self.fit_intercept:
            X = np.asarray(X)
            X = np.hstack([X, np.ones((X.shape[0], 1))])
            self.fit_intercept = False

        try:
            super().fit(X, y, sample_weight=sample_weight)

            X_pp, y_pp = validate_data(
                self, X, y, dtype=[np.float64, np.float32], reset=False
            )

            if sample_weight is not None:
                sw = _check_sample_weight(sample_weight, X_pp, dtype=X_pp.dtype)
            else:
                sw = None

            preprocess_result = _preprocess_data(
                X_pp,
                y_pp,
                fit_intercept=False,
                copy=True,
                sample_weight=sw,
            )
            X_pp, y_pp = preprocess_result[0], preprocess_result[1]

            n_samples = X_pp.shape[0]

            scaled_sigma_ = self.alpha_ * self.sigma_

            errors = y_pp - X_pp @ self.coef_
            pointwise_correction = np.dot(X_pp, scaled_sigma_)

            leverage_scores = np.sum(pointwise_correction * X_pp, axis=1)
            safe_leverage = np.where(leverage_scores > 1e-6, leverage_scores, 1e-6)
            pointwise_correction *= (errors / safe_leverage)[:, None]

            rmse = np.sqrt(np.mean(errors**2))
            filtering_mask = np.abs(errors) >= self.minimum_relative_error * rmse
            if not np.any(filtering_mask):
                filtering_mask = np.ones(n_samples, dtype=bool)

            self._pointwise_correction = pointwise_correction
            self._filtering_mask = filtering_mask

            self.posterior_samples_, self.misspecification_sigma_ = (
                self._build_posterior()
            )
            self._fitted_with_intercept = pops_fit_intercept

        finally:
            self.fit_intercept = pops_fit_intercept

        return self

    def _build_posterior(self):
        """Build the POPS posterior from pointwise corrections.

        Returns
        -------
        samples : ndarray of shape (n_features, n_samples)
            Posterior samples (weight perturbations).

        sigma : ndarray of shape (n_features, n_features)
            Misspecification covariance matrix.
        """
        pc = self._pointwise_correction[self._filtering_mask]

        if self.posterior == "ensemble":
            sigma = pc.T @ pc / pc.shape[0]
            return pc.T, sigma

        elif self.posterior == "hypercube":
            self._hypercube_support, self._hypercube_bounds = self._fit_hypercube(pc)
            return self._sample_hypercube()

    def _fit_hypercube(self, pointwise_correction):
        """Fit a hypercube to the pointwise corrections via PCA.

        Parameters
        ----------
        pointwise_correction : ndarray of shape (n_samples, n_features)
            Pointwise corrections from the selected training points.

        Returns
        -------
        projections : ndarray of shape (n_features, n_active_modes)
            Principal component vectors defining the hypercube space.

        bounds : list of ndarray
            Two arrays [low, high] giving the min/max bounds along each
            principal component.
        """
        e_values, e_vectors = eigh(pointwise_correction.T @ pointwise_correction)

        mask = e_values > self.mode_threshold * e_values.max()
        e_vectors = e_vectors[:, mask]

        projections = e_vectors.copy()
        projected = pointwise_correction @ projections

        bounds = [
            np.percentile(projected, self.percentile_clipping, axis=0),
            np.percentile(projected, 100.0 - self.percentile_clipping, axis=0),
        ]

        return projections, bounds

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
        if resampling_method is None:
            resampling_method = self.resampling_method

        low = self._hypercube_bounds[0]
        high = self._hypercube_bounds[1]

        if size is None:
            n_resample = int(self.resample_density * len(self._pointwise_correction))
        else:
            n_resample = size
        n_resample = max(n_resample, 100)

        if resampling_method == "latin":
            sampler = qmc.LatinHypercube(d=low.size)
            samples = sampler.random(n_resample).T
        elif resampling_method == "sobol":
            sampler = qmc.Sobol(d=low.size)
            n_resample = 2 ** int(np.log2(n_resample))
            samples = sampler.random(n_resample).T
        elif resampling_method == "halton":
            sampler = qmc.Halton(d=low.size)
            samples = sampler.random(n_resample).T
        elif resampling_method == "uniform":
            samples = np.random.uniform(size=(low.size, n_resample))

        samples = low[:, None] + (high - low)[:, None] * samples

        hypercube_samples = self._hypercube_support @ samples
        hypercube_sigma = (
            hypercube_samples @ hypercube_samples.T / hypercube_samples.shape[1]
        )

        return hypercube_samples, hypercube_sigma

    def predict(
        self, X, return_std=False, return_bounds=False, return_epistemic_std=False
    ):
        """Predict using the POPS regression model.

        In addition to the standard ``return_std`` from
        :class:`BayesianRidge`, this method can return prediction bounds
        (min/max over the posterior) and epistemic-only uncertainty.

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

        if getattr(self, "_fitted_with_intercept", False):
            X = np.asarray(X)
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        y_mean = self._decision_function(X)
        result = [y_mean]

        if return_std or return_bounds or return_epistemic_std:
            y_epistemic_var = (np.dot(X, self.sigma_) * X).sum(axis=1)

            if return_std:
                y_misspecification_var = (
                    np.dot(X, self.misspecification_sigma_) * X
                ).sum(axis=1)
                result.append(np.sqrt(y_misspecification_var + y_epistemic_var))

            if return_bounds:
                y_posterior = X @ self.posterior_samples_
                y_max = y_posterior.max(axis=1) + y_mean
                y_min = y_posterior.min(axis=1) + y_mean
                result.extend([y_max, y_min])

            if return_epistemic_std:
                result.append(np.sqrt(y_epistemic_var))

        if len(result) == 1:
            return result[0]
        return tuple(result)
