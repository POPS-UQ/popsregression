# Usage

How to fit, predict and configure
[`POPSRegression`][popsregression.POPSRegression]. For exact signatures see the
[API reference](api.md); for the method behind it see the
[POPS site](https://pops-uq.github.io/method/concepts/).

## Fitting

`POPSRegression` follows the scikit-learn estimator API: construct, `fit`,
`predict`.

```python
import numpy as np
from popsregression import POPSRegression

model = POPSRegression()
model.fit(X_train, y_train)      # X: (n_samples, n_features), y: (n_samples,)
```

The estimator expects an explicit design matrix — features are not generated
for you. For a polynomial fit, build `X` with
[`PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
or a pipeline (see [below](#pipelines-and-model-selection)).

### Intercept handling

`fit_intercept` defaults to `False`, unlike `BayesianRidge`. When set to
`True`, a constant column is **appended to `X`** rather than the data being
centred, so that the intercept participates in the POPS posterior and carries
its own misspecification uncertainty. The same column is appended
automatically at prediction time.

```python
model = POPSRegression(fit_intercept=True)
```

If your design matrix already contains a bias column (for example
`PolynomialFeatures(include_bias=True)`), keep `fit_intercept=False`.

### Sample weights

```python
model.fit(X_train, y_train, sample_weight=w)
```

Weights are passed through to the underlying `BayesianRidge` fit and to the
preprocessing used for the POPS corrections.

## Prediction

`predict` returns the mean by default. Each flag appends further arrays, in a
fixed order:

```python
y_pred = model.predict(X_test)

y_pred, y_std = model.predict(X_test, return_std=True)

y_pred, y_std, y_max, y_min = model.predict(
    X_test, return_std=True, return_bounds=True
)

y_pred, y_std, y_max, y_min, y_epistemic_std = model.predict(
    X_test,
    return_std=True,
    return_bounds=True,
    return_epistemic_std=True,
)
```

| Flag | Appended array | Meaning |
|---|---|---|
| — | `y_mean` | Posterior mean prediction |
| `return_std=True` | `y_std` | Combined misspecification **+** epistemic standard deviation |
| `return_bounds=True` | `y_max`, `y_min` | Upper and lower envelope over the POPS posterior samples |
| `return_epistemic_std=True` | `y_epistemic_std` | Epistemic-only standard deviation, from `sigma_` alone |

The order is always `y_mean, y_std, y_max, y_min, y_epistemic_std`, with
omitted entries dropped. With no flags set, a single array is returned rather
than a tuple.

!!! note "The aleatoric term is deliberately excluded"
    The fitted noise precision `alpha_` is not used in any predictive
    uncertainty. POPS targets the low-noise regime, where the aleatoric
    contribution should be negligible and any residual error is attributed to
    model form, not measurement noise.

## Choosing parameters

All `BayesianRidge` parameters (`max_iter`, `tol`, `alpha_1`, `alpha_2`,
`lambda_1`, `lambda_2`, `alpha_init`, `lambda_init`, `compute_score`,
`fit_intercept`, `copy_X`, `verbose`) are accepted and forwarded. The
POPS-specific parameters are:

| Parameter | Default | Notes |
|---|---|---|
| `posterior` | `'hypercube'` | `'hypercube'` fits a PCA-aligned box to the pointwise corrections and resamples it; `'ensemble'` uses the raw corrections as samples |
| `minimal_error` | `1e-3` | Residual threshold for selecting training points (see below) |
| `resampling_method` | `'uniform'` | `'uniform'`, `'sobol'`, `'latin'` or `'halton'`; hypercube posterior only |
| `resample_density` | `1.0` | Posterior samples per training point; the count is floored at 100 |
| `percentile_clipping` | `0.0` | Percentile trimmed from each end of the hypercube bounds, in `[0, 50]` |
| `mode_threshold` | `1e-8` | Relative eigenvalue cutoff setting the effective dimension of the hypercube |

### `minimal_error`

Only training points whose absolute residual `|y - X @ coef_|` is at least
`minimal_error` contribute to the POPS posterior. Points the model already
reproduces carry no misspecification information, so discarding them speeds up
the posterior construction without widening or narrowing it meaningfully.

The threshold is an **absolute value in the units of `y`**, so it must be
scaled to your target:

```python
# Keep every training point
model = POPSRegression(minimal_error=0.0)

# Energies in eV: ignore points already fit to better than 1 meV
model = POPSRegression(minimal_error=1e-3)
```

If no point clears the threshold, all points are used, so an over-large value
degrades to the unfiltered fit rather than failing.

!!! warning "`leverage_percentile` is deprecated"
    Earlier releases selected training points by leverage score percentile.
    That parameter is deprecated since 0.5, is ignored, raises a
    `FutureWarning`, and will be removed in 0.7. Replace
    `leverage_percentile=0.0` (use all points) with `minimal_error=0.0`, and
    tune `minimal_error` to your target scale otherwise.

### Sampling the hypercube posterior

`resampling_method` controls how the fitted hypercube is sampled. The
quasi-random methods (`'sobol'`, `'latin'`, `'halton'`) cover the box more
evenly than `'uniform'`, which matters for the min/max bounds returned by
`return_bounds=True`.

```python
model = POPSRegression(resampling_method="sobol", resample_density=10.0)
```

Two caveats: `'sobol'` rounds the sample count down to a power of two, and
`'uniform'` draws from the global NumPy random state, so seed with
`np.random.seed(...)` if you need reproducible bounds. The `'ensemble'`
posterior ignores both sampling parameters — it uses the pointwise corrections
directly.

## Fitted attributes

| Attribute | Description |
|---|---|
| `coef_` | Regression coefficients (posterior mean) |
| `intercept_` | Independent term; `0.0` when `fit_intercept=False` |
| `sigma_` | Epistemic variance-covariance matrix of the weights |
| `misspecification_sigma_` | Misspecification variance-covariance matrix from POPS |
| `posterior_samples_` | POPS posterior samples, shape `(n_features, n_posterior_samples)` |
| `alpha_` | Estimated noise precision — fitted, but not used for prediction |
| `lambda_` | Estimated weight precision |
| `scores_` | Log marginal likelihood per iteration; requires `compute_score=True` |
| `n_iter_` | Iterations to convergence |

`posterior_samples_` holds weight *perturbations* around `coef_`, so a
parameter ensemble is `coef_[:, None] + model.posterior_samples_`.

## Pipelines and model selection

`POPSRegression` clones, gets and sets parameters like any scikit-learn
estimator, so it drops into pipelines and search:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

pipe = make_pipeline(
    PolynomialFeatures(degree=4),
    POPSRegression(resampling_method="sobol"),
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

search = GridSearchCV(
    pipe,
    {
        "polynomialfeatures__degree": [2, 3, 4],
        "popsregression__posterior": ["hypercube", "ensemble"],
    },
)
search.fit(X_train, y_train)
```

Note that `predict` inside a pipeline returns the mean only; call the
estimator step directly (`pipe[-1].predict(X_transformed, return_std=True)`)
when you need the uncertainty outputs.

## Deprecations

| Since | Removed in | Parameter | Replacement |
|---|---|---|---|
| 0.5 | 0.7 | `leverage_percentile` | `minimal_error` |

Importing the top-level `POPSRegression` module (`import POPSRegression`) is
also deprecated; use `from popsregression import POPSRegression`.
