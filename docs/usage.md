# Usage

How to fit, predict and configure
[`POPSRegression`][popsregression.POPSRegression], the hypercube and ensemble
POPS posteriors. The ellipsoid posterior, with its empirical-Bayes and PAC
options, is a separate estimator,
[`POPSEllipseRegression`][popsregression.POPSEllipseRegression], described in
[Ellipse posteriors](ellipse.md). Exact signatures are in the
[API reference](api.md); the method itself is documented at
[pops-uq.github.io](https://pops-uq.github.io). Unfamiliar terms are defined in
the [glossary](glossary.md).

## Fitting

`POPSRegression` follows the familiar estimator usage: construct, `fit`,
`predict`. It does not depend on scikit-learn.

```python
import numpy as np
from popsregression import POPSRegression

model = POPSRegression()
model.fit(X_train, y_train)      # X: (n_samples, n_features), y: (n_samples,)
```

The estimator expects an explicit design matrix — features are not generated
for you. For a polynomial fit, build `X` yourself (for example with
`numpy.vander`), or with scikit-learn's
[`PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
in a pipeline (see [below](#pipelines-and-model-selection)).

### Intercept handling

`fit_intercept` defaults to `False`. When set to `True`, a constant column is
**appended to `X`** rather than the data being centred, so that the intercept
participates in the POPS posterior and carries its own misspecification
uncertainty. The same column is appended automatically at prediction time.
The fitted intercept is `intercept_` and `coef_` holds the feature
coefficients only; `sigma_`, `misspecification_sigma_` and
`posterior_samples_` keep the intercept as their last coordinate.

```python
model = POPSRegression(fit_intercept=True)
```

If your design matrix already contains a bias column (for example
`PolynomialFeatures(include_bias=True)`), keep `fit_intercept=False`.

### Sample weights

```python
model.fit(X_train, y_train, sample_weight=w)
```

Weights enter the Bayesian ridge fit (as fractional observation counts) and
the leverage regularization of the POPS corrections. Points of zero weight do
not contribute to the POPS posterior.

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

The Bayesian ridge parameters of the mean fit (`max_iter`, `tol`, `alpha_1`,
`alpha_2`, `lambda_1`, `lambda_2`, `alpha_init`, `lambda_init`,
`fit_intercept`) keep their usual meaning. The POPS-specific parameters are:

| Parameter | Default | Notes |
|---|---|---|
| `posterior` | `'hypercube'` | `'hypercube'` fits a PCA-aligned box to the pointwise corrections and resamples it; `'ensemble'` uses the raw corrections as samples |
| `random_state` | `None` | Seed for the posterior resampling; `None` keeps the global NumPy state |
| `prior_precision` | `None` | Shape of the Gaussian prior on the coefficients (see below) |
| `leverage_regularization` | `0.001` | Regularization of the leverage in the pointwise corrections, in units of the mean leverage (see below) |
| `minimum_relative_correction` | `0.01` | Relative correction-norm threshold for selecting training points (see below) |
| `resampling_method` | `'uniform'` | `'uniform'`, `'sobol'`, `'latin'` or `'halton'`; hypercube posterior only |
| `resample_density` | `1.0` | Posterior samples per training point; the count is floored at 100 |
| `percentile_clipping` | `0.0` | Percentile trimmed from each end of the hypercube bounds, in `[0, 50]` |
| `mode_threshold` | `1e-8` | Relative eigenvalue cutoff setting the effective dimension of the hypercube |

### `prior_precision`

By default the coefficients have the isotropic Gaussian prior of Bayesian
ridge regression, `N(0, I / λ)`, whose precision `λ` is learned by evidence
maximization. `prior_precision` replaces the identity by a shape `Λ` of your
choice, a vector (diagonal `Λ`) or a symmetric positive definite matrix:

```text
θ ~ N(0, (λ Λ)⁻¹),        A = XᵀX + (λ/α) Λ
```

```python
# Penalize the last two coefficients a thousand times more than the others
model = POPSRegression(prior_precision=[1, 1, 1, 1e3, 1e3])

# A full precision matrix, e.g. a smoothness prior
model = POPSRegression(prior_precision=Lambda)
```

Only the shape is fixed: the scale `λ` (`lambda_`) is still learned, so `Λ`
and `c·Λ` give the same fit. The mean fit, the leverages and the pointwise
corrections all use the metric `A`, so the prior also decides which direction
a pointwise correction takes. With `fit_intercept=True` the intercept keeps
unit prior precision unless `Λ` is given with one more row and column for it.

[`POPSEllipseRegression`][popsregression.POPSEllipseRegression] takes the same
argument. There it sets the center and the POPS baseline of the ellipsoid; it
does not constrain the fitted ellipsoid shape.

### `leverage_regularization`

The pointwise correction of training point `i` is the smallest parameter
move, in the metric `A = XᵀX + λ/α` of the mean fit, after which the model
passes through `y_i`:

```text
dθ_i = e_i A⁻¹x_i / h_i,      e_i = y_i - x_i·coef_,   h_i = x_iᵀA⁻¹x_i
```

It divides by the leverage `h_i`, so a training point the model is barely
sensitive to (`h_i → 0`) demands an arbitrarily large correction, and a
handful of such points can dominate the posterior. `leverage_regularization`
replaces the exact pointwise fit by the minimizer of
`(x_i·d - e_i)² / eps + dᵀA d`:

```text
dθ_i = e_i A⁻¹x_i / (h_i + eps),      eps = leverage_regularization * mean(h)
```

The correction is then bounded, `‖dθ_i‖_A ≤ |e_i| / (2 √eps)`, and the
pointwise fit is satisfied up to the relative residual `eps / (h_i + eps)`:
about `leverage_regularization` for a typical point, and a large fraction only
for points whose leverage is a thousand times below the mean at the default.

```python
# Default: mild regularization relative to the mean leverage
model = POPSRegression(leverage_regularization=1e-3)

# The exact pointwise fits of the original algorithm
model = POPSRegression(leverage_regularization=0.0)
```

!!! note "Designs with very uneven leverage"
    On a badly scaled design (for example raw monomials `1, x, ..., x⁴` over
    `x ∈ [-10, 10]`) the leverages of the training points span many orders of
    magnitude, and the corrections of the low-leverage points set much of the
    width of the hypercube. Regularizing them narrows it: in the quartic
    example of these pages with 100 training points, the hypercube bounds
    cover 90% of a dense test grid with `leverage_regularization=0.0`, 76%
    with the default `1e-3` and 63% with `1e-2` (the ellipse posteriors, which
    optimize coverage directly, are unaffected). Rescale the features, or set
    the parameter explicitly, if you need the exact pointwise fits there. The
    leverages are available as `leverage_`.

### `minimum_relative_correction`

A training point enters the singular decomposition and the bounds of the POPS
posterior only if its correction is not negligible next to the typical one:

```text
‖dθ_i‖  >=  minimum_relative_correction * rms(‖dθ‖)
```

with the Euclidean norm, the one the singular decomposition of the correction
matrix sees. Points with a negligible correction cannot change the principal
directions or widen the bounds, so discarding them does not alter the
posterior meaningfully; it only matters for the bounds when
`percentile_clipping > 0`.

Because the threshold is **relative**, it is invariant to the scale of `y`:

```python
# Discard corrections a hundred times smaller than the typical one (default)
model = POPSRegression(minimum_relative_correction=0.01)

# Keep every training point
model = POPSRegression(minimum_relative_correction=0.0)

# Aggressive: keep only corrections above half the RMS norm
model = POPSRegression(minimum_relative_correction=0.5)
```

If no point clears the threshold, all points are used, so an over-large value
degrades to the unfiltered fit rather than failing. The selected points are
`selected_`.

!!! warning "`minimum_relative_error` is deprecated"
    Earlier releases selected training points by the size of their residual,
    `|y - X @ coef_| >= minimum_relative_error * rmse`. A value passed to
    `minimum_relative_error` is forwarded to `minimum_relative_correction`
    with a `FutureWarning`. `leverage_percentile` has been removed.

### Sampling the hypercube posterior

`resampling_method` controls how the fitted hypercube is sampled. The
quasi-random methods (`'sobol'`, `'latin'`, `'halton'`) cover the box more
evenly than `'uniform'`, which matters for the min/max bounds returned by
`return_bounds=True`.

```python
model = POPSRegression(resampling_method="sobol", resample_density=10.0)
```

Two caveats: `'sobol'` rounds the sample count down to a power of two, and
with `random_state=None` (the default) `'uniform'` draws from the global NumPy
random state, so either pass `random_state=...` or seed with
`np.random.seed(...)` if you need reproducible bounds. The `'ensemble'`
posterior ignores both sampling parameters — it uses the pointwise corrections
directly.

### Ellipsoid posteriors

The uniform-ellipsoid posterior has an exact predictive density and is a
separate estimator:

```python
from popsregression import POPSEllipseRegression

model = POPSEllipseRegression(regularization="empirical-bayes").fit(X_train, y_train)
lo, hi = model.predict_interval(X_test, level=0.9545)
```

See [Ellipse posteriors](ellipse.md).

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
| `pointwise_corrections_` | Pointwise corrections `dθ_i`, shape `(n_samples, n_features)` |
| `leverage_` | Leverage `h_i` of every training point |
| `selected_` | Boolean mask of the points that entered the posterior |
| `n_iter_` | Iterations to convergence |

`posterior_samples_` holds weight *perturbations* around `coef_`, so a
parameter ensemble is `coef_[:, None] + model.posterior_samples_`.

## Pipelines and model selection

scikit-learn is not a dependency, but both estimators provide `get_params`
and `set_params`, so when scikit-learn is installed they drop into its
pipelines and search:

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

A pipeline forwards keyword arguments of `predict` to the estimator, so
`pipe.predict(X_test, return_std=True)` returns the uncertainty outputs. A
search object does not; use `search.best_estimator_.predict(...)`.

## Backend

All numerical kernels (the Bayesian ridge fit, the pointwise corrections, the
ellipsoid optimization, the PAC certification and the mixture predictives) are
written in [JAX](https://jax.readthedocs.io) and compiled on first use.

- The first call for a given problem shape pays a compilation cost of about a
  second; later calls with the same shapes reuse the compiled kernel.
- Importing the package enables JAX double precision
  (`jax_enable_x64`), a process-wide JAX setting.
- With a GPU build of JAX the same code runs on the accelerator.
- Inputs and fitted attributes (`coef_`, `sigma_`, ...) are NumPy arrays.

## Deprecations

| Parameter | Replacement |
|---|---|
| `minimum_relative_error` | `minimum_relative_correction` |

Importing the top-level `POPSRegression` module (`import POPSRegression`) is
also deprecated; use `from popsregression import POPSRegression`.
