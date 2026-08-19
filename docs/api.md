# API reference

The package exposes a single estimator.

```python
from popsregression import POPSRegression
```

| Object | Description |
|---|---|
| [`POPSRegression`](#popsregression.POPSRegression) | Bayesian regression with misspecification uncertainty; `posterior` selects `'hypercube'`, `'ensemble'` or `'ellipsoid'`, and `pac_bayes` adds the PAC-Bayes layer |
| `popsregression.__version__` | Installed package version |

## Choosing a posterior

Every POPS posterior is a `posterior` choice on the one estimator:

```python
POPSRegression(posterior="hypercube")   # default: axis-aligned box in PCA space
POPSRegression(posterior="ensemble")    # raw pointwise corrections
POPSRegression(posterior="ellipsoid")   # uniform ellipsoid, exact pushforward
POPSRegression(posterior="ellipsoid", pac_bayes=True)   # plus the PAC-Bayes layer
```

With `posterior='ellipsoid'`, `predict` uses the exact projected-ball
pushforward rather than the posterior samples, and the ellipsoid's own tuning
parameters go through `posterior_options`:

```python
POPSRegression(
    posterior="ellipsoid",
    pac_bayes=True,
    random_state=0,
    posterior_options={"rank": 16, "baseline": "ridge", "hyperprior_scale": 1.0},
)
```

`pac_bayes`, `fit_intercept` and `random_state` are set on the estimator
itself, not through `posterior_options`; sample weights are passed to `fit`.
`pac_bayes=True` requires `posterior='ellipsoid'`.

## POPSRegression

::: popsregression.POPSRegression
    options:
      members: false

## Methods

### fit

::: popsregression.POPSRegression.fit
    options:
      show_root_heading: true
      show_root_full_path: false

### predict

::: popsregression.POPSRegression.predict
    options:
      show_root_heading: true
      show_root_full_path: false

### Inherited methods

`POPSRegression` subclasses
[`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
and inherits the standard scikit-learn estimator methods unchanged:

| Method | Description |
|---|---|
| `get_params(deep=True)` | Parameters of this estimator |
| `set_params(**params)` | Set parameters of this estimator |
| `score(X, y, sample_weight=None)` | Coefficient of determination R² of the prediction |
| `get_metadata_routing()` | Metadata routing of this object |

`score` uses the mean prediction only; uncertainty outputs are available
through [`predict`](#predict).

## The ellipsoid posterior

The full ellipsoid reference — mathematical background, the PAC-Bayes layer,
and every key accepted by `posterior_options` — lives on the
[Ellipsoid posteriors](ellipse.md) page.

## Deprecated parameters

| Since | Removed in | Parameter | Replacement |
|---|---|---|---|
| 0.5 | 0.7 | `leverage_percentile` | `minimum_relative_error` |

Passing `leverage_percentile` raises a `FutureWarning` on `fit` and has no
effect on the fitted model.

The top-level `POPSRegression` module shim (`import POPSRegression`) is
deprecated in favour of `from popsregression import POPSRegression` and raises
a `DeprecationWarning` on import.
