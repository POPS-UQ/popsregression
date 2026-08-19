# API reference

The package exposes two estimators for everyday use, plus the ellipsoid
implementation they share.

```python
from popsregression import POPSRegression, POPSRegressionPAC
```

| Object | Description |
|---|---|
| [`POPSRegression`](#popsregression.POPSRegression) | Bayesian regression with misspecification uncertainty; `posterior` selects `'hypercube'`, `'ensemble'` or `'ellipsoid'` |
| [`POPSRegressionPAC`][popsregression.POPSRegressionPAC] | The ellipsoid posterior with the PAC-Bayes layer (see [Ellipsoid posteriors](ellipse.md)) |
| [`POPSRegressionEllipse`][popsregression.POPSRegressionEllipse] | The ellipsoid estimator itself, used by both of the above |
| `popsregression.__version__` | Installed package version |

## Choosing an estimator

Every POPS posterior without a PAC-Bayes layer is a `posterior` choice on the
one estimator:

```python
POPSRegression(posterior="hypercube")   # default: axis-aligned box in PCA space
POPSRegression(posterior="ensemble")    # raw pointwise corrections
POPSRegression(posterior="ellipsoid")   # uniform ellipsoid, exact pushforward
POPSRegressionPAC()                     # the ellipsoid, plus the PAC-Bayes layer
```

`posterior='ellipsoid'` delegates to
[`POPSRegressionEllipse`][popsregression.POPSRegressionEllipse]; extra settings
for it go through `posterior_options`, and `predict` then uses the exact
projected-ball pushforward rather than the posterior samples:

```python
POPSRegression(
    posterior="ellipsoid",
    random_state=0,
    posterior_options={"rank": 16, "baseline": "ridge"},
)
```

`pac_bayes` is not accepted there: use `POPSRegressionPAC`, which exposes every
ellipsoid and PAC-Bayes parameter directly.

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

## POPSRegressionEllipse and POPSRegressionPAC

The full ellipsoid reference — mathematical background, the PAC-Bayes layer and
the `fit`/`predict`/`sample` methods — lives on the
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
