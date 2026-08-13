# API reference

The package exposes a single public estimator.

```python
from popsregression import POPSRegression
```

| Object | Description |
|---|---|
| [`POPSRegression`](#popsregression.POPSRegression) | Bayesian regression with misspecification uncertainty |
| `popsregression.__version__` | Installed package version |

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

## Deprecated parameters

| Since | Removed in | Parameter | Replacement |
|---|---|---|---|
| 0.5 | 0.7 | `leverage_percentile` | `minimum_relative_error` |

Passing `leverage_percentile` raises a `FutureWarning` on `fit` and has no
effect on the fitted model.

The top-level `POPSRegression` module shim (`import POPSRegression`) is
deprecated in favour of `from popsregression import POPSRegression` and raises
a `DeprecationWarning` on import.
