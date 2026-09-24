# API reference

The package exposes two estimators and the records of the PAC bound.

```python
from popsregression import POPSRegression, POPSEllipseRegression
```

| Object | Description |
|---|---|
| [`POPSRegression`](#popsregression.POPSRegression) | Bayesian regression with misspecification uncertainty; `posterior` selects the `'hypercube'` (default) or `'ensemble'` POPS posterior |
| [`POPSEllipseRegression`](#popsregression.POPSEllipseRegression) | Uniform-ellipsoid posterior with an exact predictive density; `regularization` selects none, `'empirical-bayes'` (Ellipse+EB) or `'PAC'` (Ellipse+PAC) |
| [`PACCertificate`](#popsregression.PACCertificate), [`PACFoldBound`](#popsregression.PACFoldBound) | The PAC-Bayes bound of `regularization='PAC'` and its per-fold decomposition |
| `popsregression.__version__` | Installed package version |

Comparison methods used in the paper (Bayesian stacking, PACm, PAC²-T, PVI and
the finite POPS dictionary) are not part of the package. They live in
`examples/comparisons/` (see [Studies](studies.md)). Terms are defined in the
[glossary](glossary.md).

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

## POPSEllipseRegression

See [Ellipse posteriors](ellipse.md) for the method and the PAC protocol.

::: popsregression.POPSEllipseRegression
    options:
      members: false

### Methods

::: popsregression.POPSEllipseRegression.fit
    options:
      show_root_heading: true
      show_root_full_path: false

::: popsregression.POPSEllipseRegression.predict
    options:
      show_root_heading: true
      show_root_full_path: false

::: popsregression.POPSEllipseRegression.predict_interval
    options:
      show_root_heading: true
      show_root_full_path: false

::: popsregression.POPSEllipseRegression.predict_logpdf
    options:
      show_root_heading: true
      show_root_full_path: false

::: popsregression.POPSEllipseRegression.predict_cdf
    options:
      show_root_heading: true
      show_root_full_path: false

::: popsregression.POPSEllipseRegression.sample
    options:
      show_root_heading: true
      show_root_full_path: false

### PAC bound records

::: popsregression.PACCertificate
    options:
      members: true

::: popsregression.PACFoldBound
    options:
      members: false

## Deprecated parameters

| Since | Removed in | Parameter | Replacement |
|---|---|---|---|
| 0.5 | 0.7 | `leverage_percentile` | `minimum_relative_error` |

Passing `leverage_percentile` raises a `FutureWarning` on `fit` and has no
effect on the fitted model.

The top-level `POPSRegression` module shim (`import POPSRegression`) is
deprecated in favour of `from popsregression import POPSRegression` and raises
a `DeprecationWarning` on import.
