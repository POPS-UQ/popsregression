# popsregression

The Python implementation of the POPS (Pointwise Optimal Parameter Sets)
approach to misspecification-aware linear regression for near-deterministic
surrogate models. Two [scikit-learn](https://scikit-learn.org) compatible
estimators are provided:

- [`POPSRegression`][popsregression.POPSRegression] extends
  [`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
  with the POPS hypercube or ensemble posterior.
- [`POPSEllipseRegression`][popsregression.POPSEllipseRegression] fits a
  uniform-ellipsoid posterior with an exact predictive density. It has an
  optional empirical-Bayes layer, and a PAC construction that comes with a
  finite-sample bound.

**Method and theory** — concepts, algorithm, tutorials, citation and the Julia
implementation — are documented at
[pops-uq.github.io](https://pops-uq.github.io). **These pages** cover this
package. Specialised terms are defined in the [glossary](glossary.md).

## Installation

```bash
pip install popsregression
```

Requires Python >= 3.9. Dependencies: `scikit-learn>=1.6.1`, `scipy>=1.6.0`,
`numpy>=1.20.0`.

## Quick start

```python
from popsregression import POPSEllipseRegression, POPSRegression

X_train, X_test, y_train, y_test = ...

# POPS hypercube posterior: mean and misspecification + epistemic std
model = POPSRegression().fit(X_train, y_train)
y_pred, y_std = model.predict(X_test, return_std=True)

# Ellipse posterior with the empirical-Bayes finite-data layer
ellipse = POPSEllipseRegression(regularization="empirical-bayes")
ellipse.fit(X_train, y_train)
lo, hi = ellipse.predict_interval(X_test, level=0.9545)
theta = ellipse.sample(1000)          # parameter draws for propagation

# The same with a PAC-Bayes bound; y_bounds must be known in advance
pac = POPSEllipseRegression(regularization="PAC", y_bounds=(y_lower, y_upper))
pac.fit(X_train, y_train)
pac.certificate_.raw_bound
```

- [Usage](usage.md): fitting, prediction and parameters of `POPSRegression`
- [Ellipse posteriors](ellipse.md): `POPSEllipseRegression`, including the
  PAC protocol
- [Example](example.md): a runnable comparison
- [Studies](studies.md): the paper's studies and comparison methods
- [API reference](api.md): signatures, parameters, attributes
- [Glossary](glossary.md)

## Development

Source and issue tracker:
[github.com/POPS-UQ/popsregression](https://github.com/POPS-UQ/popsregression).

The repository is managed with [uv](https://docs.astral.sh/uv/), which
resolves the pinned environment from `uv.lock`:

```bash
uv run --group test pytest -vsl popsregression examples/comparisons   # tests
uv run --group lint ruff check popsregression                         # linter
uv run --group doc mkdocs serve                                       # docs
```
