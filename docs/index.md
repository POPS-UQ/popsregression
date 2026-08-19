# popsregression

The Python implementation of the POPS (Pointwise Optimal Parameter Sets)
algorithm. [`POPSRegression`][popsregression.POPSRegression] is a
[scikit-learn](https://scikit-learn.org) compatible estimator extending
[`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
with model misspecification uncertainty.

**Method and theory** — concepts, algorithm, tutorials, citation and the Julia
implementation — are documented at
[pops-uq.github.io](https://pops-uq.github.io).
**These pages** cover this package: installation, [usage](usage.md) and the
[API reference](api.md).

## Installation

```bash
pip install popsregression
```

Requires Python >= 3.9. Dependencies: `scikit-learn>=1.6.1`, `scipy>=1.6.0`,
`numpy>=1.20.0`.

## Quick start

```python
from popsregression import POPSRegression

X_train, X_test, y_train, y_test = ...

# fit_intercept=False by default
model = POPSRegression()
model.fit(X_train, y_train)

# Combined misspecification + epistemic standard deviation
y_pred, y_std = model.predict(X_test, return_std=True)
```

- [Usage](usage.md) — fitting, prediction, parameter choice
- [API reference](api.md) — signatures, parameters, attributes
- [Example: POPS vs BayesianRidge](example.md) — runnable comparison

## Development

Source and issue tracker:
[github.com/POPS-UQ/popsregression](https://github.com/POPS-UQ/popsregression).

The repository is managed with [uv](https://docs.astral.sh/uv/), which
resolves the pinned environment from `uv.lock`:

```bash
uv run --group test pytest -vsl popsregression   # tests
uv run --group lint ruff check popsregression    # linter
uv run --group doc mkdocs serve                  # docs
```
