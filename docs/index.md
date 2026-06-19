# popsregression

A Python library of probabilistic surrogate models adapted to misspecified
functional forms in the small observation noise regime.

The package takes its name from the POPS regression algorithm
([Perez & Swinburne, 2025](https://doi.org/10.1088/2632-2153/ad9fce)).
[`POPSRegression`][popsregression.POPSRegression] is a
[scikit-learn](https://scikit-learn.org) compatible estimator that extends
[`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
to estimate weight uncertainties accounting for model misspecification. A Julia
implementation following the [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl)
conventions is also available in
[POPSRegression.jl](https://github.com/POPS-UQ/POPSRegression.jl).

!!! note "Why misspecification uncertainty?"
    Standard Bayesian regression (e.g. `BayesianRidge`) estimates epistemic and
    aleatoric uncertainties, but provably ignores model misspecification — the
    error arising from a limited model form. In the low-noise
    (near-deterministic) limit, weight uncertainties are significantly
    underestimated, since they only capture epistemic uncertainty, which decays
    with increasing data. `POPSRegression` efficiently estimates model
    misspecification uncertainty via the Pointwise Optimal Parameter Sets (POPS)
    algorithm, yielding wider, more honest uncertainty estimates that properly
    cover the true function even when the model class cannot represent the
    target exactly.

## Implemented features

  - Misspecification-aware Bayesian regression for linear models via the POPS algorithm
  - Hypercube and ensemble posteriors over pointwise optimal parameter sets
  - Leverage-based filtering of training points for efficient fitting
  - Predictive uncertainty quantification: min–max bounds, combined and epistemic-only standard deviations
  - Full scikit-learn compatibility: pipelines, hyperparameter search, and the standard estimator API
  - Unit testing, including the standard scikit-learn estimator checks

## Installation

```bash
pip install popsregression
```

Dependencies: `scikit-learn>=1.6.1`, `scipy>=1.6.0`, `numpy>=1.20.0`.

## Running tests

From the package root directory:

```bash
pytest -vsl popsregression
```

If you have [pixi](https://pixi.sh/) installed, the pre-configured tasks are:

```bash
pixi run test       # run tests
pixi run lint       # check code style
pixi run build-doc  # build documentation
```

## Quick start

```python
from popsregression import POPSRegression

X_train, X_test, y_train, y_test = ...

# Fit POPSRegression (fit_intercept=False by default)
model = POPSRegression()
model.fit(X_train, y_train)

# Prediction with combined misspecification + epistemic uncertainty
y_pred, y_std = model.predict(X_test, return_std=True)

# Also return min/max bounds over the posterior
y_pred, y_std, y_max, y_min = model.predict(
    X_test, return_std=True, return_bounds=True
)

# Also return epistemic-only uncertainty separately
y_pred, y_std, y_max, y_min, y_epistemic_std = model.predict(
    X_test,
    return_std=True,
    return_bounds=True,
    return_epistemic_std=True,
)
```

`POPSRegression` is fully compatible with scikit-learn pipelines and
hyperparameter search:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

pipe = make_pipeline(
    PolynomialFeatures(degree=4),
    POPSRegression(resampling_method="sobol"),
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

## Citation

> *Parameter uncertainties for imperfect surrogate models in the low-noise regime*
>
> T. D. Swinburne and D. Perez,
> [Machine Learning: Science and Technology, 2025](https://doi.org/10.1088/2632-2153/ad9fce)

```bibtex
@article{swinburne2025,
    author  = {Swinburne, Thomas and Perez, Danny},
    title   = {Parameter uncertainties for imperfect surrogate models in the low-noise regime},
    journal = {Machine Learning: Science and Technology},
    doi     = {10.1088/2632-2153/ad9fce},
    year    = {2025}
}
```

## Contents

  - [API reference](api.md)
  - [Example: POPS vs BayesianRidge](example.md)
