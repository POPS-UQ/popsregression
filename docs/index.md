# popsregression

`popsregression` is the **Python implementation** of the POPS (Pointwise
Optimal Parameter Sets) algorithm. It provides
[`POPSRegression`][popsregression.POPSRegression], a
[scikit-learn](https://scikit-learn.org) compatible estimator that extends
[`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
with model misspecification uncertainty.

!!! info "This site is the package reference"
    These pages document the Python package only: installation, the estimator
    API, and how to use it. The method itself — what misspecification
    uncertainty is, why it matters in the low-noise regime, and how POPS
    estimates it — is documented on the main POPS site:

    - [Concepts](https://pops-uq.github.io/method/concepts/) — the three
      uncertainty types and what POPS does and does not claim
    - [Algorithm](https://pops-uq.github.io/method/algorithm/) — the
      pointwise optimal parameter set construction
    - [Tutorials](https://pops-uq.github.io/tutorials/) — worked material
      shared across implementations
    - [POPSRegression.jl](https://pops-uq.github.io/implementations/julia/) —
      the Julia implementation

    In one sentence: standard Bayesian regression estimates epistemic and
    aleatoric uncertainty but provably ignores model misspecification, which
    dominates when the noise is small and the model form is limited.
    `POPSRegression` estimates that missing component for one extra linear
    solve.

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

See [Usage](usage.md) for the full set of fitting and prediction options, and
[API reference](api.md) for exact signatures.

## Package scope

  - Misspecification-aware Bayesian regression for linear models via POPS
  - `'hypercube'` and `'ensemble'` posteriors over pointwise optimal
    parameter sets
  - Residual-based selection of training points
    (`minimal_error`) for efficient fitting
  - Predictive uncertainty: combined and epistemic-only standard deviations,
    plus min/max bounds over the posterior
  - Full scikit-learn compatibility: pipelines, hyperparameter search, cloning
    and the standard estimator API, validated by the scikit-learn estimator
    checks

## Development

Run the test suite from the package root:

```bash
pytest -vsl popsregression
```

With [pixi](https://pixi.sh/) the pre-configured tasks are:

```bash
pixi run test       # run tests
pixi run lint       # check code style
pixi run build-doc  # build this documentation
```

Source and issue tracker:
[github.com/POPS-UQ/popsregression](https://github.com/POPS-UQ/popsregression).

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

  - [Usage](usage.md) — fitting, prediction, parameter choice
  - [API reference](api.md) — signatures, parameters, attributes
  - [Example: POPS vs BayesianRidge](example.md) — runnable comparison
