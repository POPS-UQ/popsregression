# popsregression

[![tests](https://github.com/POPS-UQ/popsregression/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/POPS-UQ/popsregression/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/POPS-UQ/popsregression/graph/badge.svg?token=L0XPWwoPLw)](https://codecov.io/gh/POPS-UQ/popsregression)
[![docs](https://img.shields.io/badge/docs-POPS--UQ-blue)](https://pops-uq.github.io/implementations/python/)

Post-Bayesian regression for misspecified, near-deterministic models. This is the JAX implementation of **Pointwise Optimal Parameter Sets (POPS)** for models linear in their parameters, including polynomial surrogates and linear interatomic potentials.

A single coefficient vector can minimize the total training loss while leaving systematic residuals on individual observations. POPS uses pointwise constrained fits to estimate how far those coefficients must vary to account for model-form error. Sampling whole coefficient vectors lets that uncertainty propagate to downstream observables.

- **`POPSRegression`** constructs the pointwise ensemble or its PCA-aligned hypercube.
- **`POPSEllipseRegression`** fits an ellipsoidal parameter distribution with optional empirical-Bayes or PAC-Bayes regularization.

[Method and tutorials](https://pops-uq.github.io/) · [Python documentation](https://pops-uq.github.io/implementations/python/) · [Kermode group regression demo](https://kermodegroup.github.io/demos/regression-demo.html) · [Standalone Julia implementation](https://github.com/POPS-UQ/POPSRegression.jl)

## Installation

Install this checkout with Python 3.10 or newer. The numerical dependencies are JAX, SciPy and NumPy.

```bash
git clone --branch jax-integration https://github.com/POPS-UQ/popsregression.git
cd popsregression
python -m pip install -e .
```

## Quick start

```python
import numpy as np
from popsregression import POPSRegression

x = np.linspace(-1, 1, 32)
X_train = np.polynomial.legendre.legvander(x, 4)
y_train = np.sin(2 * np.pi * x) + 0.3 * x**2
X_test = np.polynomial.legendre.legvander(np.linspace(-1, 1, 201), 4)

# Fit POPSRegression
# fit_intercept=False by default
model = POPSRegression()
model.fit(X_train, y_train)

# Prediction with misspecification & epistemic uncertainty
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

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `posterior` | `'hypercube'` | Posterior form: `'hypercube'` (PCA-aligned box) or `'ensemble'` (raw corrections); the ellipsoid is `POPSEllipseRegression` (below) |
| `random_state` | `None` | Seed for the posterior resampling; `None` uses the global NumPy state |
| `resampling_method` | `'uniform'` | Sampling method: `'uniform'`, `'sobol'`, `'latin'`, `'halton'` |
| `resample_density` | `1.0` | Number of posterior samples per training point |
| `prior_precision` | `None` | Shape `Λ` (vector or matrix) of the Gaussian prior `N(0, (λΛ)⁻¹)` on the coefficients; the scale `λ` is learned. `None` is the isotropic prior |
| `leverage_regularization` | `0.001` | Regularizes the leverage in the pointwise corrections, `e_i A⁻¹x_i / (h_i + eps)` with `eps` this times the mean leverage; `0.0` is the exact (unstable for low-leverage points) pointwise fit |
| `minimum_relative_correction` | `0.01` | Relative threshold on the correction norm: only points with \|\|dθ_i\|\| >= this times the RMS norm enter the singular decomposition and bounds of the POPS posterior |
| `mode_threshold` | `1e-8` | Eigenvalue threshold for hypercube dimensionality |
| `percentile_clipping` | `0.0` | Percentile to clip from hypercube bounds (0–50) |

The Bayesian ridge parameters of the mean fit (`max_iter`, `tol`, `alpha_1`,
`alpha_2`, `lambda_1`, `lambda_2`, `alpha_init`, `lambda_init`,
`fit_intercept`) are also supported.

> [!WARNING]
> `minimum_relative_error` is deprecated: training points are now selected by
> the norm of their pointwise correction rather than by their residual. A
> value passed to it is forwarded to `minimum_relative_correction` with a
> `FutureWarning`. `leverage_percentile`, `compute_score`, `copy_X` and
> `verbose` have been removed.

## Key attributes (after fitting)

| Attribute | Description |
|---|---|
| `coef_`, `intercept_` | Regression coefficients (posterior mean) and intercept |
| `sigma_` | Epistemic variance-covariance matrix |
| `misspecification_sigma_` | Misspecification variance-covariance matrix from POPS |
| `posterior_samples_` | Weight perturbations, with one sample per column |
| `alpha_` | Estimated noise precision (not used for prediction) |
| `pointwise_corrections_`, `leverage_`, `selected_` | Pointwise corrections, leverages and the points kept for the posterior |

## Ellipsoid posteriors: `POPSEllipseRegression`
[Run the PAC-Ellipse tutorial](docs/pac-ellipse.md) to fit a bounded simulator, inspect its risk certificate and propagate joint predictions.

`POPSEllipseRegression` fits a parameter distribution that is uniform on an
ellipsoid. For a linear model its predictive density at `x` has a closed form,
so the zero-noise generalization error is minimized directly. Predictive
densities, exact quantiles and parameter draws are available in closed form
(for the hierarchical layers, exactly for the stored finite mixture of
hyperparameter draws that represents the predictive). The `regularization` parameter chooses how the finite-data uncertainty
of the ellipsoid itself is handled:

```python
from popsregression import POPSEllipseRegression

bare = POPSEllipseRegression().fit(X_train, y_train)

# Empirical-Bayes layer (Ellipse+EB): a Laplace hyperposterior over the
# ellipsoid, with its prior centered on the fit. Cheap; its diagnostic_bound_
# is not a PAC bound.
eb = POPSEllipseRegression(regularization="empirical-bayes").fit(X_train, y_train)

# PAC layer (Ellipse+PAC): the prior over the ellipsoid is fixed on a random
# half of the data, and a PAC-Bayes bound on the population log risk is
# evaluated on the other half (and vice versa). y_bounds must be known in
# advance (e.g. from a maximum principle), not read off the data.
pac = POPSEllipseRegression(regularization="PAC", y_bounds=(y_lower, y_upper))
pac.fit(X_train, y_train, groups=case_id)   # groups: rows of one simulator case
pac.certificate_.raw_bound, pac.certificate_.trivial_bound  # bound for the returned predictive

lo, hi = eb.predict_interval(X_test, level=0.9545)   # exact central interval
logp = eb.predict_logpdf(X_test, y_test)
fields = eb.sample_predictions(X_test, 1000)          # joint draws for propagation
```

All predictions refer to parameter uncertainty only; no noise term is added.
The PAC bound certifies the log risk of the predictive density, not interval
coverage. See the
[ellipse documentation](docs/ellipse.md)
and the [glossary](docs/glossary.md) for the
terms used above.

## Comparison methods and paper studies

PACm, PAC²-T, predictive variational inference and a finite
POPS-dictionary weighting ablation are implemented in `examples/comparisons/`,
outside the package. The scripts in `examples/` produce every figure and table
of the accompanying paper; see [examples/README.md](examples/README.md).

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

scikit-learn is not required. The estimators keep the familiar
`fit` / `predict` / `get_params` / `set_params` usage, so they still work
inside scikit-learn pipelines and parameter searches when it is installed.

## Pipeline compatibility

If scikit-learn is installed, both estimators can be used in its pipelines and
hyperparameter search:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

pipe = make_pipeline(
    PolynomialFeatures(degree=4),
    POPSRegression(resampling_method='sobol'),
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

## Development

The repository is managed with [uv](https://docs.astral.sh/uv/); `uv run`
resolves the pinned environment from `uv.lock` on first use.

```bash
uv run --group test pytest -vsl popsregression examples/comparisons  # tests
uv run --group lint ruff check popsregression examples               # linter
uv run --group lint black --check popsregression examples
uv run --group doc mkdocs serve                    # docs at localhost:8000
cd examples && uv run --extra examples python example_polynomial.py  # a figure
```

Without uv, `pip install -e ".[examples]"` and run the tools directly.

## Citation

> *Parameter uncertainties for imperfect surrogate models in the low-noise regime*
>
> TD Swinburne and D Perez, [Machine Learning: Science and Technology 2025](https://doi.org/10.1088/2632-2153/ad9fce)

```bibtex
@article{swinburne2025,
    author={Swinburne, Thomas and Perez, Danny},
    title={Parameter uncertainties for imperfect surrogate models in the low-noise regime},
    journal={Machine Learning: Science and Technology},
    doi={10.1088/2632-2153/ad9fce},
    year={2025}
}
```

For the PAC-Bayes construction, see Thomas D. Swinburne, Omar Hussein, Ian Lynch and Danny Perez, [PAC-Bayesian regression for misspecified, near-deterministic surrogate models](https://openreview.net/forum?id=gD3LoosP0w).

## Contributors

Thomas D. Swinburne and Danny Perez developed the original POPS method. James Kermode contributed the Gaussian parameter prior for the ellipsoid shape and its initialization in [PR #15](https://github.com/POPS-UQ/popsregression/pull/15). The [JAX adaptation](https://github.com/POPS-UQ/popsregression/commit/8c129db) uses the prior scale learned by the Bayesian ridge fit. Enable it with `prior_kl=True` on `POPSEllipseRegression`.

## AI usage

Claude assisted with documentation and test cases. Thomas D. Swinburne reviewed the code described in the original release.
