popsregression
=================================================

[![tests](https://github.com/POPS-UQ/popsregression/actions/workflows/python-app.yml/badge.svg)](https://github.com/POPS-UQ/popsregression/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/POPS-UQ/popsregression/graph/badge.svg?token=L0XPWwoPLw)](https://codecov.io/gh/POPS-UQ/popsregression)
[![docs](https://img.shields.io/badge/docs-POPS--UQ.github.io%2Fpopsregression-blue)](https://POPS-UQ.github.io/popsregression)

**popsregression** is a [scikit-learn](https://scikit-learn.org) compatible
package providing `POPSRegression`, a Bayesian regression method for low-noise
data that accounts for model misspecification uncertainty.

**paper** *Parameter uncertainties for imperfect surrogate models in the low-noise regime* [Machine Learning: Science and Technology 2025](http://iopscience.iop.org/article/10.1088/2632-2153/ad9fce)

**Documentation** 📖 [POPS-UQ.github.io/popsregression](https://POPS-UQ.github.io/popsregression) — API reference and usage for this Python package

**The POPS method** 🔬 [pops-uq.github.io](https://pops-uq.github.io) — concepts, algorithm, tutorials and the [Julia implementation](https://github.com/POPS-UQ/POPSRegression.jl)

**Try it out!** [online demo from Kermode group](https://kermodegroup.github.io/demos/regression-demo.html) comparing multiple regression schemes.

## Misspecification-aware Bayesian regression 
Standard Bayesian regression (e.g. `BayesianRidge`) estimates epistemic and
aleatoric uncertainties, but provably ignore model misspecification- errors arising from limited model form (see example below). In the low-noise (weak aleatoric / near-deterministic) limit, weight uncertainties (`sigma_`) are significantly underestimated as they only capture epistemic uncertainty, which decays with increasing data. Any remaining error is attributed to aleatoric noise (`alpha_`), which is erroneous in low-noise settings.

`POPSRegression` efficiently estimates **model misspecification uncertainty**
via the Pointwise Optimal Parameter Sets (POPS) algorithm, finidng parameter perturbations that would fit each training point exactly. 
The result is wider, more honest uncertainty estimates that properly cover the true function, even when the model class cannot perfectly represent the target.

The misspecified, near-deterministic regression problem that `POPSRegression` addresses is particularly relevant to the fitting of surrogate simulation models in computational science, i.e. interatomic potentials,where by construction the optimal surrogate model is structurally unable to capture the target function exactly.

## Example
Fitting a quartic polynomial (P=5 parameters) to a complex oscillatory function with N=10 (top row) and N=100 (bottom row) training points. Columns are BayesianRidge, the POPS hypercube, the POPS ellipse, the POPS ellipse with its empirical-Bayes Laplace layer (Ellipse+EB), and standalone Bayesian stacking (a leave-one-out stacked mixture of ordinary Bayesian linear regressions of degree 1 to 4). No band includes an aleatoric term: every method shows the pushforward of its parameter distribution alone (BayesianRidge uses `sigma_` only and its fitted `alpha_` is never used; Bayesian stacking uses the parameter-only Student-t predictive and its fitted residual scales are never used). The orange band is the 95.45% interval, the grey band the max/min posterior envelope (`±4σ` for BayesianRidge, the exact 99.9% interval for Bayesian stacking), and each panel reports the fraction of the truth covered by the outer band. BayesianRidge epistemic uncertainty vanishes with more data, while POPS maintains uncertainty where the polynomial deviates from the truth.

![Example comparison of BayesianRidge vs POPS uncertainty](https://raw.githubusercontent.com/POPS-UQ/popsregression/main/examples/example_polynomial.png)

The figure is produced by [examples/example_polynomial.py](examples/example_polynomial.py);
[examples/example_bayesian_stacking.py](examples/example_bayesian_stacking.py)
repeats the comparison over independent training draws (test log loss,
coverage and width of exact predictive intervals), and
[examples/example_low_noise_objectives.py](examples/example_low_noise_objectives.py)
compares POPS with the PACm-Bayes and PAC²-T objectives as the likelihood
width vanishes. See [Baselines](https://POPS-UQ.github.io/popsregression/comparisons/).

## Installation

```bash
pip install popsregression
```

**Dependencies**: scikit-learn >= 1.6.1, scipy >= 1.6.0, numpy >= 1.20.0

## Quick start

```python
from popsregression import POPSRegression

X_train, X_test, y_train, y_test = ...

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
| `minimum_relative_error` | `0.01` | Relative residual threshold: only points with \|y - Xw\| >= this times the fit RMSE contribute to the POPS posterior |
| `mode_threshold` | `1e-8` | Eigenvalue threshold for hypercube dimensionality |
| `percentile_clipping` | `0.0` | Percentile to clip from hypercube bounds (0–50) |

All `BayesianRidge` parameters (`max_iter`, `tol`, `alpha_1`, `alpha_2`,
`lambda_1`, `lambda_2`, `fit_intercept`, etc.) are also supported.

> [!WARNING]
> `leverage_percentile` is deprecated since 0.5 and will be removed in 0.7.
> Training points are now selected by relative residual magnitude: use
> `minimum_relative_error` instead (`leverage_percentile=0.0` becomes
> `minimum_relative_error=0.0`). Passing it raises a `FutureWarning` and has
> no effect.

## Key attributes (after fitting)

| Attribute | Description |
|---|---|
| `coef_` | Regression coefficients (posterior mean) |
| `sigma_` | Epistemic variance-covariance matrix |
| `misspecification_sigma_` | Misspecification variance-covariance matrix from POPS |
| `posterior_samples_` | Samples from the POPS posterior |
| `alpha_` | Estimated noise precision (not used for prediction) |

## Ellipsoid posteriors: `POPSEllipseRegression`

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
[ellipse documentation](https://POPS-UQ.github.io/popsregression/ellipse/)
and the [glossary](https://POPS-UQ.github.io/popsregression/glossary/) for the
terms used above.

## Comparison methods and paper studies

Bayesian stacking, PACm, PAC²-T, predictive variational inference and a finite
POPS-dictionary weighting ablation are implemented in `examples/comparisons/`,
outside the package. The scripts in `examples/` produce every figure and table
of the accompanying paper; see [examples/README.md](examples/README.md).

## Pipeline compatibility

`POPSRegression` is fully compatible with scikit-learn pipelines and
hyperparameter search:

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    PolynomialFeatures(degree=4),
    POPSRegression(resampling_method='sobol'),
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

## Documentation

https://POPS-UQ.github.io/popsregression

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
> TD Swinburne and D Perez, [Machine Learning: Science and Technology 2025](http://iopscience.iop.org/article/10.1088/2632-2153/ad9fce)

```bibtex
@article{swinburne2025,
    author={Swinburne, Thomas and Perez, Danny},
    title={Parameter uncertainties for imperfect surrogate models in the low-noise regime},
    journal={Machine Learning: Science and Technology},
    doi={10.1088/2632-2153/ad9fce},
    year={2025}
}
```

## AI Usage
Claude was used to produce full documentation and some test cases. All code was reviewed by a human (Tom)
