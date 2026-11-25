# PAC-Ellipse regression with JAX

Use `POPSEllipseRegression(regularization="PAC")` when you want an ellipsoid predictive distribution and a finite-sample certificate for its population log risk. Install the [JAX integration branch](index.md#installation) to run this example.

## Fit a bounded simulator response

The simulator below returns a sine, so its output lies in `[-1, 1]` for every input. We declare that bound before sampling. Each row comes from an independent uniform input, so one row is one certification unit.

```python
import numpy as np
from popsregression import POPSEllipseRegression

rng = np.random.default_rng(42)
x = rng.uniform(-1, 1, size=96)
X = np.polynomial.legendre.legvander(x, 4)
y = np.sin(2 * np.pi * x)

model = POPSEllipseRegression(
    regularization="PAC", y_bounds=(-1.0, 1.0), random_state=7
).fit(X, y)
cert = model.certificate_
print("Independent cases:", cert.n_units)
print("Stored-mixture bound:", cert.raw_bound)
print("Trivial loss ceiling:", cert.trivial_bound)
print("Non-vacuous:", cert.is_nonvacuous)
print("Confidence level:", 1 - cert.failure_probability)
```

If multiple rows come from each simulation, pass one case identifier per row via `model.fit(X, y, groups=case_id)`. This keeps correlated outputs together in the pilot/certification split. Any fitted feature transformation must also respect the split. Pass it as the estimator's `preprocessor` so each pilot fold fits its own transformation.

## Interpret the bound

`cert.raw_bound` bounds expected negative log density for a new independent case, scored with the uniform floor mixed into the stored predictive on `y_bounds`. The certificate is informative when this bound is below `cert.trivial_bound`. A vacuous certificate means the calculation did not improve on the loss ceiling at this sample size. The total failure budget includes the PAC inequality and both Monte Carlo steps.

## Propagate the fitted model

```python
x_eval = np.array([-0.4, 0.6])
X_eval = np.polynomial.legendre.legvander(x_eval, 4)
lo, hi = model.predict_interval(X_eval, level=0.95)
fields = model.sample_predictions(X_eval, 2000, random_state=42)
change = fields[1] - fields[0]
print("Pointwise intervals:", lo, hi)
print("Response difference, central 95%:", np.quantile(change, [0.025, 0.975]))
```

Each column of `fields` uses one model draw at both inputs. The quantiles of `change` therefore retain dependence between the two predicted responses. These are predictive intervals under the fitted mixture. The certificate bounds its floor-contaminated log risk, which is a different quantity from interval coverage.

For the pilot fit, hyperprior, cross-fitting protocol, and the distinction between continuous and stored mixtures, see [Ellipse posteriors](ellipse.md). The [API reference](api.md) documents the estimator and certificate fields.
