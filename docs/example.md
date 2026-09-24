# Example: POPS vs Bayesian ridge

A misspecified, noise-free fit of a quartic polynomial to an oscillatory
function, comparing
[`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html),
the POPS hypercube ([`POPSRegression`][popsregression.POPSRegression]) and the
three settings of
[`POPSEllipseRegression`][popsregression.POPSEllipseRegression]. The full
script is
[`examples/example_polynomial.py`](https://github.com/POPS-UQ/popsregression/blob/main/examples/example_polynomial.py).
Terms are defined in the [glossary](glossary.md).

## The problem

```python
import numpy as np


def target(x):
    return (x**3 + 0.01 * x**4) * 0.1 + np.sin(x) * x * 10.0


def features(x):
    return np.vander(x, 5, increasing=True)   # 1, x, ..., x^4  (P = 5)


rng = np.random.RandomState(1042)
x_train = np.sort(np.append(rng.uniform(-10, 10, 10), [-10.0, 10.0]))
X_train, y_train = features(x_train), target(x_train)
x_dense = np.linspace(-10, 10, 401)
X_dense = features(x_dense)
```

No quartic reproduces the target, whatever the amount of data: the model is
[misspecified](glossary.md#surrogate-engine-and-misspecification), and the
data carry no noise.

## Fitting

```python
from sklearn.linear_model import BayesianRidge
from popsregression import POPSEllipseRegression, POPSRegression

ridge = BayesianRidge(fit_intercept=False).fit(X_train, y_train)
hypercube = POPSRegression(minimum_relative_error=0.0).fit(X_train, y_train)
ellipse = POPSEllipseRegression().fit(X_train, y_train)
eb = POPSEllipseRegression(regularization="empirical-bayes").fit(X_train, y_train)
# The target lies in [-144.4, 136.9] on [-10, 10]; y_bounds is known in advance.
pac = POPSEllipseRegression(regularization="PAC", y_bounds=(-160, 160))
pac.fit(X_train, y_train)
```

## Intervals

Every panel of the figure shows the same two exact central intervals of the
[parameter-only predictive](glossary.md#parameter-only-predictive-no-aleatoric-term),
95.45% and 99.9%:

```python
lo, hi = eb.predict_interval(X_dense, level=0.9545)           # ellipse family
mean = ridge.predict(X_dense)                                  # Bayesian ridge:
std = np.sqrt(np.einsum("ij,jk,ik->i", X_dense, ridge.sigma_, X_dense))
draws = hypercube.predict(X_dense)[:, None] + X_dense @ hypercube.posterior_samples_
```

Bayesian ridge uses the weight posterior `sigma_` only; its fitted noise
precision `alpha_` is never used. The hypercube intervals are empirical
quantiles of its parameter draws.

![Quartic surrogate](images/example_polynomial.png)

Rows are N = 10 and N = 100 training points (plus the two endpoints). The
labels give the held-out coverage of both intervals on a dense grid.

- The Bayesian ridge uncertainty contracts as N grows (95.45% coverage 0.16
  and 0.09), although the approximation error does not.
- The POPS posteriors keep a finite width.
- At N = 10 the bare ellipse misses part of the domain. Ellipse+EB and
  Ellipse+PAC restore the coverage.
- Ellipse+PAC is much broader at N = 10: with 6 certification units per fold,
  its bound can only justify a hyperposterior close to its prior. At N = 100
  it is comparable to Ellipse+EB, with a non-vacuous bound of 7.0 nats against
  a trivial ceiling of 9.7.

Repeated-split versions of this comparison, including Bayesian stacking and
PVI, are in [Studies](studies.md).
