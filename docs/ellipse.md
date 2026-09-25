# Ellipse posteriors

[`POPSEllipseRegression`][popsregression.POPSEllipseRegression] fits a linear
surrogate whose parameters are uniformly distributed on an ellipsoid. For a
linear model the [pushforward](glossary.md#pushforward-density) of that
distribution at an input `x` has a closed form. This gives the exact
zero-noise [generalization error](glossary.md#generalization-error), which the
fit minimizes directly, and exact predictive densities, quantiles and
parameter draws.

```python
from popsregression import POPSEllipseRegression

bare = POPSEllipseRegression().fit(X, y)
eb = POPSEllipseRegression(regularization="empirical-bayes").fit(X, y)
pac = POPSEllipseRegression(regularization="PAC", y_bounds=(y_lo, y_hi)).fit(X, y)
```

The three settings share the same ellipse fit. They differ in how the
finite-data uncertainty in the ellipse itself is handled.

| `regularization` | Hierarchical layer | Bound | When to use |
|---|---|---|---|
| `None` | none: the fitted ellipse | none | large N, or as a reference |
| `"empirical-bayes"` | [Laplace](glossary.md#laplace-approximation) hyperposterior over center and shape, prior centered on the fit | `diagnostic_bound_`, **not** a PAC bound | a cheap finite-data broadening |
| `"PAC"` | Gaussian hyperposterior over the [axis scales](glossary.md#axis-scale-hyperparameters) of a [pilot](glossary.md#pilot-split) ellipse | `certificate_`, a [PAC-Bayes bound](glossary.md#pac-bayes-bound) | when a finite-sample guarantee is needed |

## The fit

The ellipse is `{θ : (θ - μ)ᵀ B⁻¹ (θ - μ) ≤ 1}`. Its pushforward at `x` is
supported on `μ · F(x) ± a(x)`, with `a(x)² = F(x)ᵀ B F(x) + δ²`, and has the
density

```text
p(y | x) = C_P / a(x) · (1 - (y - μ·F(x))² / a(x)²)^((P-1)/2)
```

The training objective is the mean of `-log p(y_i | x_i)`. It combines a width
penalty `log a(x_i)` with a barrier that diverges when a training point
reaches the edge of the support. The optimization therefore enforces
[sample covering](glossary.md#covering-sample-and-population).

- The shape is `B = B0 + U Uᵀ` in whitened feature coordinates. `B0` is a
  fixed baseline from a [`POPSRegression`][popsregression.POPSRegression]
  hypercube pre-fit, and `U` has rank `r` (`rank`, default 32). Each
  objective and gradient evaluation costs `O(N P r)`.
- The barrier is approached by continuation (`rho_schedule`), with L-BFGS at
  each stage.
- `optimize_center=False` (default) keeps the center at the POPS/Bayesian
  ridge mean and fits the shape only.
- `delta` is the [width floor](glossary.md#width-floor): an output-offset
  coordinate of the parameter distribution, so the projected-ball dimension is
  `P + 1` and every half-width is at least `delta`.
- `preprocessor` optionally applies a scikit-learn transformer (e.g. a PCA
  projection) first; see [preprocessor](glossary.md#preprocessor-pilot-only-features).

Covering the training data does not imply covering new data. At small N a
bare ellipse typically misses a few percent of held-out targets, where its
density is zero. The two hierarchical layers address this.

## Predictions

All outputs refer to the [parameter-only predictive](glossary.md#parameter-only-predictive-no-aleatoric-term).
With a hierarchical layer this is the finite mixture of ellipse pushforwards
over the stored hyperparameter draws (`n_hyper_samples` per fold, default
1024). Quantiles, densities and CDFs are exact for that stored mixture, which
represents the continuous hyperposterior mixture by Monte Carlo.

```python
mean = model.predict(X)
mean, std = model.predict(X, return_std=True)
mean, y_max, y_min = model.predict(X, return_bounds=True)   # support envelope
lo, hi = model.predict_interval(X, level=0.9545)           # exact central interval
logp = model.predict_logpdf(X, y)                          # -inf outside the support
cdf = model.predict_cdf(X, y)
theta, a = model.sample(1000, random_state=0, return_offset=True)
fields = model.sample_predictions(X, 1000, random_state=0)  # (n_points, 1000)
```

For [propagation](glossary.md#propagation-qoi), use one draw (a column of
`theta` with its offset `a`, or a column of `fields`) for every input of a
field. These draws come from exactly the distribution that the intervals and
densities describe. With a `preprocessor`, `sample` is unavailable (each
fold has its own features); use `sample_predictions`.

## `regularization="empirical-bayes"`

A diagonal [Laplace](glossary.md#laplace-approximation) approximation of the
[Gibbs hyperposterior](glossary.md#gibbs-hyperposterior-and-temperature)
at temperature `N`, over the center and `U`. The hyperprior is centered on the
fitted optimum, with variance `hyperprior_scale · |Ψ̂|² / d`. The predictive is
therefore never narrower than the bare fit, and it converges to it as N grows.

The whitening, the baseline and the hyperprior all come from the same sample.
So `diagnostic_bound_` has the form of a PAC-Bayes right side but is **not** a
bound, and `certificate_status_` is `"diagnostic_empirical_bayes"`.

## `regularization="PAC"`

This path computes an actual PAC-Bayes bound on the population log risk of its
own predictive. It follows Theorem 1 of the paper. The theorem requires the
hyperprior, and every transformation entering the loss, to be fixed
independently of the data the bound is evaluated on.

1. The independent units are split at random into two halves. See
   [pilot split](glossary.md#pilot-split) and
   [independent units](glossary.md#independent-units-groups); pass
   `groups=` when several rows come from one simulator case.
2. On the pilot half an ellipse is fitted. It fixes any `preprocessor`
   (fitted on the pilot half only), the centering, the whitening, the pilot
   shape `V diag(λ₀) Vᵀ` and the width floor `a_min`.
3. The hyperparameters are the log multipliers `ω` of the pilot semi-axes,
   with a Gaussian hyperprior `N(0, τ²)`. `τ` is chosen from the predeclared
   grid `pac_log_scale_std`.
4. On the certification half, for each temperature of a predeclared grid,
   the hyperposterior is the Laplace approximation of the
   [Gibbs hyperposterior](glossary.md#gibbs-hyperposterior-and-temperature).
   The bound is evaluated for each one, and the smallest is kept, with the
   [union correction](glossary.md#union-bound-over-a-grid).
5. The bound is evaluated with:
   - the exact Gaussian KL;
   - a [Monte Carlo term](glossary.md#monte-carlo-term-empirical-bernstein)
     for the hyperposterior-averaged loss;
   - either the Seeger–Maurer [kl inequality](glossary.md#linear-theorem-1-and-kl-forms)
     (default) or the paper's linear Theorem 1 with a Hoeffding moment bound.
6. The predictive keeps `n_hyper_samples` fresh draws from each Gaussian
   hyperposterior. A last step carries the bound from the continuous mixture
   to this [stored mixture](glossary.md#stored-mixture-continuous-mixture),
   with its own failure probability `mixture_failure_probability`.
7. With `cross_fit=True` (default) the halves are swapped, the predictive is
   the equal mixture of both folds, and the reported bound is the average of
   the fold bounds. See [cross-fitting](glossary.md#cross-fitting).

The certified quantity is the population log risk of the
[floor-contaminated](glossary.md#floor-contamination-floor-weight) predictive,
`(1 - β) p(y | x) + β / R_y`, on the
[declared output interval](glossary.md#declared-output-interval) `y_bounds`.
`certificate_.raw_bound` applies to the stored mixture that `predict*`
returns; `certificate_.continuous_bound` applies to the continuous mixture.
It holds with probability at least `1 - failure_probability -
mc_failure_probability - mixture_failure_probability` (default 0.93) over the
training draw and the Monte Carlo draws, if:

- the units are i.i.d. draws from the deployment distribution, and
- `y_bounds` contains every possible output.

The software checks that the training outputs lie in `y_bounds`, and refuses
to fit otherwise. It cannot verify either assumption.

```python
pac = POPSEllipseRegression(regularization="PAC", y_bounds=(-2.6, 2.6))
pac.fit(X, y, groups=case_id)
cert = pac.certificate_
cert.raw_bound, cert.continuous_bound, cert.trivial_bound, cert.is_nonvacuous
cert.n_units, cert.pac_failure_probability, cert.mc_failure_probability
for fold in cert.folds:
    fold.n_units, fold.empirical, fold.monte_carlo, fold.kl, fold.complexity
    fold.lam, fold.moment_constant, fold.stored_mixture, fold.n_stored
```

What the bound does **not** say:

- It is not a coverage guarantee for `predict_interval`.
- It says nothing about the unfloored compact-support loss, which is infinite
  at uncovered points.
- It does not cover a test distribution that differs from the training one.

Coverage and width are reported separately in the
[studies](studies.md).

### Protocol constants

Every PAC parameter must be fixed before the data are seen:

- `y_bounds`, `floor_weight` (`β`, default 0.02);
- `min_half_width` (default 1% of the width of `y_bounds`);
- `pilot_fraction`, `cross_fit`;
- `lambda_fractions` (the temperature grid, as fractions of `N₁`);
- `pac_log_scale_std` (the hyperprior grid);
- `pac_inequality`;
- `failure_probability`, `mc_failure_probability`,
  `mixture_failure_probability`, `n_bound_samples`, `n_hyper_samples`;
- any `preprocessor` (it is refitted on each pilot half automatically).

A smaller `β` or `min_half_width` widens the loss range and loosens the bound.

### Cost of the guarantee

With few certification units the bound prefers a small temperature, so the
hyperposterior stays close to its prior and the predictive is broad. On the
quartic example with 10 training points, the 95.45% interval of Ellipse+PAC is
several times wider than that of Ellipse+EB. From a few tens of units onwards,
the two are comparable.
