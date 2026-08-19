# Ellipsoid posteriors

[`POPSRegression(posterior="ellipsoid")`][popsregression.POPSRegression] fits a
linear model whose parameter posterior is **uniform on an ellipsoid**, obtained
by directly minimizing the empirical generalization error of the exact
predictive pushforward. The POPS covering condition becomes a log-barrier
inside a smooth objective, so the ellipsoid is found by optimization rather
than sampling, and `pac_bayes=True` adds a hierarchical PAC-Bayes layer
entirely in closed form.

## Model

The posterior over weights is uniform on an ellipsoid,

$$
\pi_\Psi(\theta) = \mathrm{Unif}\{\theta :
(\theta-\mu)^\top B^{-1} (\theta-\mu) \le 1\},
\qquad \Psi = (\mu, B),\ B \succeq 0,
$$

equivalently $\theta = \mu + Lz$ with $z$ uniform on the unit $P$-ball and
$B = LL^\top$. For a feature vector $\phi_i$, the pushforward of
$\pi_\Psi$ under $\theta \mapsto \phi_i \cdot \theta$ is the exact
**projected-ball density** with mean $m_i = \phi_i \cdot \mu$ and squared
half-width $s_i = \phi_i^\top B \phi_i$:

$$
p(t) = \frac{C_P}{a}\Big(1 - \tfrac{(t-m)^2}{a^2}\Big)^{k},
\qquad k = \tfrac{P-1}{2},\quad
C_P = \frac{\Gamma(P/2+1)}{\sqrt{\pi}\,\Gamma((P+1)/2)}.
$$

With residual $r_i = y_i - m_i$, width floor $v_i = s_i + \delta^2$ and
$q_i = 1 - r_i^2 / v_i$, the per-datum loss (negative log predictive) is

$$
\ell_i(\Psi) = \tfrac{1}{2}\log v_i - \log C_P - k\, L_\rho(q_i),
$$

where $L_\rho$ is a smooth continuation of $\log$ (exact for
$q \ge \rho$, quadratic below). The term $\tfrac12 \log v_i$ penalizes
ellipsoid width while $-k\,L_\rho(q_i)$ is a log-barrier enforcing the POPS
covering condition $|r_i| < \sqrt{v_i}$ — the scalar image of "the ellipsoid
intersects every pointwise optimal parameter set". Minimizing the empirical
generalization error $\hat G(\Psi) = \tfrac1N \sum_i w_i \ell_i(\Psi)$ over a
decreasing schedule of $\rho$ is therefore an **interior-point method for
POPS coverage**.

## Whitened low-rank parameterization

The ellipsoid is never represented as a dense $P \times P$ matrix during
fitting. Features are whitened with
$W = (\Phi^\top\Phi/N + \lambda_w I)^{-1/2}$ and the shape matrix is
parameterized in whitened coordinates as

$$
B_t = B_{0,t} + U U^\top, \qquad U \in \mathbb{R}^{P \times r},
$$

with a fixed baseline $B_{0,t}$ (`baseline='pops'` uses the whitened
covariance of a `POPSRegression` hypercube pre-fit, which also provides the
warm start and the hyperprior center). All fit operations are
$O(N P r)$; gradients are closed form (no autodiff, no sampling). When
`fit_intercept=True` the intercept coordinate is appended **after**
whitening and recovered by the corresponding affine correction.

By default (`optimize_center=False`) the center $\mu$ is frozen at the
POPS pre-fit coefficients — `coef_` is the familiar
`BayesianRidge`-style mean — and only the widths are optimized. With
`optimize_center=True` the center is optimized jointly with the widths;
its stationarity condition is then a heteroscedastic weighted least
squares under the fitted widths (per-point precision $1/(q_i v_i)$):
the mean is pinned where the ellipsoid pinches and relaxed where the
misspecification width is large, so it deliberately differs from an
OLS/`BayesianRidge` mean and the fit is tighter but less conservative
at small $N$.

## PAC-Bayes layer

With `pac_bayes=True` the Catoni/Gibbs hyperposterior
$\pi_H^*(\Psi) \propto \pi_{0H}(\Psi)\, e^{-N \hat G(\Psi)}$ is followed via
its Laplace approximation:

- the **hyperprior** is $\mathcal{N}(\psi_0, \tau^2 I)$;
  `hyperprior_scale` sets $\tau^2$ *relative* to the center scale
  ($\tau^2 =$ `hyperprior_scale` $\cdot\, \|\psi_0\|^2/d$), so the
  default is independent of the units of $y$. With the default
  `hyperprior_center='phase1'`, $\psi_0$ is the phase-1 optimum itself:
  the MAP coincides with the bare fit, and the hyperposterior spread
  **strictly broadens** the predictive bounds, concentrating on the bare
  values at rate $N$ — never narrower than the bare ellipse;
- with `hyperprior_center='warm_start'` (the POPS warm start with zero
  low-rank block), the **MAP** adds a ridge
  $\|\psi-\psi_0\|^2 / (2N\tau^2)$ to the phase-1 objective and is
  shrunk toward the baseline ellipsoid (`hyperprior_scale=np.inf`
  recovers the unregularized fit exactly);
- the **covariance** is the diagonal
  $\Sigma_H^{-1} = N\,\mathrm{Hess}[\hat G](\psi^*) + I/\tau^2$, using the
  exact per-datum Hessian diagonal;
- the **KL**, averaged empirical error and PAC bound (`kl_`,
  `empirical_H_`, `bound_`) are closed form. The bound holds for *all*
  hyperposteriors simultaneously, so evaluating it at the Laplace Gaussian
  is rigorous — the Laplace step costs tightness, never validity;
- optionally (`update_hyperprior=True`, requires
  `hyperprior_center='warm_start'`) $\tau^2$ is updated by a
  Tipping/MacKay evidence iteration following the conventions of
  `sklearn.linear_model.BayesianRidge`.

The hyperposterior enters prediction in two documented ways: `y_std`
averages the pushforward over it, and `return_bounds` returns the
max/min over the ensemble of ellipses within the 2σ range of the
hyperposterior, `mean ± (sqrt(v) + 2·y_bound_std)` — strictly broader
than the fitted ellipse's own support (recovered as
`bounds ∓ 2·y_bound_std`), reverting to it at rate $N$.

For conservative uncertainty in the scarce-data regime (N/P of order
one), simply use `pac_bayes=True` on the default (frozen-center)
configuration: it keeps the mean at the POPS pre-fit and takes the bounds
over the 2σ hyperposterior ensemble of optimized-width ellipses.

## Reaching it

The ellipsoid is one of the three `posterior` choices on the single
[`POPSRegression`][popsregression.POPSRegression] estimator, so it composes
with everything else on that estimator:

```python
POPSRegression(posterior="ellipsoid")                   # bare ellipsoid
POPSRegression(posterior="ellipsoid", pac_bayes=True)   # + PAC-Bayes layer
```

Its own tuning parameters — `rank`, `delta`, `baseline`, `rho_schedule`,
`optimize_center`, and the PAC-Bayes settings — go through the
`posterior_options` dict, listed in full under
[Ellipsoid parameters](#ellipsoid-parameters) below. `pac_bayes`,
`fit_intercept` and `random_state` are set on the estimator itself, and
sample weights are passed to `fit`. After fitting, the ellipsoid is exposed
as `ellipsoid_`.

## Quick start

```python
from popsregression import POPSRegression

X_train, X_test, y_train, y_test = ...

# Defaults: baseline='pops', optimize_center=False (mean = POPS pre-fit)
model = POPSRegression(posterior="ellipsoid")
model.fit(X_train, y_train)

# Predictive std of the pushforward: sqrt(v / (P + 2))
y_pred, y_std = model.predict(X_test, return_std=True)

# Support bounds: mean +/- sqrt(v) for a bare fit; with pac_bayes=True
# the max/min over the 2-sigma hyperposterior ensemble of ellipses,
# mean +/- (sqrt(v) + 2 * y_bound_std) -- strictly broader than bare
y_pred, y_std, y_max, y_min = model.predict(
    X_test, return_std=True, return_bounds=True
)

# Hyperposterior std of the bound curves (zero for a bare fit); the
# fitted ellipse's own support is bounds -/+ 2 * y_bound_std
y_pred, y_max, y_min, y_bound_std = model.predict(
    X_test, return_bounds=True, return_bound_std=True
)

# Posterior parameter draws (affine map of uniform ball samples), also
# available as model.posterior_samples_ (perturbations about the mean)
theta_samples = model.ellipsoid_.sample(1000)

# Closed-form PAC-Bayes layer
model = POPSRegression(
    posterior="ellipsoid",
    pac_bayes=True,
    posterior_options={"update_hyperprior": True, "hyperprior_center": "warm_start"},
)
model.fit(X_train, y_train)
print(model.bound_, model.kl_, model.ellipsoid_.tau2_)
```

!!! note "std vs bounds convention"
    `return_std` returns the predictive standard deviation of the
    projected-ball pushforward, $\sqrt{v/(P+2)}$ — *not* the half-width.
    `return_bounds` returns the support, $\text{mean} \pm \sqrt{v}$. The
    ellipsoid shape matrix `ellipsoid_B_` relates to the parameter
    covariance as $\mathrm{Cov} = B/(P+2)$.

## Example: bounds that do not shrink

The misspecified oscillatory example (see
[`examples/example_polynomial.py`](https://github.com/POPS-UQ/popsregression/blob/main/examples/example_polynomial.py),
rendered on the [Example](example.md) page) fits a quartic polynomial to an
oscillatory target at $N = 10$ and $N = 100$ training points. `BayesianRidge`
epistemic uncertainty vanishes as $N$ grows; the ellipsoid bounds track the
`POPSRegression` hypercube bounds — but are obtained by direct optimization of
the generalization-error objective, retaining the misspecification uncertainty
at any $N$.

## Ellipsoid parameters

The ellipsoid is implemented by an internal engine that
[`POPSRegression`][popsregression.POPSRegression] fits, exposes as
`ellipsoid_`, and forwards `posterior_options` to. Its parameters below are
exactly the keys `posterior_options` accepts, and its fitted attributes are
reachable through `ellipsoid_` (the headline ones — `coverage_fraction_`,
`objective_`, `rank_`, and `bound_`, `empirical_H_`, `kl_`, `gamma_` under
`pac_bayes=True` — are copied onto the estimator itself).

Fitting and prediction are documented on the
[API reference](api.md) page: call `POPSRegression.fit` and
`POPSRegression.predict`, not the engine's own methods.

::: popsregression._ellipse._EllipsoidPosterior
    options:
      members: false

### Sampling

::: popsregression._ellipse._EllipsoidPosterior.sample
    options:
      show_root_heading: true
      show_root_full_path: false
