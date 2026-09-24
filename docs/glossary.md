# Glossary

Terms used in these pages, the docstrings and the example scripts. Each entry
has an anchor, so other pages link to it directly, e.g.
[pilot split](glossary.md#pilot-split).

## Models and predictive distributions

### Surrogate, engine and misspecification
The *engine* is the expensive simulation being approximated; the *surrogate*
is a linear model `y ≈ θ · F(x)` of it. The surrogate is *misspecified* when no
single parameter vector `θ` reproduces the engine everywhere, so some
approximation error remains however much data is used.

### Near-deterministic
The engine's output has negligible random noise, so the observed error is
almost entirely approximation error of the surrogate.

### Parameter distribution, posterior
A probability distribution over the surrogate parameters `θ`. Each draw of `θ`
is a complete surrogate function, which is what makes propagation to derived
quantities possible (see [propagation](#propagation-qoi)).

### POPS hypercube, POPS ensemble
The two posteriors of [`POPSRegression`][popsregression.POPSRegression]. For
each training point, the *pointwise optimal parameter set* is the set of `θ`
that fit that point exactly. The ensemble keeps one pointwise-corrected `θ` per
training point. The hypercube fits an axis-aligned box to those corrections, in
their principal-component coordinates, and samples it uniformly.

### Ellipse posterior
The posterior of
[`POPSEllipseRegression`][popsregression.POPSEllipseRegression]: `θ` is
uniform on an ellipsoid with center `μ` and shape matrix `B`. It is fitted by
minimizing the [empirical generalization error](#generalization-error) of its
[pushforward](#pushforward-density).

### Pushforward density
The distribution of the prediction `θ · F(x)` when `θ` is drawn from the
parameter distribution. For a uniform ellipsoid it is a *projected-ball*
density: supported on `μ · F(x) ± a(x)`, with half-width
`a(x) = sqrt(F(x)ᵀ B F(x) + δ²)`. It vanishes outside that interval, i.e. it
has *compact support*.

### Width floor δ
A small constant added inside every half-width, `a(x)² = F(x)ᵀ B F(x) + δ²`. It
keeps the objective finite and gives every predictive density an upper bound
`C_P / δ`.

### Parameter-only predictive (no aleatoric term)
Every prediction, interval and score in this package uses the pushforward of
the parameter distribution alone. No fitted observation-noise variance is
added, for POPS or for any comparison method: in the near-deterministic
setting, all error is attributed to the parameters.

### Covering (sample and population)
A parameter distribution *covers* a data point if the point lies inside the
support of its pushforward, i.e. the predictive density there is positive.
*Sample covering* means every training point is covered. *Population covering*
means the same holds for almost every point the engine can produce, which
sample covering does not guarantee.

### Propagation, QoI
A *quantity of interest* (QoI) is a derived quantity computed from a whole
surrogate field, e.g. the dissipation of a Burgers solution. It is propagated
by computing the QoI once per parameter draw, reusing that draw for the whole
field.

## Learning objectives

### Generalization error
`G[π] = E[-log p_π(Y | X)]`: the expected negative log density of the
predictive `p_π` at a new data point. Its training-sample average is the
*empirical generalization error* `Ĝ_N`. The ellipse posterior minimizes `Ĝ_N`
directly, in closed form at zero noise.

### Expected loss
`L[π] = ∫ E[-log p_θ(Y | X)] dπ(θ)`: the loss of a single draw, averaged over
draws. Standard Bayesian inference targets `L`. By Jensen's inequality
`L ≥ G`. The minimizer of `L` concentrates on a single `θ`, which is why
Bayesian parameter uncertainty collapses under misspecification.

### Hyperparameters Ψ, hyperprior, hyperposterior
The ellipse itself is described by hyperparameters `Ψ` (its center and shape).
A distribution over `Ψ` is a *hyperposterior* `π_H`. A reference distribution
over `Ψ` that is fixed before the data it is scored on is a *hyperprior*
`π_0H`. The predictive of a hyperposterior is the mixture of ellipse
pushforwards over `Ψ`.

### Gibbs hyperposterior and temperature λ
The distribution `π_H(Ψ) ∝ π_0H(Ψ) exp(-λ Ĝ_N(Ψ))`, which minimizes the
PAC-Bayes right side at a fixed *temperature* `λ`. `λ = N` corresponds to an
ordinary Bayesian update; smaller `λ` keeps the hyperposterior closer to the
hyperprior.

### Laplace approximation
A Gaussian approximation of a distribution: centered on its mode, with
covariance given by the inverse Hessian of the negative log density there.
Here the diagonal of that Hessian is used.

### Ellipse+EB (empirical Bayes)
`POPSEllipseRegression(regularization="empirical-bayes")`: a Laplace
hyperposterior whose hyperprior is centered on the ellipse fitted to the same
data. It broadens the predictive at small N. Because the prior is chosen after
seeing the data, its `diagnostic_bound_` is not a PAC bound.

### Ellipse+PAC
`POPSEllipseRegression(regularization="PAC")`: the construction for which the
package computes an actual PAC-Bayes bound. See
[PAC-Bayes bound](#pac-bayes-bound) and the protocol terms below.

## PAC-Bayes bounds

### PAC-Bayes bound
A bound on the population risk of a (hyper)posterior that holds with
probability at least `1 - ξ` over the draw of the training data,
simultaneously for every hyperposterior. It has the form
"empirical risk + complexity term + concentration term". The complexity term
is a KL divergence from the hyperprior, which must not depend on the data it
is evaluated on. `ξ` is the *failure probability*.

### Floor contamination, floor weight β
The compact-support pushforward can give infinite log loss at uncovered
points. The bound is therefore stated for the proper density
`(1 - β) p(y | x) + β / R_y` on the [declared output interval](#declared-output-interval),
of width `R_y`, whose log loss is bounded. The floor enters only the
certified loss, never the predictions.

### Declared output interval
`y_bounds`: an interval known, from the physics or the simulator, to contain
every output the engine can produce (e.g. a maximum principle). It must not be
read off the training sample. It fixes `R_y` and the loss ceiling.

### Minimum half-width
`min_half_width`, written `a_min`: the width floor used by the PAC path. Every
pushforward half-width is at least `a_min`, so the density is at most
`C_P / a_min` and the floored loss lies in a known interval `[a_β, b_β]`.

### Trivial bound, non-vacuous
The floored loss can never exceed `b_β = log(R_y / β)`. A PAC bound below this
ceiling is *non-vacuous*, i.e. informative.

### Pilot split
The PAC path randomly divides the independent units into two halves. The
*pilot* half fixes everything the bound treats as given in advance: centering
and whitening, the pilot ellipse, the hyperprior over `Ψ`. The *certification*
half is then used only to fit the hyperposterior and to evaluate the bound, so
the hyperprior is independent of the data the bound is computed on.

### Cross-fitting
With `cross_fit=True` the roles of the two halves are also swapped. The
predictive is the equal mixture of both folds, and the bound is the average of
the two fold bounds, each at half the failure probability. Every observation
then informs the final predictor.

### Independent units, groups
The sample size in the bound counts independent draws, not rows of `X`. Rows
from one simulator case (e.g. several spatial points of one Burgers run) form
one unit, passed via `groups=`. Their losses are averaged before the bound is
formed, and the split never separates them.

### Axis-scale hyperparameters
The PAC hyperposterior is over the log lengths of the pilot ellipse's
principal semi-axes, and optionally over center shifts along them. This keeps
the hyperparameter dimension at `P` (or `2P`) instead of `P (1 + r)`, so the KL
cost of a concentrated hyperposterior grows with `P` only.

### Linear (Theorem 1) and kl forms
Two PAC-Bayes inequalities are available for the PAC path.
- `pac_inequality="linear"` is the paper's Theorem 1, with the Hoeffding bound
  `λ R² / (8 N₁)` for the moment term of a loss of range `R`.
- `pac_inequality="kl"` (default) is the Seeger–Maurer bound
  `kl(Ĝ ‖ G) ≤ (KL + log(2 sqrt(N₁) / ξ)) / N₁`, for the loss rescaled to
  `[0, 1]`. It is usually tighter with few units.

Both lift a standard PAC-Bayes inequality to the hyperparameters and then use
the Jensen step `G[mixture] ≤ average of G[Ψ]`.

### Union bound over a grid
Choosing the temperature `λ`, or the hyperprior width, from a predeclared grid
after seeing the data is valid if the confidence term includes the log of the
grid size.

### Monte Carlo term (empirical Bernstein)
The hyperposterior average of `Ĝ` is estimated from independent draws of `Ψ`.
The bound adds an empirical-Bernstein upper deviation for that estimate, with
its own failure probability `mc_failure_probability`.

### Jensen gap
The difference between the averaged loss `Σ_k q_k Ĝ(Ψ_k)` and the loss of the
mixture predictive, `-log Σ_k q_k p_{Ψ_k}`. It is non-negative. The bound
controls the former, which upper-bounds the latter.

## Loss fluctuations

### One-sided moment, CGF
The term `Ψ_0H` in Theorem 1 is built from the cumulant generating function
`ψ(t) = log E exp{t [G(Ψ) - ℓ(Ψ, Z)]}` of the per-datum loss. Only
`0 < t = λ / N ≤ 1` is needed. It measures how much a lucky sample can
understate the risk.

### J, the Jensen moment term
`J(Ψ) = log E p_Ψ(Y | X) - E log p_Ψ(Y | X) ≥ 0`, estimable from held-out
data. Concavity gives `ψ(t) ≤ t J(Ψ)`, with equality at `t = 1`, so at `λ = N`
the moment term of Theorem 1 is exactly `log E_{π_0H} exp{N J(Ψ)}`. `J` is zero
for a predictive whose density at the data is constant, about 0.153 nats for a
calibrated fixed-width Gaussian, and infinite when population covering fails.

### Sub-Gaussian, sub-gamma
Envelopes for the moment term: `ψ(t) ≤ t² s² / 2` (sub-Gaussian) or
`ψ(t) ≤ t² s² / (2 (1 - c t))` (sub-gamma, as in Corollary 1). For the
one-sided moment, a bounded predictive density suffices for a sub-gamma
envelope, with `s² = Var(ℓ)` and `c = (G - a_Ψ) / 3`, where `a_Ψ` is the
smallest possible loss. Heavy upper tails of the loss do not enter `c`.

## Evaluation and comparison methods

### Exact central interval, coverage
The interval between the `(1 - level) / 2` and `(1 + level) / 2` quantiles of a
predictive distribution: computed from the mixture CDF for the ellipse family,
from Gaussian quantiles for Bayesian ridge, PVI, PACm and PAC²-T, and from parameter draws
for the hypercube. *Coverage* is the fraction of held-out targets inside it.
The same levels (95.45% and 99.9%) are used for every method.

### Interval score
The Gneiting–Raftery score of a central interval:
`width + (2 / α) × (distance by which the target falls outside)`, with
`α = 1 - level`. It is a proper score that rewards narrow intervals that still
cover, and is finite for every method. Lower is better.

### Shared design matrix
Every method is fitted on the same design matrix. The comparison methods
rescale features and target but never center them, since centering would
quietly add an intercept that the POPS methods and Bayesian ridge do not
have. Where an intercept is wanted (ACE), it is an explicit constant column
of the design, so every method treats it the same way.

### Floored NLL
The test log loss of `(1 - b) p + b / R_y` for a common small `b`. It is finite
even for compact-support predictives.

### Bayesian stacking
The comparison of Yao et al. (2018): ordinary Bayesian linear regressions on
predeclared feature subsets, combined with global weights chosen by
cross-validated predictive log score. No POPS construction is involved.
Implemented in `examples/comparisons/bayesian_stacking.py`.

### Finite POPS dictionary, POPS-dictionary stacking
A supplementary ablation (`examples/comparisons/pops_dictionary.py`). A pilot
ellipse is rescaled into a finite list of candidate ellipses, and categorical
weights over them are chosen either by the Gibbs rule of the bound or by
stacking. It isolates the effect of the weight-selection rule. It is not an
alternative to POPS.

### PACm, PAC²-T
Finite-noise PAC-Bayes objectives from prior work.
- *PACm* (Morningstar et al. 2022) averages the likelihood over `m` parameter
  draws.
- *PAC²-T* (Masegosa 2020) adds a second-order variance correction.

Both become singular as the likelihood width goes to zero. Implemented in
`examples/comparisons/low_noise_objectives.py`.

### PVI
Predictive variational inference (Lai, Linero and Yao, ICML 2026). It
maximizes the training log score of the predictive,
`log ∫ p(y | θ) q(θ) dθ`, optionally minus `λ KL(q ‖ prior)`: it minimizes
`Ĝ_N` with a parameter-level penalty. The published code estimates the
predictive integral with `s` fresh Monte Carlo draws per step,
`log (1/s) Σ_j p(y | θ_j)`, which is the PACm objective; its class is even
named `PACMVIBasic`.

`examples/comparisons/pvi.py` ports that code:
- a mean-field Gaussian `q`;
- `λ = 0`, the published default;
- the published RMSprop option with learning rate 1e-3, run for 20000 steps;
- `s = 16` draws per step.
