# MERGE NOTES — `POPSRegressionEllipse` (delete at merge time)

Branch: `claude/pops-ellipse-estimator-z58bnk` (the handoff named it
`feature/pops-ellipse`; this session's designated remote branch was used
instead — content and structure are as specified). Merges cleanly onto
`main`.

## Files added

| File | Purpose |
|---|---|
| `popsregression/_projected_ball.py` | Pure kernels: `log(C_P)` via `gammaln`, smooth log continuation `L_rho` (value/grad/hess), projected-ball pdf/logpdf/variance |
| `popsregression/_ellipse.py` | `POPSRegressionEllipse` estimator: objective/gradient/Hessian-diagonal kernels (`_ellipse_nll`, `_ellipse_nll_hess_diag`) + estimator class |
| `popsregression/tests/test_ellipse_gradients.py` | FD exactness of gradients (rtol 1e-6) and Hessian diagonal (rtol 1e-5), both `L_rho` branches, with/without hyperprior ridge |
| `popsregression/tests/test_ellipse_statistics.py` | 1e6-sample MC pushforward vs closed-form density (KS vs Beta((P+1)/2,(P+1)/2), P∈{2,5,9}), `Var = s/(P+2)` to 1%, normalization to P=200, `L_rho` continuity/limits |
| `popsregression/tests/test_ellipse_estimator.py` | Behavioral, sklearn-compliance, PAC-Bayes and scaling tests (details below) |
| `docs/ellipse.md` | Math summary + API reference page (mkdocstrings) |
| `examples/EllipseExample.ipynb` | 4x3 demo: BayesianRidge / POPS Hypercube / POPS Ellipse / POPS Ellipse + PAC at N=10/50/500, + PAC-Bayes sweep cell |

## Files touched (small, required)

- `popsregression/__init__.py` — export `POPSRegressionEllipse`.
- `popsregression/utils/tests/test_discovery.py` — estimator/function counts
  (1→2 estimators, 3→8 functions; the 5 new `_projected_ball` kernels are
  discovered by `all_functions`).
- `mkdocs.yml` — nav entry + `pymdownx.arithmatex` (MathJax) for the math page.
- `docs/api.md` — cross-link to the ellipse page (class documented there only,
  to avoid duplicate-anchor warnings).
- `README.md`, `examples/README.md` — new-estimator section / listing.
- No CI config changes; no version bumps; `_pops.py` untouched.

## Public API surface

```python
POPSRegressionEllipse(
    rank=32, delta=1e-3, baseline='pops', baseline_ridge=1e-6,
    whiten_ridge=1e-8, mode_threshold=1e-8,
    rho_schedule=(1e-1, 1e-2, 1e-3, 1e-4), tol=1e-8, max_iter=500,
    fit_intercept=False, weights=None, optimize_center=False,
    random_state=None, pac_bayes=False, hyperprior_center='phase1',
    hyperprior_scale=1.0, update_hyperprior=False,
    hh_lambda_1=1e-6, hh_lambda_2=1e-6, n_outer=5, hess_floor=1e-12,
    bound_xi=0.05, subgamma_const=0.0,
)
model.fit(X, y)
model.predict(X, return_std=..., return_bounds=..., return_bound_std=...)
model.sample(n_samples, random_state=None)
```

Additions relative to the handoff contract (all keyword-only with defaults):
`baseline_ridge` (the spec's `lam_B`), `mode_threshold` (spec §1.3 "clip like
POPSRegression"), `bound_xi` and `subgamma_const` (spec §1.5(d) confidence /
sub-gamma constants). `hyperprior_scale=np.inf` is accepted and reproduces
`pac_bayes=False` bitwise (used by the tau2→inf test).

Three review-driven deviations, decided with Tom during the session:

- **`hyperprior_center='phase1'` is the default** (deviation from spec
  §1.5(a), which centers the hyperprior on the POPS warm start with zero
  low-rank block — available as `hyperprior_center='warm_start'`).
  Rationale: with the warm-start center the MAP is ridge-shrunk toward
  the baseline ellipsoid and can be *narrower* than the bare fit at
  small N — the opposite of the PAC layer's purpose. Centering on the
  phase-1 optimum makes the MAP coincide with the bare fit exactly
  (`pac_bayes=True` never changes `coef_`/`U_`), and the predictive
  bounds are strictly broader pointwise, concentrating on the bare
  values at rate N (locked in by `test_pac_bayes_never_narrower_than_
  bare`). Caveat, documented: the phase-1 center is chosen after seeing
  the data (empirical Bayes), which weakens the formal reading of
  `bound_`; `'warm_start'` keeps the spec construction and is required
  for `update_hyperprior=True` (ill-posed at `'phase1'`, warned and
  ignored there).
- **PAC bounds are the 2σ hyperposterior ensemble** (refinement of spec
  §1.6, decided with Tom): for a bare fit `return_bounds` is the fitted
  ellipse's max/min support; for a `pac_bayes=True` fit it is the
  max/min over the ensemble of ellipses within the 2σ range of the
  hyperposterior, `mean ± (sqrt(v) + 2·y_bound_std)` — strictly broader
  than the bare support by construction. `return_std` averages the
  pushforward over the hyperposterior
  (`sqrt((v + dv)/(P+2) + z^2 Sigma_c)`), and the new
  `return_bound_std` exposes the first-order delta-method std of the
  bound curves (`sqrt(z^2 Sigma_c + Var[v]/(4v))`,
  `Var[v] = 4 sum_m (z U_m)^2 (z^2 Sigma_U_m)`); the fitted ellipse's
  own support is `bounds ∓ 2·y_bound_std`.

- **`hyperprior_scale` is relative, not absolute** (deviation from spec
  §1.5(a)): the effective hyperprior variance is
  `tau2 = hyperprior_scale * max(||psi_0||^2/d, 1e-12)`. Rationale: psi
  carries the units of `y`, so a fixed absolute default (`tau2 = 1.0`)
  over-shrinks any data with large targets — on the quartic example it
  collapsed extrapolation coverage to 0.05. `tau2_` stores the effective
  absolute value (after any evidence updates, which operate on absolute
  tau2 as before).
- **`optimize_center` parameter, default False** (deviation from spec
  §1.3/1.4, which jointly optimizes `(c_t, U)`): by default the center
  is frozen at the warm start (`coef_` equals the POPS pre-fit
  coefficients exactly, i.e. the familiar BayesianRidge-style mean) and
  only the widths are optimized; the PAC-Bayes hyperposterior then
  covers the width block only (center block of `hyper_sigma_diag_` is
  zero and excluded from `kl_`/`gamma_`). Rationale: the
  jointly-optimized center satisfies a heteroscedastic-WLS stationarity
  condition (per-point precision `1/(q_i v_i)`) and deliberately
  differs from an OLS/BayesianRidge mean, which surprised review and is
  less conservative at small N; it remains available as
  `optimize_center=True` and its well-specified-limit math stays under
  test.

Fitted attributes: `coef_`, `intercept_`, `center_whitened_`, `U_`, `rank_`,
`objective_`, `coverage_fraction_`, `n_iter_`, `n_outer_iter_`, lazy
properties `ellipsoid_B_` (original/affine coordinates) and `baseline_B0_`
(whitened), and with `pac_bayes=True`: `tau2_`, `hyper_sigma_diag_`, `kl_`,
`empirical_H_`, `bound_`, `gamma_`.

## Design decisions worth reviewing

- **`weights` is a constructor parameter and `fit(X, y)` takes no
  `sample_weight`**, exactly as in the handoff contract §2. Consequence:
  sklearn's sample-weight equivalence checks are skipped by introspection
  (they key off the `fit` signature). Weights are passed through to the
  POPS baseline pre-fit so zero-weighted points also drop out of the
  baseline.
- **`random_state=None` behaves like `random_state=0`** (documented): the
  objective is nonconvex with a rotation-degenerate optimum manifold
  (`U → UQ`), so global-RNG initialization would break
  `check_fit_idempotent`-style checks and reproducibility expectations.
- **Whitening uses a symmetric `G^{-1/2}` with eigenvalue flooring**
  (`max(e, mode_threshold * e_max)`) rather than POPS's mode discarding:
  keeps the map invertible so whitened↔original round-trips are exact; the
  `whiten_ridge > 0` floor makes discarding unnecessary.
- **`baseline='pops'` never forms a dense B0**: the hypercube covariance is
  factored analytically from the fitted support/bounds
  (`S diag(var) S^T + (S mid)(S mid)^T = F F^T`, rank `n_modes + 1`), and
  `b0_i = ||F^T x_i||²` is computed in original design coordinates (exactly
  consistent with the whitened quadratic form since the flooring is shared).
- **The whitening transform itself is a dense (P, P) matrix** (mandated by
  spec §1.3 preprocessing); "no (P, P) arrays" therefore applies to the
  optimizer state and per-iteration work, which are O(N·P·r). The slow test
  bounds the tracemalloc peak at 4x the design matrix.
- `predict` computes `s(x)` via `b0(x) + ||U^T z||²`; `ellipsoid_B_`/
  `baseline_B0_`/`sample()` are the only dense O(P²) paths and are lazy.
- Phase 2 shares the phase-1 code path (`prec = 1/tau2`, 0 when
  `pac_bayes=False`), guaranteeing the `pac_bayes=False` invariance.

## Test coverage of new code (`.coveragerc`, branch coverage)

- `popsregression/_ellipse.py`: **99%** (1 missed statement: the
  `update_hyperprior` early-exit when tau2 converges before `n_outer`).
- `popsregression/_projected_ball.py`: **100%**.
- Full suite: 306 passed, 11 skipped, ~12 s (`pytest -vsl popsregression`).
- The O(N·P·r) scaling test (P=2000, r=32, N=20000, <60 s, tracemalloc
  bound) is opt-in via `POPS_RUN_SLOW=1` to keep the default suite fast;
  it passes in ~15 s locally.

## Known limitations

- **P=1 has no barrier** (`k = (P-1)/2 = 0`): the pushforward of a 1-ball is
  uniform, whose NLL has a support constraint rather than a log-barrier, so
  a single-feature fit collapses widths to the `delta` floor. Harmless for
  the sklearn checks; not a practical use case.
- With `baseline='zero'`/`'ridge'` and large-scale targets, the spec's fixed
  `1e-3` random `U` init starts deep in the continued branch and needs more
  L-BFGS iterations than `baseline='pops'` (which starts near the POPS
  scale). The default baseline avoids this.
- The Laplace covariance is diagonal by construction (spec §1.5(c)); the
  Hessian floor warning (`hess_floor`) can trigger at barrier-active optima
  where the exact Hessian is not PSD — expected, documented.
- `subgamma_const` defaults to 0: the bound is stated under the
  near-deterministic bounded-loss idealization unless the user supplies the
  sub-gamma constant (documented in the docstring).
- The evidence update's tau2 convergence shortcut compares successive tau2
  values against `tol`; the final reported `tau2_` may lag the last
  optimization by less than `tol`.
- At very small N the bare ellipse is deliberately tighter than the POPS
  hypercube min/max bounds (minimum covering support vs un-optimized box
  support). With the default `hyperprior_center='phase1'` the PAC layer
  strictly broadens the predictive std and adds a strictly positive
  bound spread (never narrower). The conservative low-N configuration —
  the PAC layer's main motivation — is simply `pac_bayes=True` on the
  (frozen-center) defaults, whose bounds are the 2σ hyperposterior
  ensemble: on a 10-seed N=10 sweep of the misspecified example they
  cover 0.99 (mean) / 0.92 (min) of the dense truth at ~70% of the
  hypercube width, vs 0.79 / 0.63 for the hypercube. Locked in by
  `test_low_n_conservatism_recipe` and shown as the fourth row of the
  notebook's demo figure.

## Suggested squash points

The branch is structured for review commit-by-commit; if squashing:

1. `Add projected-ball pushforward kernels` (standalone).
2. `Add POPSRegressionEllipse estimator` (module + export + discovery counts).
3. The three test commits (`...gradients...`, `...pushforward...`,
   `...estimator tests` + `Extend baseline...`) → one "Add ellipse tests".
4. Docs + notebook commits → one "Add ellipse docs and example".

`MERGE_NOTES.md` (this file) is to be deleted at merge time.
