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
| `examples/EllipseExample.ipynb` | 3x3 figure: BayesianRidge / POPS Hypercube / POPS Ellipse at N=10/50/500 + PAC-Bayes demo cell |

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
    fit_intercept=False, weights=None, random_state=None,
    pac_bayes=False, hyperprior_scale=1.0, update_hyperprior=False,
    hh_lambda_1=1e-6, hh_lambda_2=1e-6, n_outer=5, hess_floor=1e-12,
    bound_xi=0.05, subgamma_const=0.0,
)
model.fit(X, y)
model.predict(X, return_std=..., return_bounds=...)
model.sample(n_samples, random_state=None)
```

Additions relative to the handoff contract (all keyword-only with defaults):
`baseline_ridge` (the spec's `lam_B`), `mode_threshold` (spec §1.3 "clip like
POPSRegression"), `bound_xi` and `subgamma_const` (spec §1.5(d) confidence /
sub-gamma constants). `hyperprior_scale=np.inf` is accepted and reproduces
`pac_bayes=False` bitwise (used by the tau2→inf test).

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

## Suggested squash points

The branch is structured for review commit-by-commit; if squashing:

1. `Add projected-ball pushforward kernels` (standalone).
2. `Add POPSRegressionEllipse estimator` (module + export + discovery counts).
3. The three test commits (`...gradients...`, `...pushforward...`,
   `...estimator tests` + `Extend baseline...`) → one "Add ellipse tests".
4. Docs + notebook commits → one "Add ellipse docs and example".

`MERGE_NOTES.md` (this file) is to be deleted at merge time.
