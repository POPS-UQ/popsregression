# Examples

Runnable example scripts for `popsregression`.

- `plot_pops_regression.py` — compares POPS vs `BayesianRidge` uncertainty for a
  misspecified, low-noise polynomial fit. See the rendered
  [Example](https://POPS-UQ.github.io/popsregression/example/) in the docs.
- `EllipseExample.ipynb` — compares `BayesianRidge`, `POPSRegression` and
  `POPSRegressionEllipse` at N = 10/50/500, showing the optimized ellipsoid
  bounds tracking the sampled POPS hypercube bounds, plus the closed-form
  PAC-Bayes layer. See [Ellipsoid
  posteriors](https://POPS-UQ.github.io/popsregression/ellipse/) in the docs.
- `eh_emulator.py` — the Sim2Science paper's Eisenstein–Hu emulator
  demonstration: a polynomial surrogate for ln P(k*) of the EH98 linear
  matter power spectrum (implemented directly, no CAMB/CLASS), certified by
  `POPSRegressionEllipse` with the PAC-Bayes layer. Deterministic from a
  master seed; writes the paper figure (`eh_emulator.png`/`.pdf`, plot code
  in `eh_emulator_figure.py`) and the quoted summary statistics
  (`eh_emulator_summary.md`). Run with `--quick` for a smoke test.
- `demo_workshop.py` — the Sim2Science paper's first figure
  (`demo_workshop.png`/`.pdf`): the quartic-vs-oscillatory benchmark
  (P = 5) fit by Bayesian Ridge, POPS hypercube, POPS ellipse and
  POPS ellipse + PAC at N = 10 and N = 500, deterministic from the
  master seed.
