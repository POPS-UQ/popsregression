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
  demonstration: a joint (θ, k) tensor-polynomial surrogate for the EH98
  BAO wiggle ratio ln[P/P_nw] (transfer functions implemented directly, no
  CAMB/CLASS), certified by `POPSRegressionEllipse` with the PAC-Bayes
  layer. Deterministic from a master seed; writes the paper figure
  (`eh_emulator.png`/`.pdf`, plot code in `eh_emulator_figure.py`, figure
  inputs cached in `eh_emulator_figdata.npz` for `--replot`) and the quoted
  summary statistics (`eh_emulator_summary.md`; the superseded scalar-QoI
  results are archived in `eh_emulator_summary_kstar.md`). Run with
  `--quick` for a smoke test.
- `demo_workshop.py` — the Sim2Science paper's first figure
  (`demo_workshop.png`/`.pdf`): the quartic-vs-oscillatory benchmark
  (P = 5) fit by Bayesian Ridge, POPS hypercube, POPS ellipse and
  POPS ellipse + PAC at N = 10 and N = 500, deterministic from the
  master seed.
