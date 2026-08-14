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
- `darcy_flow.py` — high-dimensional surrogate demonstration for the
  Sim2Science paper: a self-contained 2D Darcy-flow finite-difference engine
  with log-Gaussian coefficient fields, a misspecified random-Fourier-feature
  surrogate (P = 512) of the boundary-flux QoI, and the full
  `BayesianRidge` / POPS hypercube / `POPSRegressionEllipse` (+ PAC-Bayes)
  comparison over N = 64 … 4096. Deterministic end-to-end; regenerates
  `darcy_flow.pdf/.png` and `darcy_flow_summary.txt/.json`. Run with
  `--quick` for a fast smoke test.
