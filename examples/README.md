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
- `burgers_sim2science.py` — small-N simulation-to-science example using a
  deterministic periodic viscous Burgers solver. A deliberately restricted
  fundamental-mode emulator (P=13) omits the higher spatial harmonics produced
  by nonlinear front steepening. It compares epistemic-only `BayesianRidge`,
  the POPS ellipse, and the PAC-Bayes ellipse for 6/10/16/24/40 simulator cases,
  and reports held-out coverage, PAC broadening, held-out projected-ball NLL,
  the uniform-density reference, PAC bound and certificate gap. It writes
  `burgers_sim2science.png`.
