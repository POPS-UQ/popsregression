# Examples

Runnable example scripts for `popsregression`.

- `example_mliap.py` — compares `BayesianRidge`, the POPS hypercube, the POPS
  ellipse, and the PAC-Bayes POPS ellipse for a linear 267-feature Cu ACE
  interatomic potential. The bundled `ace_linear_uq_energies.npz` (1 MB) holds
  the energy equations only: 700 training and 300 held-out test structures.
  The design is projected onto its leading PCA modes (95% of the variance, at
  least 35) so that small observation/parameter ratios are reachable, and two
  rows compare $N/P=1.5$ with $N/P=20$.

  It writes `example_mliap.png`, a probability-probability (P-P) plot of the
  posterior-sampled CDF of the held-out energy error against the observed CDF.
  A calibrated posterior lies on the parity line, and each panel is annotated
  with the *signed miscalibration area* between the two: the signed area
  enclosed by the P-P curve and the parity line, which equals
  $P(|e_\mathrm{post}|<|e_\mathrm{obs}|)-\tfrac12$ and so is half the Gini
  coefficient of the two error samples. It is positive when the posterior is
  too narrow and negative when it is too wide. `--error-output` additionally
  writes the two error densities that the P-P plot compares. A trusted pickle
  with the same four keys can be passed with `--data`.
- `plot_pops_regression.py` — compares POPS vs `BayesianRidge` uncertainty for a
  misspecified, low-noise polynomial fit. See the rendered
  [Example](https://POPS-UQ.github.io/popsregression/example/) in the docs.
- `EllipseExample.ipynb` — compares `BayesianRidge`, `POPSRegression` and
  `POPSRegressionEllipse` at N = 10/50/500, showing the optimized ellipsoid
  bounds tracking the sampled POPS hypercube bounds, plus the closed-form
  PAC-Bayes layer. See [Ellipsoid
  posteriors](https://POPS-UQ.github.io/popsregression/ellipse/) in the docs.
- `example_burgers.py` / `example_burgers_pod.py` — small-N
  simulation-to-science example using a deterministic periodic viscous Burgers
  solver. The POD basis is learned only from smooth snapshots, so the linear
  modal-coefficient emulator is deliberately misspecified for the steep-front
  regime, and training keeps only a few random spatial observations per PDE
  run. It compares epistemic-only `BayesianRidge`, the POPS ellipse, and the
  PAC-Bayes ellipse as simulator cases are added. The POD script writes
  `example_burgers_pod_randomx_r<rank>_m<points>_s<seed>.png`; the committed
  `example_burgers_pod.png` is that figure for the default settings.
