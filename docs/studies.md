# Studies and comparison methods

The numerical studies of the paper are scripts in
[`examples/`](https://github.com/POPS-UQ/popsregression/tree/main/examples).
The comparison methods they use are deliberately **not** part of the package.
They are baselines, not POPS models, and live in `examples/comparisons/`:

| Module | Method |
|---|---|
| `bayesian_stacking.py`, `stacking_weights.py` | [Bayesian stacking](glossary.md#bayesian-stacking) of normal–inverse-gamma linear regressions (Yao et al. 2018) |
| `low_noise_objectives.py` | [PACm and PAC²-T](glossary.md#pacm-pac2-t) with a Gaussian parameter distribution or particle ensemble |
| `pvi.py` | [Predictive variational inference](glossary.md#pvi) as published (Monte Carlo log score, mean-field Gaussian) |
| `pops_dictionary.py` | The [finite POPS dictionary](glossary.md#finite-pops-dictionary-pops-dictionary-stacking) weighting ablation |
| `harness.py` | The quartic, Burgers and ACE problems, every method behind one interface, and the common metrics |
| `fluctuations.py` | Estimators of the [one-sided moment](glossary.md#one-sided-moment-cgf) and [J(Ψ)](glossary.md#j-the-jensen-moment-term) |

## Conventions shared by every study

- No reported interval, density or score includes a residual-noise term, for
  any method; see
  [parameter-only predictive](glossary.md#parameter-only-predictive-no-aleatoric-term).
- Every method sees the same design matrix
  ([shared design](glossary.md#shared-design-matrix)).
- Intervals are [exact central intervals](glossary.md#exact-central-interval-coverage)
  at 95.45% and 99.9% for every method.
- Scores:
  - the [interval score](glossary.md#interval-score);
  - the [floored NLL](glossary.md#floored-nll), finite for compact-support
    predictives;
  - the raw NLL, together with the fraction of test targets with zero density.
- Every study repeats the training draw with predeclared seeds and reports
  medians with central 90% intervals. Failed fits are counted, never dropped.

## Scripts

| Script | Output | Content |
|---|---|---|
| `example_polynomial.py` | `example_polynomial.png` | [Worked example](example.md): quartic surrogate, N = 10 and 100 |
| `example_burgers_pod.py` | `example_burgers_pod.png` | Rank-2 POD Burgers emulator, N = 8 and 80 simulator cases |
| `example_mliap.py` | `example_mliap.png` | ACE energies, calibration (P-P) curves at N/P = 1.5 and 20; `--basis subset` (default) builds the PCA basis from the training subset only |
| `example_repeated_splits.py` | `repeated_splits.png`, `generated/repeated_splits_summary.{md,tex}` | All methods, all three problems, 30 training draws per size |
| `example_low_noise_objectives.py` | `low_noise_objectives.png` | PACm, PAC²-T and PVI against the zero-noise ellipse as the likelihood width shrinks |
| `example_loss_fluctuations.py` | `loss_fluctuations.png` | Held-out estimates of the moment term of the bound |
| `example_burgers_qoi.py` | `burgers_qoi.png` | Propagation to Burgers dissipation and front steepness |
| `example_scaling.py` | `scaling.png` | Fit time and memory against P, N and rank |
| `example_pops_dictionary_ablation.py` | `pops_dictionary_ablation.png` | Supplementary weighting ablation on a finite POPS dictionary |

Run any script from `examples/` with
`uv run --extra examples python <script>`; the repeated studies accept
`--seeds` (or `--repeats`) and `--workers`, and `--plot-only` rebuilds the
figure and tables from the saved CSV.
