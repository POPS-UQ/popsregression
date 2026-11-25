# Examples

Scripts that produce the figures and tables of the accompanying paper, and the
comparison methods they use. Plotting and tables need the optional `examples`
extra; with [uv](https://docs.astral.sh/uv/), run from this directory:

```bash
uv run --extra examples python example_polynomial.py
```

Terms such as *pilot split*, *floored NLL* or *interval score* are defined in
the [glossary](../docs/glossary.md).

## Comparison methods (`comparisons/`)

These are baselines and ablations for the paper, not part of the
`popsregression` package:

- `bayesian_stacking.py`, `stacking_weights.py`: Bayesian stacking of
  normal–inverse-gamma linear regressions on predeclared feature subsets
  (Yao et al. 2018).
- `low_noise_objectives.py`: PACm (Morningstar et al. 2022) and PAC²-T
  (Masegosa 2020).
- `pvi.py`: predictive variational inference as published (Lai, Linero and
  Yao, ICML 2026; a port of github.com/lll6924/pvi). Its Monte Carlo log
  score is the PACm objective.
- `pops_dictionary.py`: the finite POPS dictionary and its weighting ablation
  (Gibbs versus POPS-dictionary stacking).
- `harness.py`: the quartic, Burgers and ACE problems, every method behind
  one interface, and the common parameter-only metrics.
- `fluctuations.py`: estimators of the moment term of the PAC-Bayes bound.
- `plotting.py`: the shared band style.

Their tests run with `uv run --group test pytest comparisons`.

## Conventions

- No interval, density or score includes a residual-noise term, for any
  method.
- Every method is fitted on the same design matrix. The comparison methods
  rescale but never center, so none gains an implicit intercept. The ACE
  design has an explicit constant column (P = 36).
- Every method reports the same two exact central intervals, 95.45% and
  99.9%.
- Repeated studies use predeclared seeds and report medians with central 90%
  intervals. Failed fits are counted.
- Declared output intervals come from domain knowledge:
  - quartic: `[-160, 160]`, since the target lies in `[-144.4, 136.9]` on
    `[-10, 10]`;
  - Burgers: `[-2.6, 2.6]`, from the maximum principle;
  - ACE: `[-4.2, -2.2]` eV/atom, an assumption stated in `harness.py`.

## Scripts

| Script | Outputs | Content |
|---|---|---|
| `example_polynomial.py` | `example_polynomial.png` | Quartic surrogate of an oscillatory function (P = 5), N = 10 and 100; Bayesian ridge, POPS hypercube, POPS ellipse, Ellipse+EB, Ellipse+PAC |
| `example_burgers_pod.py` | `example_burgers_pod.png` | Rank-2 POD Burgers emulator (P = 8), N = 8 and 80 simulator cases, same five methods |
| `example_mliap.py` | `example_mliap.png` | Linear ACE potential for Cu, energies only (267 features projected on 35 PCA modes plus a constant column), calibration curves at N/P = 1.5 and 20. `--basis subset` (default) builds the PCA basis from each training subset only; `--basis pool` uses all 700 training descriptors |
| `example_repeated_splits.py` | `repeated_splits.png`, `generated/repeated_splits.csv`, `generated/repeated_splits_summary.{md,tex}` | All nine methods on all three problems, 30 training draws per size. Supplementary variants: PVI with its KL regularizer at unit weight, and free-center ellipse variants |
| `example_low_noise_objectives.py` | `low_noise_objectives.png`, `generated/low_noise_objectives.{csv,md}` | PACm, PAC²-T and PVI as the likelihood width shrinks, against the zero-noise ellipse family and Bayesian stacking, N = 10 and 50, 20 draws |
| `example_loss_fluctuations.py` | `loss_fluctuations.png`, `generated/loss_fluctuations.{csv,md}` | Held-out estimates of the one-sided moment of the bound: CGF curves, `J(Ψ)`, generalization gap, 20 draws |
| `example_burgers_qoi.py` | `burgers_qoi.png`, `generated/burgers_qoi.{csv,md}` | Propagation to Burgers dissipation and front steepness, 20 draws |
| `example_scaling.py` | `scaling.png`, `generated/scaling.{csv,md}` | Fit time and peak memory against P (to 2000), N and rank; full 267-feature ACE design |
| `example_pops_dictionary_ablation.py` | `pops_dictionary_ablation.png`, `generated/pops_dictionary_ablation.{csv,md}` | Supplementary ablation: weight rules on a finite POPS dictionary |

The repeated studies accept `--seeds` (or `--repeats`) and `--workers`;
`--plot-only` rebuilds figures and tables from the saved CSV.
`example_burgers.py` is the shared Burgers solver. The ACE data
(`ace_linear_uq_energies.npz`, 1 MB) holds the energy equations of a linear
267-feature Cu ACE potential: 700 training and 300 test structures.
