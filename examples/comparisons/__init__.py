"""Comparison methods and evaluation helpers used by the paper examples.

These are deliberately kept outside the :mod:`popsregression` package: they
are baselines and ablations for the numerical study, not POPS models.

- :mod:`.bayesian_stacking` — stacking of normal-inverse-gamma Bayesian linear
  regressions (Yao et al. 2018), with the weight solver in
  :mod:`.stacking_weights`.
- :mod:`.low_noise_objectives` — PACm (Morningstar et al. 2022) and PAC^2_T
  (Masegosa 2020) with a Gaussian parameter distribution.
- :mod:`.pvi` — predictive variational inference (Lai, Linero and Yao).
- :mod:`.pops_dictionary` — the pilot-split finite POPS dictionary and its
  weighting ablation (Gibbs versus POPS-dictionary stacking).
- :mod:`.metrics` — common parameter-only predictive metrics.
- :mod:`.fluctuations` — loss-fluctuation diagnostics for the PAC moment
  condition.
"""
