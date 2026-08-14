# Eisenstein-Hu emulator example: summary statistics

Master seed 0; total runtime 8.7 min. Deterministic end-to-end (timing section excepted).

QoI: y = ln P(k*) at k* = 0.15 h/Mpc, sigma8-normalized EH98 linear matter power spectrum, z = 0, flat universe, T_CMB = 2.7255 K, no noise anywhere (eps = 0). Pool M = 40000 uniform draws over the box; test = last 8000; validation = 4096; std(y) = 0.1807, test range = 0.7548.

## Engine validation

| fiducial | s eq.(6) [Mpc] | vs quadrature | vs eq.(26) fit | k_eq vs exact | wiggle/no-wiggle |
|---|---|---|---|---|---|
| sCDM (Omega_0=1, Omega_b=0.05, h=0.5) | 137.212 | 1.7e-04 | 1.6e-02 | 1.7e-04 | 4 crossings, max 0.9% |
| box center | 150.841 | 1.7e-04 | 6.6e-03 | 1.7e-04 | 9 crossings, max 3.3% |

The closed-form sound horizon is checked against direct quadrature of s = int c_s da/(a^2 H) (same early-universe parameterization), the published eq. 26 fitting form (~2% accuracy), and k_eq against its exact square-root form; the wiggle/no-wiggle transfer ratio oscillates about 1 with percent-level amplitude in the BAO range, as required.

## Degree scan (misspecification calibration)

| degree | P | val RMSE | RMSE / std(y) |
|---|---|---|---|
| 1 | 5 | 0.02309 | 12.78% |
| 2 **(frozen)** | 20 | 0.01889 | 10.46% |
| 3 | 55 | 0.00648 | 3.58% |
| 4 | 125 | 0.00407 | 2.25% |

Frozen choice: degree 2, P = 20 (BayesianRidge on 16384 train rows, validated on the 4096-sample split; window [4%, 12%]). Higher degrees are too well-specified (below the window floor), lower too coarse. All quoted fits are safely underparametrized: N/P = 12.8, 51.2, 204.8, 819.2 for N = 256, 1024, 4096, 16384. Support/std ratio sqrt(P + 2) = 4.7. Note rank = 32 >= n_dim = 21: the low-rank update is full-rank here (rank_ = n_dim), so rank truncation is not a binding approximation in this example.

## Test coverage per N (mean +/- std [min] over 10 replicates)

| N | BayesianRidge +/-4sigma | POPS hypercube max/min* | ellipse support | ellipse+PAC ensemble |
|---|---|---|---|---|
| 256 | 0.709 +/- 0.018 [0.676] | 0.982 +/- 0.009 [0.966] | 0.999 +/- 0.001 [0.997] | 1.000 +/- 0.000 [1.000] |
| 1024 | 0.358 +/- 0.010 [0.345] | 0.999 +/- 0.001 [0.998] | 1.000 +/- 0.000 [1.000] | 1.000 +/- 0.000 [1.000] |
| 4096 | 0.159 +/- 0.004 [0.151] | 1.000 +/- 0.000 [0.999] | 1.000 +/- 0.000 [1.000] | 1.000 +/- 0.000 [1.000] |
| 16384 | 0.072 +/- 0.001 [0.070] | 1.000 +/- 0.000 [1.000] | 1.000 +/- 0.000 [1.000] | 1.000 +/- 0.000 [1.000] |

*Sampled with >= 150000 posterior draws per fit. The sampled hypercube max/min under-covers its own analytic pushforward by pure concentration of measure in moderate-to-high P; at this P the effect is mild but present, which is a genuine advantage of the ellipse's analytic pushforward (see appendix discussion).

Uncovered test points for the bare ellipse (excluded from G_test, never clipped): N = 256: mean 9.6, max 24 of 8000; N = 1024: mean 0.0, max 0 of 8000; N = 4096: mean 0.0, max 0 of 8000; N = 16384: mean 0.0, max 0 of 8000.

## Band widths

| N | ellipse / hypercube | +PAC / hypercube | ellipse full width / std(y_test) |
|---|---|---|---|
| 256 | 199.0 +/- 6.0% | 276.5 +/- 10.6% | 0.84 +/- 0.05 |
| 1024 | 176.6 +/- 5.7% | 230.2 +/- 7.1% | 0.86 +/- 0.01 |
| 4096 | 164.7 +/- 1.4% | 195.5 +/- 2.7% | 0.87 +/- 0.01 |
| 16384 | 156.2 +/- 0.9% | 171.0 +/- 1.0% | 0.87 +/- 0.00 |

Width ratios are means over the test set of pointwise band-width ratios; the last column is quoted against the data spread as a sampling-artifact-free alternative. Finding: at this moderate P the certified ellipse support is systematically WIDER than the sampled hypercube max/min range (ratio > 100%), unlike the O(50-80%) anticipated in the handoff. The interior-point condition forces the ellipse to cover every training residual, while the sampled hypercube band spans only the bulk of the pointwise corrections (its test coverage is below 1 above); the anticipated regime presupposes the strong sampling-concentration of much larger P. Reported as-is, not tuned away.

## PAC broadening of the support band

| N | mean broadening (+%) |
|---|---|
| 256 | +46.0 +/- 3.0% |
| 1024 | +36.4 +/- 3.2% |
| 4096 | +22.4 +/- 1.4% |
| 16384 | +11.3 +/- 0.3% |

Decay +46% -> +36% -> +22% -> +11% over N: the hyperposterior concentrates on the phase-1 optimum at rate N.

## Certificate vs truth (nats; mean +/- std over replicates)

| N | bound_ | G_test | objective_ (train) | gap bound_ - G_test | KL | gamma_ |
|---|---|---|---|---|---|---|
| 256 | -1.061 +/- 0.144 | -2.614 +/- 0.071 | -2.927 +/- 0.073 | 1.553 +/- 0.154 | 281.7 +/- 35.2 | 386.4 +/- 10.9 |
| 1024 | -2.286 +/- 0.037 | -2.784 +/- 0.014 | -2.835 +/- 0.020 | 0.497 +/- 0.037 | 356.5 +/- 38.4 | 404.0 +/- 7.7 |
| 4096 | -2.631 +/- 0.022 | -2.805 +/- 0.011 | -2.816 +/- 0.022 | 0.174 +/- 0.014 | 542.1 +/- 23.9 | 425.8 +/- 1.7 |
| 16384 | -2.749 +/- 0.003 | -2.807 +/- 0.003 | -2.813 +/- 0.003 | 0.058 +/- 0.002 | 829.6 +/- 5.2 | 437.0 +/- 0.1 |

ln-uniform reference (uniform density over the test data range): -0.281 nats. The certificate is non-vacuous (bound_ below the reference) at N in {256, 1024, 4096, 16384} - every quoted N.

## Appendix: estimator variants (N = 1024, single replicate)

| variant | coverage | width/std | objective_ | G_test | bound_ | KL | gamma_ |
|---|---|---|---|---|---|---|---|
| default (phase1, frozen center) | 1.000 | 1.11 | -2.825 | -2.755 | -2.268 | 364.9 | 405.6 |
| optimize_center=True | 1.000 | 0.92 | -3.342 | -3.295 | -2.246 | 889.8 | 457.8 |
| hyperprior_center='warm_start' | 1.000 | 1.07 | -2.822 | -2.749 | -2.274 | 360.8 | 394.8 |

'phase1' centers the hyperprior on the phase-1 optimum (empirical-Bayes caveat on the formal reading of bound_); 'warm_start' is the caveat-free certificate; optimize_center=True tightens the fit at the cost of a less conservative center.

## Appendix: secondary QoI y = ln[P(0.15)/P(0.05)] (N = 1024, single replicate)

BayesianRidge val RMSE 9.15% of std(y2). Coverage BR/HC/ellipse/+PAC = 0.346 / 0.999 / 1.000 / 1.000; bound_ = -2.319, G_test = -2.847, PAC broadening +35.1%. Same mechanics as the primary QoI: dimensionless BAO-envelope amplitude ratio.

## Timing

- degree 4 (P = 125), rank 32, N = 16384, pac_bayes=True: 33.3 s, n_iter_ = 837 (converged)
- degree 9 (P = 2001), rank 32, N = 16384, pac_bayes=True: 193.4 s, n_iter_ = 965 (converged)

## Acceptance checks

- [PASS] EH98 sound horizon vs quadrature (sCDM (Omega_0=1, Omega_b=0.05, h=0.5)): rel = 1.72e-04 (eq. 6 = 137.212 Mpc)
- [PASS] EH98 k_eq vs exact sqrt form (sCDM (Omega_0=1, Omega_b=0.05, h=0.5)): rel = 1.72e-04
- [PASS] EH98 sound horizon vs published eq. 26 fit (sCDM (Omega_0=1, Omega_b=0.05, h=0.5)): rel = 1.59e-02 (published accuracy ~2%)
- [PASS] EH98 T(k->0) -> 1 (sCDM (Omega_0=1, Omega_b=0.05, h=0.5)): T(1e-6/Mpc) = 1.000000
- [PASS] EH98 wiggle/no-wiggle oscillates about 1 (sCDM (Omega_0=1, Omega_b=0.05, h=0.5)): 4 sign changes, max |ratio-1| = 0.92%
- [PASS] EH98 sound horizon vs quadrature (box center): rel = 1.72e-04 (eq. 6 = 150.841 Mpc)
- [PASS] EH98 k_eq vs exact sqrt form (box center): rel = 1.72e-04
- [PASS] EH98 sound horizon vs published eq. 26 fit (box center): rel = 6.62e-03 (published accuracy ~2%)
- [PASS] EH98 T(k->0) -> 1 (box center): T(1e-6/Mpc) = 1.000000
- [PASS] EH98 wiggle/no-wiggle oscillates about 1 (box center): 9 sign changes, max |ratio-1| = 3.33%
- [PASS] degree scan lands in the misspecification window: window = [4%, 12%], deg 1: 12.8%, deg 2: 10.5%, deg 3: 3.6%, deg 4: 2.2%
- [PASS] fit converged & covering (N=256, rep=0): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (141, 141) of cap 20000
- [PASS] fit converged & covering (N=256, rep=1): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (100, 100) of cap 20000
- [PASS] fit converged & covering (N=256, rep=2): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (95, 95) of cap 20000
- [PASS] fit converged & covering (N=256, rep=3): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (128, 128) of cap 20000
- [PASS] fit converged & covering (N=256, rep=4): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (127, 127) of cap 20000
- [PASS] fit converged & covering (N=256, rep=5): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (185, 185) of cap 20000
- [PASS] fit converged & covering (N=256, rep=6): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (88, 88) of cap 20000
- [PASS] fit converged & covering (N=256, rep=7): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (166, 166) of cap 20000
- [PASS] fit converged & covering (N=256, rep=8): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (136, 136) of cap 20000
- [PASS] fit converged & covering (N=256, rep=9): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (151, 151) of cap 20000
- [PASS] fit converged & covering (N=1024, rep=0): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (72, 72) of cap 20000
- [PASS] fit converged & covering (N=1024, rep=1): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (73, 73) of cap 20000
- [PASS] fit converged & covering (N=1024, rep=2): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (88, 88) of cap 20000
- [PASS] fit converged & covering (N=1024, rep=3): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (84, 84) of cap 20000
- [PASS] fit converged & covering (N=1024, rep=4): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (94, 94) of cap 20000
- [PASS] fit converged & covering (N=1024, rep=5): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (68, 68) of cap 20000
- [PASS] fit converged & covering (N=1024, rep=6): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (83, 83) of cap 20000
- [PASS] fit converged & covering (N=1024, rep=7): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (168, 168) of cap 20000
- [PASS] fit converged & covering (N=1024, rep=8): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (89, 89) of cap 20000
- [PASS] fit converged & covering (N=1024, rep=9): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (103, 103) of cap 20000
- [PASS] fit converged & covering (N=4096, rep=0): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (112, 112) of cap 20000
- [PASS] fit converged & covering (N=4096, rep=1): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (109, 109) of cap 20000
- [PASS] fit converged & covering (N=4096, rep=2): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (113, 113) of cap 20000
- [PASS] fit converged & covering (N=4096, rep=3): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (102, 102) of cap 20000
- [PASS] fit converged & covering (N=4096, rep=4): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (77, 77) of cap 20000
- [PASS] fit converged & covering (N=4096, rep=5): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (106, 106) of cap 20000
- [PASS] fit converged & covering (N=4096, rep=6): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (103, 103) of cap 20000
- [PASS] fit converged & covering (N=4096, rep=7): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (96, 96) of cap 20000
- [PASS] fit converged & covering (N=4096, rep=8): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (107, 107) of cap 20000
- [PASS] fit converged & covering (N=4096, rep=9): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (79, 79) of cap 20000
- [PASS] fit converged & covering (N=16384, rep=0): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (91, 91) of cap 20000
- [PASS] fit converged & covering (N=16384, rep=1): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (98, 98) of cap 20000
- [PASS] fit converged & covering (N=16384, rep=2): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (101, 101) of cap 20000
- [PASS] fit converged & covering (N=16384, rep=3): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (90, 90) of cap 20000
- [PASS] fit converged & covering (N=16384, rep=4): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (93, 93) of cap 20000
- [PASS] fit converged & covering (N=16384, rep=5): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (97, 97) of cap 20000
- [PASS] fit converged & covering (N=16384, rep=6): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (108, 108) of cap 20000
- [PASS] fit converged & covering (N=16384, rep=7): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (85, 85) of cap 20000
- [PASS] fit converged & covering (N=16384, rep=8): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (91, 91) of cap 20000
- [PASS] fit converged & covering (N=16384, rep=9): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (93, 93) of cap 20000
- [PASS] PAC bounds strictly contain bare bounds at every test point: 40/40 fits
- [PASS] PAC (phase1) MAP identical to bare optimum: coef_ and U_ allclose in every fit
- [PASS] bound_ finite and monotone non-increasing for N >= 1024: mean bound_ = -1.061 -> -2.286 -> -2.631 -> -2.749
- [PASS] bound_ >= G_test for every fit: no violations
- [PASS] certificate non-vacuous (bound_ < ln-uniform reference): reference = -0.281 nats; non-vacuous at N in [256, 1024, 4096, 16384]
- [PASS] deterministic across reruns: bitwise-identical refit at N = 1024, rep = 0
- [PASS] variant converged & covering (default (phase1, frozen center)): coverage_fraction_ = 1.0000, n_iter_ = 72
- [PASS] variant converged & covering (optimize_center=True): coverage_fraction_ = 1.0000, n_iter_ = 343
- [PASS] variant converged & covering (hyperprior_center='warm_start'): coverage_fraction_ = 1.0000, n_iter_ = 75
- [PASS] fit converged & covering (secondary QoI, N=1024): coverage_fraction_ = (1.0000, 1.0000), n_iter_ = (76, 76) of cap 20000

**ALL CHECKS PASSED**
