# Headline table (medians over training draws)

Medians over training draws at the smallest and largest training size (u/P: independent units per parameter). cov.: coverage of the 95.45% central interval; IS: interval score of that interval; NLL$_b$: floored negative log predictive density (b = 0.01); A$_{abs}$: unsigned calibration area of the pooled absolute-error P-P curve. '*': at least one fit did not report optimizer convergence (kept). Abbreviations: POPS = pointwise optimal parameter sets; EB = empirical Bayes; PAC = probably approximately correct (PAC-Bayes hyperparameter bound); PACm = PAC-Bayes objective with an m-sample predictive (Morningstar et al. 2022); PAC$^2_T$ = second-order PAC-Bayes objective with the tandem Taylor weight (Masegosa 2020); PVI = predictive variational inference (Lai, Linero and Yao); IS = interval score (Gneiting and Raftery), of the 95.45% central interval; NLL = negative log predictive density.


## quartic

| method | cov. (2 u/P) | IS (2 u/P) | NLL$_b$ (2 u/P) | A$_{abs}$ (2 u/P) | cov. (60 u/P) | IS (60 u/P) | NLL$_b$ (60 u/P) | A$_{abs}$ (60 u/P) |
|---|---|---|---|---|---|---|---|---|
| Bayesian ridge | 0.19 | 983 | 8.76 | 0.357 | 0.07 | 1.06e+03 | 9.53 | 0.439 |
| POPS hypercube | 0.64 | 564 | -- | 0.128 | 0.81 | 132 | -- | 0.074 |
| POPS Ellipse | 0.93 | 247 | 5.16 | 0.084 | 1.00 | 107 | 4.54 | 0.082 |
| POPS Ellipse+EB | 1.00 | 270 | 5.04 | 0.074 | 1.00 | 116 | 4.56 | 0.073 |
| POPS Ellipse+PAC | 1.00 | 1.58e+03 | 5.30 | 0.115 | 1.00 | 111 | 4.56 | 0.069 |
| Bayesian stacking | 0.67 | 431 | 5.74 | 0.170 | 0.36 | 743 | 7.81 | 0.331 |
| PVI | 0.77 | 713 | 5.40 | 0.123 | 1.00 | 134 | 4.87 | 0.084 |
| PACm | 0.62 | 964 | 6.07 | 0.172 | 1.00 | 153 | 4.88 | 0.056 |
| PAC$^2_T$ | 0.01 | 1.98e+03 | 10.14 | 0.489 | 0.00 | 1.28e+03 | 10.32 | 0.499 |

## burgers

| method | cov. (1 u/P) | IS (1 u/P) | NLL$_b$ (1 u/P) | A$_{abs}$ (1 u/P) | cov. (10 u/P) | IS (10 u/P) | NLL$_b$ (10 u/P) | A$_{abs}$ (10 u/P) |
|---|---|---|---|---|---|---|---|---|
| Bayesian ridge | 0.70 | 0.859 | -1.09 | 0.135 | 0.40 | 1.03 | 0.67 | 0.290 |
| POPS hypercube | 0.50 | 0.926 | -- | 0.159 | 0.83 | 0.432 | -- | 0.175 |
| POPS Ellipse | 0.76 | 0.645 | -1.00 | 0.109 | 0.97 | 0.309 | -1.95 | 0.051 |
| POPS Ellipse+EB | 0.92 | 0.517 | -1.47 | 0.054 | 0.99 | 0.32 | -1.90 | 0.074 |
| POPS Ellipse+PAC | 0.96 | 0.956 | -1.18 | 0.101 | 0.99 | 0.338 | -1.78 | 0.097 |
| Bayesian stacking | 0.89 | 0.651 | -1.30 | 0.069 | 0.87 | 0.689 | -0.70 | 0.080 |
| PVI | 0.74 | 0.881 | -0.89 | 0.118 | 0.97 | 0.465 | -1.60 | 0.063 |
| PACm | 0.86 | 0.588 | -1.70 | 0.045 | 0.99 | 0.377 | -1.89 | 0.042 |
| PAC$^2_T$ | 0.05 | 1.92 | 5.16 | 0.454 | 0.03 | 1.58 | 5.55 | 0.473 |

## ace

| method | cov. (1.4 u/P) | IS (1.4 u/P) | NLL$_b$ (1.4 u/P) | A$_{abs}$ (1.4 u/P) | cov. (19 u/P) | IS (19 u/P) | NLL$_b$ (19 u/P) | A$_{abs}$ (19 u/P) |
|---|---|---|---|---|---|---|---|---|
| Bayesian ridge | 0.82 | 1.07 | -1.09 | 0.068 | 0.42 | 0.636 | 0.20 | 0.301 |
| POPS hypercube | 0.35 | 2.54 | -- | 0.265 | 0.80 | 0.312 | -- | 0.025 |
| POPS Ellipse | 0.84 | 0.904 | -1.11 | 0.092 | 0.97 | 0.128 | -2.57 | 0.045 |
| POPS Ellipse+EB | 1.00 | 0.783 | -1.01 | 0.111 | 0.99 | 0.146 | -2.41 | 0.055 |
| POPS Ellipse+PAC | 0.98 | 1.25 | -0.86 | 0.063 | 0.99 | 0.144 | -2.63 | 0.018 |
| Bayesian stacking | 1.00 | 0.446 | -1.51 | 0.103 | 0.83 | 0.355 | -1.34 | 0.023 |
| PVI | 0.96 | 1.13 | -0.23 | 0.133 | 0.99 | 0.311 | -1.47 | 0.172 |
| PACm | 1.00* | 2.18* | 0.12* | 0.398* | 1.00* | 0.52* | -1.23* | 0.307* |
| PAC$^2_T$ | 0.69* | 0.485* | -1.87* | 0.124* | 0.12* | 0.921* | 3.26* | 0.442* |
