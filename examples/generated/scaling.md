# Runtime and memory

Single fits, default BLAS threading; `peak/design` is the tracemalloc peak divided by the bytes of the design matrix.


## ACE full design

| method | N | P | rank | seconds | peak/design |
|---|---|---|---|---|---|
| Bayesian ridge | 700 | 267 | 32 | 0.0147 | 3.17 |
| POPS hypercube | 700 | 267 | 32 | 0.016 | 7.18 |
| POPS ellipse | 700 | 267 | 32 | 0.861 | 22.77 |
| Ellipse+EB | 700 | 267 | 32 | 0.873 | 52.99 |
| Ellipse+PAC | 700 | 267 | 32 | 2 | 120.27 |

## P sweep (N = 4P, r = 32)

| method | N | P | rank | seconds | peak/design |
|---|---|---|---|---|---|
| Bayesian ridge | 200 | 50 | 32 | 0.000856 | 2.89 |
| POPS hypercube | 200 | 50 | 32 | 0.00124 | 7.76 |
| POPS ellipse | 200 | 50 | 32 | 0.0474 | 16.59 |
| Ellipse+EB | 200 | 50 | 32 | 0.0509 | 86.90 |
| Ellipse+PAC | 200 | 50 | 32 | 0.166 | 293.99 |
| Bayesian ridge | 400 | 100 | 32 | 0.00253 | 2.80 |
| POPS hypercube | 400 | 100 | 32 | 0.00367 | 7.03 |
| POPS ellipse | 400 | 100 | 32 | 0.125 | 10.11 |
| Ellipse+EB | 400 | 100 | 32 | 0.131 | 43.62 |
| Ellipse+PAC | 400 | 100 | 32 | 0.272 | 148.48 |
| Bayesian ridge | 1000 | 250 | 32 | 0.011 | 2.77 |
| POPS hypercube | 1000 | 250 | 32 | 0.0175 | 6.80 |
| POPS ellipse | 1000 | 250 | 32 | 0.681 | 9.82 |
| Ellipse+EB | 1000 | 250 | 32 | 0.7 | 17.82 |
| Ellipse+PAC | 1000 | 250 | 32 | 0.658 | 61.42 |
| Bayesian ridge | 2000 | 500 | 32 | 0.044 | 2.76 |
| POPS hypercube | 2000 | 500 | 32 | 0.0739 | 6.76 |
| POPS ellipse | 2000 | 500 | 32 | 2.1 | 9.77 |
| Ellipse+EB | 2000 | 500 | 32 | 2.27 | 9.77 |
| Ellipse+PAC | 2000 | 500 | 32 | 1.5 | 32.45 |
| Bayesian ridge | 4000 | 1000 | 32 | 0.218 | 2.75 |
| POPS hypercube | 4000 | 1000 | 32 | 0.374 | 6.75 |
| POPS ellipse | 4000 | 1000 | 32 | 7.33 | 9.76 |
| Ellipse+EB | 4000 | 1000 | 32 | 7.46 | 9.76 |
| Ellipse+PAC | 4000 | 1000 | 32 | 4.72 | 17.97 |
| Bayesian ridge | 8000 | 2000 | 32 | 1.61 | nan |
| POPS hypercube | 8000 | 2000 | 32 | 2.86 | nan |
| POPS ellipse | 8000 | 2000 | 32 | 26.8 | nan |
| Ellipse+EB | 8000 | 2000 | 32 | 27.2 | nan |
| Ellipse+PAC | 8000 | 2000 | 32 | 19.8 | nan |

## N sweep (P = 250, r = 32)

| method | N | P | rank | seconds | peak/design |
|---|---|---|---|---|---|
| Bayesian ridge | 500 | 250 | 32 | 0.0077 | nan |
| POPS hypercube | 500 | 250 | 32 | 0.0127 | nan |
| POPS ellipse | 500 | 250 | 32 | 0.371 | nan |
| Ellipse+EB | 500 | 250 | 32 | 0.39 | nan |
| Ellipse+PAC | 500 | 250 | 32 | 0.434 | nan |
| Bayesian ridge | 1000 | 250 | 32 | 0.011 | nan |
| POPS hypercube | 1000 | 250 | 32 | 0.0181 | nan |
| POPS ellipse | 1000 | 250 | 32 | 0.692 | nan |
| Ellipse+EB | 1000 | 250 | 32 | 0.704 | nan |
| Ellipse+PAC | 1000 | 250 | 32 | 0.67 | nan |
| Bayesian ridge | 2000 | 250 | 32 | 0.0178 | nan |
| POPS hypercube | 2000 | 250 | 32 | 0.0281 | nan |
| POPS ellipse | 2000 | 250 | 32 | 0.91 | nan |
| Ellipse+EB | 2000 | 250 | 32 | 0.924 | nan |
| Ellipse+PAC | 2000 | 250 | 32 | 0.964 | nan |
| Bayesian ridge | 4000 | 250 | 32 | 0.028 | nan |
| POPS hypercube | 4000 | 250 | 32 | 0.0434 | nan |
| POPS ellipse | 4000 | 250 | 32 | 1.69 | nan |
| Ellipse+EB | 4000 | 250 | 32 | 1.69 | nan |
| Ellipse+PAC | 4000 | 250 | 32 | 1.59 | nan |
| Bayesian ridge | 8000 | 250 | 32 | 0.0542 | nan |
| POPS hypercube | 8000 | 250 | 32 | 0.0854 | nan |
| POPS ellipse | 8000 | 250 | 32 | 3.32 | nan |
| Ellipse+EB | 8000 | 250 | 32 | 3.34 | nan |
| Ellipse+PAC | 8000 | 250 | 32 | 3 | nan |

## rank sweep (N = 1000, P = 250)

| method | N | P | rank | seconds | peak/design |
|---|---|---|---|---|---|
| POPS ellipse | 1000 | 250 | 4 | 0.304 | nan |
| Ellipse+EB | 1000 | 250 | 4 | 0.307 | nan |
| Ellipse+PAC | 1000 | 250 | 4 | 0.62 | nan |
| POPS ellipse | 1000 | 250 | 8 | 0.377 | nan |
| Ellipse+EB | 1000 | 250 | 8 | 0.382 | nan |
| Ellipse+PAC | 1000 | 250 | 8 | 0.628 | nan |
| POPS ellipse | 1000 | 250 | 16 | 0.541 | nan |
| Ellipse+EB | 1000 | 250 | 16 | 0.552 | nan |
| Ellipse+PAC | 1000 | 250 | 16 | 0.643 | nan |
| POPS ellipse | 1000 | 250 | 32 | 0.685 | nan |
| Ellipse+EB | 1000 | 250 | 32 | 0.704 | nan |
| Ellipse+PAC | 1000 | 250 | 32 | 0.672 | nan |
| POPS ellipse | 1000 | 250 | 64 | 1.22 | nan |
| Ellipse+EB | 1000 | 250 | 64 | 1.26 | nan |
| Ellipse+PAC | 1000 | 250 | 64 | 0.729 | nan |
| POPS ellipse | 1000 | 250 | 128 | 2.03 | nan |
| Ellipse+EB | 1000 | 250 | 128 | 2.1 | nan |
| Ellipse+PAC | 1000 | 250 | 128 | 0.839 | nan |
