Study `fixed_pilot`: medians over repeats; nats unless stated. `empirical` is the PAC term sum_k q_k g_k, `mixture` the mixture log loss on D1, `Jensen` their difference, `KL`/`complexity`/`Hoeffding` the bound terms at the selected temperature, `raw` the complete PAC right side, `ceiling` the trivial bound log(R_y/beta), `non-vac.` the fraction of repeats with raw < ceiling, `test NLL` the mixture log loss on an independent test sample (with its standard error), `floor` the mean fraction of predictive density supplied by the uniform floor, and `cov`/`width` the descriptive coverage and mean width of the exact 95.45% central interval on the test sample.

| N1 | rule | empirical | mixture | Jensen | KL | complexity | Hoeffding | raw | ceiling | non-vac. | test NLL | floor | cov | width |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | fixed prior $q_0$ | 5.12 | 4.95 | 0.225 | 0.00 | 2.82 | 3.05 | **10.99** | 9.95 | 0% | 4.879 ± 0.022 | 0.008 | 1.000 | 292 |
| 10 | single best candidate | 4.62 | 4.62 | 0.000 | 2.83 | 3.14 | 4.00 | **11.76** | 9.95 | 0% | 4.594 ± 0.021 | 0.006 | 1.000 | 117 |
| 10 | predictive stacking | 4.62 | 4.62 | 0.000 | 2.71 | 3.47 | 3.53 | **11.68** | 9.95 | 0% | 4.599 ± 0.022 | 0.006 | 1.000 | 117 |
| 10 | PAC Gibbs | 4.84 | 4.77 | 0.066 | 0.26 | 2.94 | 3.05 | **10.83** | 9.95 | 0% | 4.740 ± 0.021 | 0.007 | 1.000 | 216 |
| 10 | Ellipse+EB (one sample, N0+N1) | – | – | – | – | – | – | – (diagnostic objective, not a PAC bound) | – | – | – | – | 1.000 | 134 |
| 10 | bare ellipse (one sample, N0+N1) | – | – | – | – | – | – | – | – | – | 4.589 (contaminated), 4.575 (unmodified, 0.0% non-finite) | – | 1.000 | 113 |
| 30 | fixed prior $q_0$ | 5.05 | 4.85 | 0.206 | 0.00 | 1.64 | 1.75 | **8.43** | 9.95 | 100% | 4.879 ± 0.022 | 0.008 | 1.000 | 292 |
| 30 | single best candidate | 4.56 | 4.56 | 0.000 | 2.83 | 1.82 | 2.30 | **8.68** | 9.95 | 100% | 4.591 ± 0.022 | 0.006 | 1.000 | 116 |
| 30 | predictive stacking | 4.56 | 4.56 | 0.000 | 2.83 | 1.82 | 2.30 | **8.62** | 9.95 | 100% | 4.591 ± 0.022 | 0.006 | 1.000 | 115 |
| 30 | PAC Gibbs | 4.70 | 4.65 | 0.044 | 0.49 | 1.77 | 1.75 | **8.22** | 9.95 | 100% | 4.686 ± 0.021 | 0.007 | 1.000 | 172 |
| 30 | Ellipse+EB (one sample, N0+N1) | – | – | – | – | – | – | – (diagnostic objective, not a PAC bound) | – | – | – | – | 1.000 | 128 |
| 30 | bare ellipse (one sample, N0+N1) | – | – | – | – | – | – | – | – | – | 4.576 (contaminated), 4.562 (unmodified, 0.0% non-finite) | – | 1.000 | 109 |
| 100 | fixed prior $q_0$ | 5.09 | 4.88 | 0.209 | 0.00 | 0.95 | 0.90 | **6.95** | 9.95 | 100% | 4.879 ± 0.022 | 0.008 | 1.000 | 292 |
| 100 | single best candidate | 4.59 | 4.59 | 0.000 | 2.83 | 1.06 | 1.18 | **6.84** | 9.95 | 100% | 4.587 ± 0.021 | 0.006 | 1.000 | 117 |
| 100 | predictive stacking | 4.59 | 4.59 | 0.000 | 2.83 | 1.06 | 1.18 | **6.84** | 9.95 | 100% | 4.587 ± 0.021 | 0.006 | 1.000 | 117 |
| 100 | PAC Gibbs | 4.67 | 4.65 | 0.024 | 0.79 | 1.07 | 0.90 | **6.65** | 9.95 | 100% | 4.644 ± 0.021 | 0.007 | 1.000 | 147 |
| 100 | Ellipse+EB (one sample, N0+N1) | – | – | – | – | – | – | – (diagnostic objective, not a PAC bound) | – | – | – | – | 1.000 | 122 |
| 100 | bare ellipse (one sample, N0+N1) | – | – | – | – | – | – | – | – | – | 4.565 (contaminated), 4.551 (unmodified, 0.0% non-finite) | – | 0.999 | 105 |
| 300 | fixed prior $q_0$ | 5.08 | 4.87 | 0.208 | 0.00 | 0.55 | 0.52 | **6.15** | 9.95 | 100% | 4.879 ± 0.022 | 0.008 | 1.000 | 292 |
| 300 | single best candidate | 4.57 | 4.57 | 0.000 | 2.83 | 0.62 | 0.68 | **5.87** | 9.95 | 100% | 4.583 ± 0.022 | 0.006 | 1.000 | 117 |
| 300 | predictive stacking | 4.57 | 4.57 | 0.000 | 2.83 | 0.62 | 0.68 | **5.87** | 9.95 | 100% | 4.583 ± 0.022 | 0.006 | 1.000 | 117 |
| 300 | PAC Gibbs | 4.62 | 4.61 | 0.013 | 1.08 | 0.65 | 0.52 | **5.79** | 9.95 | 100% | 4.616 ± 0.021 | 0.006 | 1.000 | 132 |
| 300 | Ellipse+EB (one sample, N0+N1) | – | – | – | – | – | – | – (diagnostic objective, not a PAC bound) | – | – | – | – | 1.000 | 118 |
| 300 | bare ellipse (one sample, N0+N1) | – | – | – | – | – | – | – | – | – | 4.569 (contaminated), 4.555 (unmodified, 0.0% non-finite) | – | 1.000 | 107 |

Study `repeated_pilot`: medians over repeats; nats unless stated. `empirical` is the PAC term sum_k q_k g_k, `mixture` the mixture log loss on D1, `Jensen` their difference, `KL`/`complexity`/`Hoeffding` the bound terms at the selected temperature, `raw` the complete PAC right side, `ceiling` the trivial bound log(R_y/beta), `non-vac.` the fraction of repeats with raw < ceiling, `test NLL` the mixture log loss on an independent test sample (with its standard error), `floor` the mean fraction of predictive density supplied by the uniform floor, and `cov`/`width` the descriptive coverage and mean width of the exact 95.45% central interval on the test sample.

| N1 | rule | empirical | mixture | Jensen | KL | complexity | Hoeffding | raw | ceiling | non-vac. | test NLL | floor | cov | width |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | fixed prior $q_0$ | 5.29 | 5.02 | 0.233 | 0.00 | 2.82 | 3.05 | **11.16** | 9.95 | 0% | 4.949 ± 0.023 | 0.009 | 1.000 | 344 |
| 10 | single best candidate | 4.69 | 4.69 | 0.000 | 2.83 | 3.14 | 4.00 | **11.83** | 9.95 | 0% | 4.692 ± 0.023 | 0.007 | 1.000 | 151 |
| 10 | predictive stacking | 4.73 | 4.69 | 0.000 | 2.73 | 3.14 | 4.00 | **11.76** | 9.95 | 0% | 4.703 ± 0.023 | 0.007 | 1.000 | 150 |
| 10 | PAC Gibbs | 4.90 | 4.84 | 0.071 | 0.27 | 2.94 | 3.05 | **10.90** | 9.95 | 0% | 4.822 ± 0.023 | 0.008 | 1.000 | 268 |
| 10 | Ellipse+EB (one sample, N0+N1) | – | – | – | – | – | – | – (diagnostic objective, not a PAC bound) | – | – | – | – | 1.000 | 140 |
| 10 | bare ellipse (one sample, N0+N1) | – | – | – | – | – | – | – | – | – | 4.643 (contaminated), 4.629 (unmodified, 0.4% non-finite) | – | 1.000 | 126 |
| 30 | fixed prior $q_0$ | 5.16 | 4.93 | 0.255 | 0.00 | 1.64 | 1.75 | **8.55** | 9.95 | 100% | 4.949 ± 0.023 | 0.010 | 1.000 | 344 |
| 30 | single best candidate | 4.70 | 4.70 | 0.000 | 2.83 | 1.82 | 2.30 | **8.81** | 9.95 | 100% | 4.681 ± 0.023 | 0.007 | 1.000 | 142 |
| 30 | predictive stacking | 4.70 | 4.69 | 0.000 | 2.83 | 1.82 | 2.30 | **8.75** | 9.95 | 100% | 4.686 ± 0.023 | 0.007 | 1.000 | 141 |
| 30 | PAC Gibbs | 4.84 | 4.77 | 0.046 | 0.50 | 1.77 | 1.75 | **8.35** | 9.95 | 100% | 4.768 ± 0.023 | 0.008 | 1.000 | 215 |
| 30 | Ellipse+EB (one sample, N0+N1) | – | – | – | – | – | – | – (diagnostic objective, not a PAC bound) | – | – | – | – | 1.000 | 133 |
| 30 | bare ellipse (one sample, N0+N1) | – | – | – | – | – | – | – | – | – | 4.612 (contaminated), 4.599 (unmodified, 0.0% non-finite) | – | 1.000 | 117 |
| 100 | fixed prior $q_0$ | 5.20 | 4.95 | 0.238 | 0.00 | 0.95 | 0.90 | **7.05** | 9.95 | 100% | 4.949 ± 0.023 | 0.009 | 1.000 | 344 |
| 100 | single best candidate | 4.69 | 4.69 | 0.000 | 2.83 | 1.06 | 1.18 | **6.94** | 9.95 | 100% | 4.681 ± 0.023 | 0.007 | 1.000 | 139 |
| 100 | predictive stacking | 4.69 | 4.69 | 0.000 | 2.83 | 1.06 | 1.18 | **6.93** | 9.95 | 100% | 4.681 ± 0.023 | 0.007 | 1.000 | 142 |
| 100 | PAC Gibbs | 4.77 | 4.75 | 0.026 | 0.80 | 1.08 | 0.90 | **6.75** | 9.95 | 100% | 4.730 ± 0.023 | 0.008 | 1.000 | 181 |
| 100 | Ellipse+EB (one sample, N0+N1) | – | – | – | – | – | – | – (diagnostic objective, not a PAC bound) | – | – | – | – | 1.000 | 121 |
| 100 | bare ellipse (one sample, N0+N1) | – | – | – | – | – | – | – | – | – | 4.570 (contaminated), 4.556 (unmodified, 0.0% non-finite) | – | 1.000 | 105 |
| 300 | fixed prior $q_0$ | 5.23 | 4.95 | 0.252 | 0.00 | 0.55 | 0.52 | **6.31** | 9.95 | 100% | 4.949 ± 0.023 | 0.010 | 1.000 | 344 |
| 300 | single best candidate | 4.71 | 4.71 | 0.000 | 2.83 | 0.62 | 0.68 | **6.00** | 9.95 | 100% | 4.681 ± 0.023 | 0.008 | 1.000 | 140 |
| 300 | predictive stacking | 4.71 | 4.71 | 0.000 | 2.83 | 0.62 | 0.68 | **5.98** | 9.95 | 100% | 4.682 ± 0.023 | 0.007 | 1.000 | 142 |
| 300 | PAC Gibbs | 4.75 | 4.73 | 0.016 | 1.10 | 0.65 | 0.52 | **5.92** | 9.95 | 100% | 4.711 ± 0.023 | 0.008 | 1.000 | 163 |
| 300 | Ellipse+EB (one sample, N0+N1) | – | – | – | – | – | – | – (diagnostic objective, not a PAC bound) | – | – | – | – | 1.000 | 118 |
| 300 | bare ellipse (one sample, N0+N1) | – | – | – | – | – | – | – | – | – | 4.569 (contaminated), 4.555 (unmodified, 0.0% non-finite) | – | 1.000 | 106 |
