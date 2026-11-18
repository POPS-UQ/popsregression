"""Sensitivity of the POPS Ellipse+PAC bound to the declared output range and
to the uniform-mixture weight beta, with held-out floored log risk alongside.

Range scale s widens y_bounds about its centre. 'tied' keeps the paper's
a_min = 0.01 R_Y; 'fixed' holds a_min at the paper's value while R_Y grows.
The stored bound uses the paper's Hoeffding transfer (cap after averaging).
"""
import _common  # noqa
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from comparisons import harness
from example_repeated_splits import make_problem

from popsregression import POPSEllipseRegression

SIZES = {"quartic": (10, 20, 50, 100, 300), "burgers": (8, 16, 24, 40, 80), "ace": (1.5, 3.0, 6.0, 20.0)}
CONFIGS = (
    [("beta", 1.0, b, "tied") for b in (0.005, 0.01, 0.02, 0.05, 0.1)]
    + [("range", s, 0.02, "tied") for s in (1.5, 2.0, 4.0)]
    + [("range", s, 0.02, "fixed") for s in (1.5, 2.0, 4.0)]
)


def run(task):
    name, size, seed = task
    warnings.filterwarnings("ignore")
    part = _common.HERE / "sens_parts" / f"{name}_{size}_{seed}.csv"
    if part.exists():
        return pd.read_csv(part).to_dict("records")
    problem = make_problem(name, size, seed)
    lo, hi = problem.y_bounds
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    rows = []
    for kind, s, beta, amin in CONFIGS:
        yb = (mid - s * half, mid + s * half)
        kw = dict(regularization="PAC", random_state=seed, preprocessor=problem.preprocessor,
                  y_bounds=yb, floor_weight=beta)
        if amin == "fixed":
            kw["min_half_width"] = 0.01 * (hi - lo)
        row = dict(problem=name, n_units=problem.n_units, seed=seed, sweep=kind, scale=s, beta=beta, a_min=amin)
        try:
            model = POPSEllipseRegression(**kw).fit(problem.X_train, problem.y_train, groups=problem.groups)
            c = model.certificate_
            Rg = c.loss_upper - c.loss_lower
            K = c.folds[0].n_stored
            stored = min(c.loss_upper, c.continuous_bound + Rg * np.sqrt(np.log(200) / (2 * K)))
            Xe, y = problem.X_test, problem.y_test
            logp = np.asarray(model.predict_logpdf(Xe, y))
            w = yb[1] - yb[0]
            test = -np.mean(np.logaddexp(np.log1p(-beta) + logp, np.log(beta / w)))
            l, u = model.predict_interval(Xe, 0.9545)
            row.update(ceiling=c.loss_upper, loss_lower=c.loss_lower, continuous=c.continuous_bound,
                       stored=stored, test_floored=test, test_raw=-np.mean(logp),
                       zero_density=np.mean(np.isneginf(logp)),
                       coverage=np.mean((y >= l) & (y <= u)), width=np.mean(u - l),
                       support_lb=(c.loss_upper - stored) / Rg)
        except Exception as e:  # noqa
            row["error"] = repr(e)[:200]
        rows.append(row)
    part.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False)
    return rows


if __name__ == "__main__":
    tasks = [(n, s, seed) for n, ss in SIZES.items() for s in ss for seed in range(30)]
    tasks = _common.shard(tasks)
    with ProcessPoolExecutor(_common.N_WORKERS) as ex:
        rows = [r for rs in ex.map(run, tasks) for r in rs]
    print("done", len(rows))  # merge sens_parts/*.csv with collect.py
