"""Burgers QoI propagation: function space versus parameter uncertainty.

For each configuration (POD basis regime, rank, spatial observations per
case, training cases) and method we report, on 60 held-out cases:

- coverage of the TRUE dissipation / front steepness by the 95.45% interval
  of 2000 propagated parameter draws (the paper's metric);
- coverage of the ORACLE value: the QoI of the L2 projection of the true
  field on the surrogate space (the best field any parameter can produce for
  that case). High oracle coverage with low true coverage means the
  parameter distribution is adequate for what the space can express and the
  shortfall is the function space;
- the relative gap between oracle and true QoI (a property of the space
  only);
- pointwise field coverage over the whole grid and near the front.

basis='smooth' is the paper's offline basis (nu in [0.055, 0.08], t in
[0.15, 0.35]); 'full' draws the offline snapshots from the deployment box.
"""
import _common  # noqa
import warnings
from concurrent.futures import ProcessPoolExecutor

import example_burgers as burgers
import example_burgers_pod as pod
import numpy as np
import pandas as pd
from comparisons import harness
from example_burgers_qoi import qoi

METHODS = ("POPS ellipse", "Ellipse+EB", "Ellipse+PAC")
LEVEL = 0.9545
GRAD_SCALE = 0.2  # |d/dx (u - mean)| < 13 on the deployment box (checked on 2000 cases)
_CACHE = {}


def basis(regime, rank, n_cases=48, seed=pod.SEED + 1000):
    key = (regime, rank)
    if key not in _CACHE:
        rng = np.random.default_rng(seed)
        nu_r = pod.POD_NU_RANGE if regime == "smooth" else pod.NU_RANGE
        t_r = pod.POD_T_RANGE if regime == "smooth" else pod.T_RANGE
        cases = np.column_stack([rng.uniform(*nu_r, n_cases), rng.uniform(*pod.AMP_RANGE, n_cases),
                                 rng.uniform(*t_r, n_cases)])
        snaps = np.asarray([burgers.burgers_solution(*c)[1] for c in cases])
        mean = snaps.mean(axis=0)
        _, _, vt = np.linalg.svd(snaps - mean, full_matrices=False)
        test_cases = harness._draw_burgers_cases(np.random.default_rng(pod.TEST_SEED), 120)[:60]
        truth = np.asarray([burgers.burgers_solution(*c)[1] for c in test_cases])
        _CACHE[key] = (mean, vt[:rank], test_cases, truth)
    return _CACHE[key]


def run(task):
    regime, rank, ppc, n_cases, seed, grad = task
    warnings.filterwarnings("ignore")
    mean, modes, test_cases, truth = basis(regime, rank)
    cases = harness._draw_burgers_cases(np.random.default_rng(10_000 + seed), n_cases)
    X, y = pod.simulate_cases(cases, modes, mean, points_per_case=ppc, random_x=True, seed=20_000 + seed)
    groups = np.repeat(np.arange(n_cases), ppc)
    x = np.linspace(0.0, 2.0 * np.pi, burgers.N_GRID, endpoint=False)
    if grad:
        # Add the spatial derivative at the same sampled points as extra rows
        # (central differences on the solver grid, as in the QoI), scaled by
        # GRAD_SCALE so both outputs share the declared range.
        dx = x[1] - x[0]
        ddx = lambda f: (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * dx)
        rng = np.random.default_rng(20_000 + seed)
        Xg, yg = [], []
        for nu, amp, t in cases:
            idx = np.sort(rng.choice(burgers.N_GRID, size=ppc, replace=False))  # same draws as simulate_cases
            if grad == "front":
                idx = np.array([burgers.N_GRID // 2])  # x = pi, where the front sits by symmetry
            u = burgers.burgers_solution(nu, amp, t)[1]
            g = pod.parameter_features(np.array([nu]), np.array([amp]), np.array([t]))[0]
            Xg.append(np.einsum("ni,j->nij", ddx(modes).T[idx], g).reshape(idx.size, -1))
            yg.append(ddx(u - mean)[idx])
        X = np.vstack([X, GRAD_SCALE * np.vstack(Xg)])
        y = np.concatenate([y, GRAD_SCALE * np.concatenate(yg)])
        groups = np.concatenate([groups, np.repeat(np.arange(n_cases), len(yg[0]))])
    problem = harness.Problem("burgers", X, y, X[:1], y[:1], harness.BURGERS_Y_BOUNDS, [], groups=groups)
    near = np.abs(x - np.pi) < 0.4
    resid = truth - mean
    proj = mean + (resid @ modes.T) @ modes  # POD modes are orthonormal on the grid
    true_q = qoi(truth, test_cases[:, 0])
    orac_q = qoi(proj, test_cases[:, 0])
    rows = []
    for method in METHODS:
        row = dict(grad=grad, basis=regime, rank=rank, ppc=ppc, cases=n_cases, seed=seed, method=method, P=X.shape[1])
        try:
            coef, offset = harness.parameter_draws(method, problem, seed, 2000)
            acc = {k: [] for k in ("d_true", "s_true", "d_orac", "s_orac", "d_w", "s_w", "pt", "pt_near")}
            for i, (nu, amp, t) in enumerate(test_cases):
                F = pod.rom_features(np.full_like(x, nu), np.full_like(x, amp), np.full_like(x, t), x, modes)
                fields = mean[None, :] + (F @ coef + offset[None, :]).T
                lo, hi = np.quantile(fields, [0.5 * (1 - LEVEL), 0.5 * (1 + LEVEL)], axis=0)
                inside = (truth[i] >= lo) & (truth[i] <= hi)
                acc["pt"].append(inside.mean()); acc["pt_near"].append(inside[near].mean())
                for tag, vals, j in (("d", qoi(fields, nu)[0], 0), ("s", qoi(fields, nu)[1], 1)):
                    l, u = np.quantile(vals, [0.5 * (1 - LEVEL), 0.5 * (1 + LEVEL)])
                    acc[f"{tag}_true"].append(l <= true_q[j][i] <= u)
                    acc[f"{tag}_orac"].append(l <= orac_q[j][i] <= u)
                    acc[f"{tag}_w"].append((u - l) / abs(true_q[j][i]))
            row.update({k: float(np.mean(v)) for k, v in acc.items() if not k.endswith("_w")})
            row.update(d_w=float(np.median(acc["d_w"])), s_w=float(np.median(acc["s_w"])))
        except Exception as e:  # noqa
            row["error"] = repr(e)[:200]
        row["d_gap"] = float(np.median(orac_q[0] / true_q[0] - 1))
        row["s_gap"] = float(np.median(orac_q[1] / true_q[1] - 1))
        row["s_gap_q05"] = float(np.quantile(orac_q[1] / true_q[1] - 1, 0.05))
        row["field_rel_err"] = float(np.median(np.linalg.norm(truth - proj, axis=1) / np.linalg.norm(resid, axis=1)))
        rows.append(row)
    return rows


if __name__ == "__main__":
    import sys
    grad = sys.argv[1] if len(sys.argv) > 1 else False
    tasks = [(b, r, p, n, s, grad) for b in ("smooth", "full") for r in ((2, 4) if grad else (2, 4, 6))
             for p in (3, 16) for n in (8, 80) for s in range(20)]
    with ProcessPoolExecutor(7) as ex:
        rows = [r for rs in ex.map(run, tasks, chunksize=2) for r in rs]
    pd.DataFrame(rows).to_csv(_common.HERE / (f"burgers_propagation_{grad}.csv" if grad else "burgers_propagation.csv"), index=False)
    print("done", len(rows))
