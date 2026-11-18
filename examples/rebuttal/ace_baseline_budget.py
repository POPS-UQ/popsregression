"""Do the unconverged ACE PACm / PAC2-T fits change with a 5x larger budget?"""
import _common  # noqa
import sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from comparisons import harness, calibration
from comparisons.low_noise_objectives import LowNoiseObjective

def run(task):
    method, ratio, seed, max_iter = task
    warnings.filterwarnings("ignore")
    part = _common.HERE / f'ace_budget_part_{method}_{ratio}_{seed}.csv'
    if part.exists():
        return pd.read_csv(part).iloc[0].to_dict()
    problem = harness.ace_problem(ratio, seed)
    X, Xe, _ = problem.design()
    s = dict(harness.LOW_NOISE_SETTINGS[method]); obj = s.pop("objective")
    t = time.perf_counter()
    model = LowNoiseObjective(obj, random_state=seed, max_iter=max_iter, max_fun=2 * max_iter, **s).fit(X, problem.y_train)
    loc, scale = model.predict(Xe, return_std=True)
    pred = harness._gaussian(method, loc, scale, time.perf_counter() - t, harness._optimizer_info(model))
    row = dict(pred.info); row.update(method=method, n=problem.y_train.size, seed=seed, budget=max_iter)
    row.update(harness.evaluate(pred, problem.y_test, problem.y_bounds))
    for k in ("objective_value_", "objective_data_", "objective_kl_"):
        if hasattr(model, k): row[k] = float(getattr(model, k))
    pd.DataFrame([row]).to_csv(_common.HERE / f'ace_budget_part_{method}_{ratio}_{seed}.csv', index=False)
    return row

if __name__ == "__main__":
    mi = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    # Largest fits first so array elements have similar run times.
    tasks = [(m, r, s, mi) for r in (20.0, 1.5) for m in ("PAC2-T", "PACm") for s in range(n_seeds)]
    tasks = _common.shard(tasks)
    with ProcessPoolExecutor(_common.N_WORKERS) as ex:
        rows = list(ex.map(run, tasks))
    print("done", len(rows))  # merge ace_budget_part_*.csv with collect.py
