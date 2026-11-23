"""PACm / PAC2-T on ACE as a function of the iteration budget.

The optimiser is deterministic given the seed, so a run with a smaller budget
is the start of the run with a larger one: the rows for one training draw
trace what would be reported by stopping at each budget.

Usage: pacm_trajectory.py METHODS RATIO N_SEEDS BUDGETS
  e.g. pacm_trajectory.py PACm 20.0 10 5000,10000,20000,50000,100000,200000,400000
One part file per fit (``traj_parts/``); ``collect.py``-style merge at the end
when not run as an array.
"""
import _common  # noqa
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from comparisons import harness
from comparisons.low_noise_objectives import LowNoiseObjective

OUT = _common.HERE / "traj_parts"


def run(task):
    method, ratio, seed, max_iter = task
    warnings.filterwarnings("ignore")
    part = OUT / f"{method}_{ratio}_{seed}_{max_iter}.csv"
    if part.exists():
        return pd.read_csv(part).iloc[0].to_dict()
    problem = harness.ace_problem(ratio, seed)
    X, Xe, _ = problem.design()
    s = dict(harness.LOW_NOISE_SETTINGS[method])
    obj = s.pop("objective")
    t = time.perf_counter()
    model = LowNoiseObjective(
        obj, random_state=seed, max_iter=max_iter, max_fun=2 * max_iter, **s
    ).fit(X, problem.y_train)
    loc, scale = model.predict(Xe, return_std=True)
    pred = harness._gaussian(
        method, loc, scale, time.perf_counter() - t, harness._optimizer_info(model)
    )
    row = dict(pred.info)
    row.update(method=method, n=problem.y_train.size, seed=seed, budget=max_iter)
    row.update(harness.evaluate(pred, problem.y_test, problem.y_bounds))
    for k in ("objective_value_", "objective_data_", "objective_kl_"):
        if hasattr(model, k):
            row[k] = float(getattr(model, k))
    OUT.mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv(part, index=False)
    return row


if __name__ == "__main__":
    methods = sys.argv[1].split(",")
    ratio = float(sys.argv[2])
    n_seeds = int(sys.argv[3])
    budgets = sorted((int(b) for b in sys.argv[4].split(",")), reverse=True)
    tasks = _common.shard(
        [(m, ratio, s, b) for b in budgets for m in methods for s in range(n_seeds)]
    )
    with ProcessPoolExecutor(_common.N_WORKERS) as ex:
        rows = list(ex.map(run, tasks))
    print("done", len(rows))
