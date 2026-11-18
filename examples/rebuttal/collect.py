"""Merge per-task outputs into sensitivity.csv and ace_baseline_budget.csv."""
import _common  # noqa
import pandas as pd

for pattern, name, expected in (
    ("sens_parts/*.csv", "sensitivity.csv", 420),
    ("ace_budget_part_*.csv", "ace_baseline_budget.csv", None),
):
    files = sorted(_common.HERE.glob(pattern))
    if files:
        pd.concat([pd.read_csv(f) for f in files]).to_csv(_common.HERE / name, index=False)
    print(name, len(files), "parts" + (f" of {expected}" if expected else ""))
