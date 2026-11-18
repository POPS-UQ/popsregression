"""Shared setup for the AISTATS rebuttal analyses.

Run from the repository root with uv, e.g.
``uv run --extra examples python examples/rebuttal/sensitivity.py``.
Outputs go to ``$REBUTTAL_OUT`` (default ``examples/rebuttal/out``).
"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1]
HERE = Path(os.environ.get("REBUTTAL_OUT", Path(__file__).resolve().parent / "out"))
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(EXAMPLES))
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("N_WORKERS", 7)))


def shard(tasks):
    """Keep the tasks of this Slurm array element (all of them if not in an array).

    ``N_SHARDS`` sets the number of array elements; element ``i`` takes tasks
    ``i, i + N_SHARDS, ...``.
    """
    i = os.environ.get("SLURM_ARRAY_TASK_ID")
    if i is None:
        return tasks
    return tasks[int(i) :: int(os.environ["N_SHARDS"])]
