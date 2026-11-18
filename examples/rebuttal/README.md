# AISTATS rebuttal analyses

Additional runs for the rebuttal of submission 1391. All use the fitted
estimators and comparison harness in `examples/` unchanged.

| Script | Question | Output |
|---|---|---|
| `sensitivity.py` | PAC bound vs output range and mixture weight, with held-out risk | `out/sens_parts/` |
| `ace_baseline_budget.py` | ACE PACm / PAC2-T with 5x the optimizer budget | `out/ace_budget_part_*.csv` |
| `burgers_propagation.py [grad\|front]` | Burgers QoI coverage: function space vs parameter distribution | `out/burgers_propagation*.csv` |

Every task writes its own file and finished tasks are skipped, so runs can be
stopped, restarted and split across machines.

## Great Lakes

```bash
cd popsregression                      # this branch
mkdir -p examples/rebuttal/out/logs
uv sync --frozen --extra examples      # once, on the login node
sbatch -A <account> examples/rebuttal/slurm/sensitivity.sbatch
sbatch -A <account> examples/rebuttal/slurm/ace_budget.sbatch
uv run --frozen --extra examples python examples/rebuttal/collect.py   # when done
```
