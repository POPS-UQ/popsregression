#!/bin/bash
# ACE PACm / PAC2-T at 100k iterations, 30 training draws, N=52 and N=700: one fit per element.
# Submit from the repository root:  sbatch -A <account> examples/rebuttal/slurm/ace_budget.sbatch
#SBATCH --job-name=pops-ace
#SBATCH --partition=standard
#SBATCH --array=0-119
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=examples/rebuttal/out/logs/%x_%A_%a.out
export N_SHARDS=120
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}
uv run --frozen --extra examples python examples/rebuttal/ace_baseline_budget.py 100000 30
