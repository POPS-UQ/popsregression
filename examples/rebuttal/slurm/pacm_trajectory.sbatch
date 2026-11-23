#!/bin/bash
# PACm on ACE N=700 at seven iteration budgets (5k to 400k), 10 training draws: one fit per element.
# Submit from the repository root:  sbatch -A <account> examples/rebuttal/slurm/pacm_trajectory.sbatch
#SBATCH --job-name=pops-traj
#SBATCH --partition=standard
#SBATCH --array=0-69
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --output=examples/rebuttal/out/logs/%x_%A_%a.out
export N_SHARDS=70
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}
uv run --frozen --extra examples python examples/rebuttal/pacm_trajectory.py PACm 20.0 10 5000,10000,20000,50000,100000,200000,400000
