#!/bin/bash
# Bound sensitivity to output range and mixture weight: 420 tasks over 70 array elements.
# Submit from the repository root:  sbatch -A <account> examples/rebuttal/slurm/sensitivity.sbatch
#SBATCH --job-name=pops-sens
#SBATCH --partition=standard
#SBATCH --array=0-69
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=examples/rebuttal/out/logs/%x_%A_%a.out
export N_SHARDS=70
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}
uv run --frozen --extra examples python examples/rebuttal/sensitivity.py
