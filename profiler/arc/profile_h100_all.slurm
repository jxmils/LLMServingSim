#!/bin/bash
#SBATCH --job-name=llmsim-prof-h100-all
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_sku:H100
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-logs/profile_h100_all_%j.out
#SBATCH --error=slurm-logs/profile_h100_all_%j.err
#
# Multi-model H100 sweep (same models as profiler/profile-all.sh).
# Submit: mkdir -p slurm-logs && sbatch profiler/arc/profile_h100_all.slurm
#
# Coarser / faster first pass:
#   sbatch --export=ALL,SKIP_SKEW=1,ATTENTION_CHUNK_FACTOR=4.0 \
#          profiler/arc/profile_h100_all.slurm

set -euo pipefail

export HARDWARE="${HARDWARE:-H100}"
export TP_DEGREES="${TP_DEGREES:-1,2}"
export SKIP_SKEW="${SKIP_SKEW:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm-logs

# shellcheck disable=SC1090
[[ -f "${VENV_DIR:-$REPO_ROOT/.venv-vllm}/bin/activate" ]] || {
    echo "ERROR: create .venv-vllm first (see profiler/arc/README.md)" >&2
    exit 1
}
source "${VENV_DIR:-$REPO_ROOT/.venv-vllm}/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

exec "$REPO_ROOT/profiler/profile-all.sh"
