#!/bin/bash
#SBATCH --job-name=llmsim-prof-h100
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_sku:H100
#SBATCH --mem=128G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-logs/profile_h100_%j.out
#SBATCH --error=slurm-logs/profile_h100_%j.err
#
# Submit from the repo root:
#   mkdir -p slurm-logs
#   sbatch profiler/arc/profile_h100.sh
#
# Common quick first pass:
#   sbatch --export=ALL,MODEL=meta-llama/Llama-3.1-8B,TP_DEGREES=1,SKIP_SKEW=1 \
#          profiler/arc/profile_h100.sh
#
# For configs/cluster/single_node_single_instance_H100.json:
#   sbatch --export=ALL,MODEL=meta-llama/Llama-3.1-70B,TP_DEGREES=1,4,SKIP_SKEW=1 \
#          profiler/arc/profile_h100.sh

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
HARDWARE="${HARDWARE:-H100}"

TP_DEGREES="${TP_DEGREES:-1,2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-16384}"
ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-2.0}"
ATTENTION_KV_FACTOR="${ATTENTION_KV_FACTOR:-2.0}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"

# Set SKIP_SKEW=1 for a faster bring-up run. Re-submit with ONLY_SKEW=1 later
# if you want to add just the heterogeneous-decode skew fit.
SKIP_SKEW="${SKIP_SKEW:-}"
ONLY_SKEW="${ONLY_SKEW:-}"
FORCE="${FORCE:-}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm-logs

echo "=== LLMServingSim profiler on ARC HTC H100 ==="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      ${SLURMD_NODENAME:-$(hostname)}"
echo "Repo:      $REPO_ROOT"
echo "Model:     $MODEL"
echo "Hardware:  $HARDWARE"
echo "TP:        $TP_DEGREES"
echo "Started:   $(date -Is)"
echo

# Uncomment and adjust for the ARC module stack available to your account.
# module purge
# module load cuda/12.4
# module load python/3.12

if [[ -z "${VENV_DIR:-}" ]]; then
    if [[ -f "$REPO_ROOT/.venv-vllm/bin/activate" ]]; then
        VENV_DIR="$REPO_ROOT/.venv-vllm"
    else
        VENV_DIR="$REPO_ROOT/.venv"
    fi
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: vLLM venv not found at $VENV_DIR" >&2
    echo "Create it on an interactive H100 node from the repo root with:" >&2
    echo "  ./scripts/install-vllm.sh" >&2
    echo "Then run interactively with:" >&2
    echo "  VENV_DIR=$VENV_DIR bash profiler/arc/profile_h100.sh" >&2
    echo "Or submit with VENV_DIR=/path/to/venv if your environment uses a different location." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN is unset; gated model downloads may fail." >&2
fi

nvidia-smi
echo

cmd=(python3 -m profiler profile "$MODEL" --hardware "$HARDWARE")
cmd+=(--tp "$TP_DEGREES")
cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
cmd+=(--attention-max-kv "$ATTENTION_MAX_KV")
cmd+=(--attention-chunk-factor "$ATTENTION_CHUNK_FACTOR")
cmd+=(--attention-kv-factor "$ATTENTION_KV_FACTOR")
cmd+=(--measurement-iterations "$MEASUREMENT_ITERATIONS")
[[ -n "${SKIP_SKEW:-}" ]]       && cmd+=(--skip-skew)
[[ -n "${ONLY_SKEW:-}" ]]       && cmd+=(--only-skew)
[[ -n "${FORCE:-}" ]]           && cmd+=(--force)
[[ -n "${DTYPE:-}" ]]           && cmd+=(--dtype "$DTYPE")
[[ -n "${KV_CACHE_DTYPE:-}" ]]  && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ -n "${VARIANT:-}" ]]         && cmd+=(--variant "$VARIANT")
[[ -n "${VERBOSITY:-}" ]]       && cmd+=($VERBOSITY)

echo "Running: ${cmd[*]}"
echo
"${cmd[@]}"

echo
echo "Finished: $(date -Is)"
echo "Output:   $REPO_ROOT/profiler/perf/$HARDWARE/$MODEL/"
