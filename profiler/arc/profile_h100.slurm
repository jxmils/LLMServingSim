#!/bin/bash
#SBATCH --job-name=llmsim-prof-h100
#SBATCH --clusters=htc
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_sku:H100
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/profile_h100_%j.out
#SBATCH --error=slurm-logs/profile_h100_%j.err
#
# Profile LLMServingSim operator latencies on an Oxford ARC HTC H100 node.
#
# Prerequisites (once, on an interactive H100 session):
#   git clone <LLMServingSim> && cd LLMServingSim
#   source profiler/arc/env_h100.sh    # after editing module lines in that file
#   ./scripts/install-vllm.sh
#   export HF_TOKEN=...                # if profiling gated models
#
# Submit from the repo root:
#   mkdir -p slurm-logs
#   sbatch profiler/arc/profile_h100.slurm
#
# Override sweep knobs at submit time:
#   sbatch --export=ALL,MODEL=meta-llama/Llama-3.1-8B,TP_DEGREES=1,SKIP_SKEW=1 \
#          profiler/arc/profile_h100.slurm
#
# ARC H100 nodes (see ARC User Guide — GPU Resources):
#   htc-g053–055  4× H100 82GB
#   htc-g058      4× H100 96GB (NVLink)
#   htc-g059–060  8× H100 80GB (NVLink)
#
# https://arc-user-guide.readthedocs.io/en/latest/arc-systems.html

set -euo pipefail

# =============================================================================
# Edit these, or pass via: sbatch --export=ALL,MODEL=...,SKIP_SKEW=1 ...
# =============================================================================

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
HARDWARE="${HARDWARE:-H100}"

TP_DEGREES="${TP_DEGREES:-1,2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-16384}"
ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-2.0}"
ATTENTION_KV_FACTOR="${ATTENTION_KV_FACTOR:-2.0}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"

# First-time on a new (hardware, model): set SKIP_SKEW=1 to save ~1–2 h per TP.
# Re-submit without it (or ONLY_SKEW=1) once the uniform grid is done.
SKIP_SKEW="${SKIP_SKEW:-}"
FORCE="${FORCE:-}"

# Optional: DTYPE=, KV_CACHE_DTYPE=, VARIANT=, VERBOSITY="--verbose"

# =============================================================================
# Runtime setup
# =============================================================================

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm-logs

echo "=== LLMServingSim profiler on ARC HTC H100 ==="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      ${SLURMD_NODENAME:-$(hostname)}"
echo "Submit:    $REPO_ROOT"
echo "Model:     $MODEL"
echo "Hardware:  $HARDWARE"
echo "TP:        $TP_DEGREES"
echo "Started:   $(date -Is)"
echo

# Modules — uncomment and match `module avail` on ARC
# module purge
# module load cuda/12.4
# module load python/3.12

VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-vllm}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: vLLM venv not found at $VENV_DIR" >&2
    echo "On an interactive H100 node run: source profiler/arc/env_h100.sh && ./scripts/install-vllm.sh" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN is unset — gated model configs may fail to download." >&2
fi

nvidia-smi
echo

# Build profiler command (same flags as profiler/profile.sh)
cmd=(python3 -m profiler profile "$MODEL" --hardware "$HARDWARE")
cmd+=(--tp "$TP_DEGREES")
cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
cmd+=(--attention-max-kv "$ATTENTION_MAX_KV")
cmd+=(--attention-chunk-factor "$ATTENTION_CHUNK_FACTOR")
cmd+=(--attention-kv-factor "$ATTENTION_KV_FACTOR")
cmd+=(--measurement-iterations "$MEASUREMENT_ITERATIONS")
[[ -n "${SKIP_SKEW:-}" ]]       && cmd+=(--skip-skew)
[[ -n "${FORCE:-}" ]]            && cmd+=(--force)
[[ -n "${DTYPE:-}" ]]            && cmd+=(--dtype "$DTYPE")
[[ -n "${KV_CACHE_DTYPE:-}" ]]   && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ -n "${VARIANT:-}" ]]          && cmd+=(--variant "$VARIANT")
[[ -n "${VERBOSITY:-}" ]]        && cmd+=($VERBOSITY)

echo "Running: ${cmd[*]}"
echo
"${cmd[@]}"

echo
echo "Finished: $(date -Is)"
echo "Output:   $REPO_ROOT/profiler/perf/$HARDWARE/$MODEL/"
