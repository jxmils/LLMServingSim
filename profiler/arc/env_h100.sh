#!/bin/bash
# Source this from an interactive H100 session before building the vLLM venv:
#
#   srun -p interactive --clusters=htc --gres=gpu:1 \
#        --constraint='gpu_sku:H100' --mem=128G --cpus-per-task=8 \
#        --time=04:00:00 --pty bash
#   source profiler/arc/env_h100.sh
#   ./scripts/install-vllm.sh
#
# Adjust module names to match your ARC software stack:
#   module avail cuda
#   module avail python

set -euo pipefail

# --- ARC modules (edit for your project) ---------------------------------------
# module purge
# module load cuda/12.4
# module load python/3.12

# --- Hugging Face cache (use scratch if home quota is tight) -------------------
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# Gated models (Llama, some Qwen variants)
# export HF_TOKEN="hf_..."

# --- vLLM venv (created by scripts/install-vllm.sh on a GPU node) ------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-vllm}"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
else
    echo "No venv at $VENV_DIR — run ./scripts/install-vllm.sh from $REPO_ROOT on an H100 node first." >&2
    return 1 2>/dev/null || exit 1
fi

cd "$REPO_ROOT"
echo "LLMServingSim repo: $REPO_ROOT"
python -c "import vllm; print('vLLM', vllm.__version__)"
nvidia-smi -L
