#!/bin/bash
# Create the vLLM profiler environment on an interactive ARC H100 node.
#
# Usage from the repo root after getting an interactive H100 allocation:
#   bash profiler/arc/setup_h100_env.sh
#
# The profiling batch script auto-detects this environment at .venv.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Uncomment and adjust for the ARC module stack available to your account.
# module purge
# module load cuda/12.4
# module load python/3.12

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not on PATH; installing it into the user Python environment."
    python3 -m pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is still not on PATH after installing with pip." >&2
    echo "Try: export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
    exit 1
fi

echo "Creating vLLM profiler environment in $REPO_ROOT/.venv"
./scripts/install-vllm.sh

echo
echo "Environment ready."
echo "Verify with:"
echo "  source .venv/bin/activate"
echo "  python -c 'import vllm; print(vllm.__version__)'"
echo
echo "Then submit:"
echo "  mkdir -p slurm-logs"
echo "  sbatch profiler/arc/profile_h100.sh"
