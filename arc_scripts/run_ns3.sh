#!/bin/bash
#SBATCH --job-name=llmservingsim-ns3
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=short
#SBATCH --output=logs/ns3-%j.out
#SBATCH --error=logs/ns3-%j.err

set -euo pipefail

module purge
module load GCCcore/13.2.0
module load CMake/3.26.3-GCCcore-13.2.0
module load protobuf/25.3-GCCcore-13.2.0
module load OpenMPI/4.1.6-GCC-13.2.0

ROOT=/data/engs-glass/catz0932/LLMServingSim
PY310=/apps/system/easybuild/software/Python/3.10.8-GCCcore-12.2.0
PROTO=/apps/system/easybuild/software/protobuf/25.3-GCCcore-13.2.0

export PATH="$PROTO/bin:$PATH"
export LD_LIBRARY_PATH="$PY310/lib:/apps/system/easybuild/software/GCCcore/13.2.0/lib64:$PROTO/lib64:$PROTO/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="$PROTO${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export PROTOBUF_FROM_SOURCE=True

cd "$ROOT"
source .venv-sim/bin/activate

mkdir -p outputs

python -m serving \
  --cluster-config configs/cluster/ns3_8x_rtxpro6000_tp2.json \
  --dtype bfloat16 \
  --block-size 16 \
  --network-backend ns3 \
  --dataset workloads/example_trace.jsonl \
  --output outputs/ns3_8x_rtxpro6000_tp2.csv \
  --num-reqs 1 \
  --log-level DEBUG