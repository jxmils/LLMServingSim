#!/bin/bash
#SBATCH --job-name=llmservingsim-pd-baselines
#SBATCH --output=logs/pd-baselines-%j.out
#SBATCH --error=logs/pd-baselines-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=short

set -euo pipefail

module purge
module load GCCcore/13.2.0
module load CMake/3.26.3-GCCcore-13.2.0
module load protobuf/25.3-GCCcore-13.2.0
module load OpenMPI/4.1.6-GCC-13.2.0

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
PY310="${PY310:-/apps/system/easybuild/software/Python/3.10.8-GCCcore-12.2.0}"
PROTO="${PROTO:-/apps/system/easybuild/software/protobuf/25.3-GCCcore-13.2.0}"
VENV="${VENV:-.venv-sim}"

export PATH="$PROTO/bin:$PATH"
export LD_LIBRARY_PATH="$PY310/lib:/apps/system/easybuild/software/GCCcore/13.2.0/lib64:$PROTO/lib64:$PROTO/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="$PROTO${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export PROTOBUF_FROM_SOURCE=True

cd "$ROOT"
source "$VENV/bin/activate"

mkdir -p logs outputs workloads/pd_len_sweep outputs/baselines

echo "=== LLMServingSim PD baselines ==="
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "Repo:    $ROOT"
echo "Started: $(date -Is)"
echo

echo "Running bundled PD example baseline"
python -m serving \
  --cluster-config configs/cluster/single_node_pd_instance.json \
  --dtype bfloat16 \
  --block-size 16 \
  --dataset workloads/example_trace.jsonl \
  --output outputs/baseline_pd_example.csv \
  --num-reqs 10 \
  --log-interval 0.5

echo
echo "Generating fixed PD length-sweep workloads"
python - <<'PY'
import json
import os

os.makedirs("workloads/pd_len_sweep", exist_ok=True)

pairs = [(128, 128), (128, 256), (256, 128), (256, 256)]
num_reqs = 128

for inp, out in pairs:
    path = f"workloads/pd_len_sweep/i{inp}_o{out}.jsonl"
    with open(path, "w") as f:
        for _ in range(num_reqs):
            f.write(json.dumps({
                "input_toks": inp,
                "output_toks": out,
                "arrival_time_ns": 0,
            }) + "\n")
    print(path)
PY

echo
echo "Running fixed PD length-sweep baselines"
for w in i128_o128 i128_o256 i256_o128 i256_o256; do
  python -m serving \
    --cluster-config configs/cluster/single_node_pd_instance.json \
    --dtype bfloat16 \
    --block-size 16 \
    --dataset "workloads/pd_len_sweep/${w}.jsonl" \
    --output "outputs/baselines/pd_${w}.csv" \
    --num-reqs 128 \
    --log-interval 0.5
done

echo
echo "Baseline table:"
echo "Run,Input,Output,Purpose"
echo "pd_i128_o128,128,128,balanced small"
echo "pd_i128_o256,128,256,decode-heavy"
echo "pd_i256_o128,256,128,prefill-heavy"
echo "pd_i256_o256,256,256,balanced larger"
echo
echo "Finished: $(date -Is)"
