# Profiling on Oxford ARC (HTC H100)

Run the vLLM layerwise profiler on [ARC](https://www.arc.ox.ac.uk/) HTC **H100** nodes
(`htc-g053–055`, `htc-g058`, `htc-g059–060`). GPUs are only on the **htc** cluster;
use `--clusters=htc` in SLURM scripts ([ARC User Guide — GPU resources](https://arc-user-guide.readthedocs.io/en/latest/job-scheduling.html#gpu-resources)).

Output lands under `profiler/perf/H100/<model>/<variant>/` for use in simulations with
`"hardware": "H100"` in your cluster config.

## One-time setup (interactive H100)

Log in to `htc-login.arc.ox.ac.uk`, then start an interactive GPU session:

```bash
srun -p interactive --clusters=htc --gres=gpu:1 \
     --constraint='gpu_sku:H100' --mem=128G --cpus-per-task=8 \
     --time=04:00:00 --pty bash
```

Clone LLMServingSim, edit module lines in `profiler/arc/env_h100.sh` if needed, then:

```bash
cd LLMServingSim
source profiler/arc/env_h100.sh
export HF_TOKEN="hf_..."   # gated models only
./scripts/install-vllm.sh   # creates .venv-vllm with vLLM 0.19.0
```

## Submit a profile job

From the repo root on a login node:

```bash
mkdir -p slurm-logs
sbatch profiler/arc/profile_h100.slurm
```

Defaults: `meta-llama/Llama-3.1-8B`, `HARDWARE=H100`, `TP_DEGREES=1,2`, full skew sweep.

### Common overrides

```bash
# Smaller / faster first pass (skip skew, single TP)
sbatch --export=ALL,MODEL=meta-llama/Llama-3.1-8B,TP_DEGREES=1,SKIP_SKEW=1 \
       profiler/arc/profile_h100.slurm

# Qwen3-32B (needs ~1× H100 80GB; use 128G host RAM)
sbatch --export=ALL,MODEL=Qwen/Qwen3-32B,SKIP_SKEW=1 \
       profiler/arc/profile_h100.slurm

# Three-model sweep (long partition, ~1–2 days)
sbatch profiler/arc/profile_h100_all.slurm
```

## After profiling

Point cluster configs at the new perf tree:

```json
"hardware": "H100"
```

Run the simulator (CPU-only on ARC is fine) with the same `model_name` you profiled.

## Notes

- **Wall time**: uniform + skew for one model at TP 1,2 is often **2–6 hours**; 70B-class models can approach the paper’s **~2 h per TP** on H100. `medium` allows up to 48 h.
- **Memory**: scripts request **128G** host RAM for vLLM; raise `--mem` for very large models.
- **Co-investment H100** nodes may be limited to the **short** partition (12 h max); switch `#SBATCH --partition=short` if your job is killed on time.
- **Docker** (`scripts/docker-vllm.sh`) is optional on ARC; bare-metal `install-vllm.sh` is the intended path here.
- Check available constraints: `sinfo -o "%P %G %f" | grep -i h100`
