#!/usr/bin/env python3
"""Write a profile bundle's meta.yaml offline.

The profiler writes meta.yaml at the very end of a session; a session that
measured every category and then died in that last step (here: a venv without
pandas, which only the skew fit needs) leaves a complete bundle the simulator
refuses to load. This writes the same keys from the CSVs on disk plus what the
job log recorded (GPU, vLLM/CUDA versions, engine settings); no skew fit.

Usage:
  python -m serving.tools.write_bundle_meta profiler/perf/H100/Qwen/Qwen3-30B-A3B-Instruct-2507/bf16 \
      --gpu "NVIDIA H100 80GB HBM3" --hardware H100 --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --architecture qwen3_moe --vllm-version 0.19.2.dev0+precompiled --cuda-version 12.9 \
      --max-num-seqs 256 --max-num-batched-tokens 2048 --measurement-iterations 3 --note "..."
"""
import argparse
import csv
import datetime
import hashlib
import os
import sys

import yaml


def _vals(path, col):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return sorted({int(r[col]) for r in csv.DictReader(f)})


def _spec(vals):
    return f"{vals[0]}..{vals[-1]} ({len(vals)} points)" if vals else "none"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant_root")
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--architecture", required=True, help="profiler/models/<name>.yaml")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--vllm-version", required=True)
    ap.add_argument("--cuda-version", required=True)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--measurement-iterations", type=int, default=3)
    ap.add_argument("--attention-max-kv", type=int, default=16384)
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.variant_root)
    tps = sorted(int(d[2:]) for d in os.listdir(root) if d.startswith("tp") and d[2:].isdigit())
    if not tps:
        sys.exit(f"no tp<N> folders under {root}")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    arch_path = os.path.join(repo, "profiler", "models", f"{a.architecture}.yaml")
    arch_sha = hashlib.sha256(open(arch_path, "rb").read()).hexdigest() if os.path.isfile(arch_path) else None
    tp1 = os.path.join(root, f"tp{tps[0]}")
    att = os.path.join(tp1, "attention.csv")
    meta = {
        "profiler_version": "1.0.0",
        "vllm_version": a.vllm_version,
        "cuda_version": a.cuda_version,
        "gpu": a.gpu,
        "hardware": a.hardware,
        "profiled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "architecture": a.architecture,
        "architecture_sha256": arch_sha,
        "model": a.model,
        "variant": a.variant or os.path.basename(root),
        "tp_degrees": tps,
        "engine_effective": {"enforce_eager": True, "load_format": "dummy",
                             "max_num_batched_tokens": a.max_num_batched_tokens, "max_num_seqs": a.max_num_seqs,
                             "tensor_parallel_size": 1},
        "attention_grid": {"max_kv": a.attention_max_kv, "chunk_factor": 2.0, "kv_factor": 2.0,
                           "chunks": _spec(_vals(att, "prefill_chunk")), "n_decode": _spec(_vals(att, "n_decode")),
                           "kv": _spec(_vals(att, "kv_decode"))},
        "measurement_iterations": a.measurement_iterations,
        "skew_profile": {"enabled": False},
        "skew_fit": {"enabled": False, "note": "no skew sweep; alpha = 0"},
        "meta_written_offline": {"by": "serving/tools/write_bundle_meta.py",
                                 "reason": a.note or "profiler session ended before persist_meta"},
    }
    out = os.path.join(root, "meta.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"wrote {out}: tp {tps}, attention {meta['attention_grid']['chunks']} chunks")


if __name__ == "__main__":
    main()
