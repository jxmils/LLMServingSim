#!/usr/bin/env python3
"""Derive ``workspace_gib`` from a vLLM bench run.

vLLM's ``meta.json`` (``python -m bench run``) records what the engine
actually allocated for KV: ``kv_cache.num_gpu_blocks`` and ``block_size`` at
``gpu_memory_utilization``. Everything vLLM held outside weights and KV --
activation peak, CUDA context, kernel workspaces -- is then

    workspace = hbm_total - weights_per_rank - num_gpu_blocks * block_size * kv_bytes_per_token_per_rank

which is the ``workspace_gib`` a HardwareSpec (or an instance's
``npu_mem.workspace_gib``) should carry so the simulator's KV capacity equals
the engine's block for block (G5 design 2.3; hardware validation tier 3).

Usage:
  python -m serving.tools.workspace_from_bench BENCH_DIR/meta.json --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --tp 8 [--hbm-gib 79.6] [--dtype bfloat16] [--kv-cache-dtype auto] [--ep 1] [--pp 1]
"""
import argparse
import json
import sys

from serving.core.memory_model import MemoryModel, GB_TO_BYTE


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meta", help="bench run meta.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--hbm-gib", type=float, default=None,
                    help="per-rank HBM in GiB; default: meta.json hardware.total_memory_bytes")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--kv-cache-dtype", default="auto")
    a = ap.parse_args(argv)
    meta = json.load(open(a.meta))
    kv = meta.get("kv_cache") or {}
    if "num_gpu_blocks" not in kv or "block_size" not in kv:
        sys.exit("meta.json carries no kv_cache.num_gpu_blocks / block_size (older bench recorder?)")
    hbm_bytes = None
    if a.hbm_gib is not None:
        hbm_bytes = int(a.hbm_gib * GB_TO_BYTE)
    else:
        hw = meta.get("hardware") or {}
        for key in ("total_memory_bytes", "total_memory"):
            if key in hw:
                hbm_bytes = int(hw[key]); break
    if hbm_bytes is None:
        sys.exit("pass --hbm-gib: meta.json has no hardware.total_memory_bytes")
    fp = {"bfloat16": 16, "float16": 16, "float32": 32, "fp8": 8, "int8": 8}[a.dtype]
    # A throwaway model at utilization 1.0 gives the weights and KV bytes per
    # token per rank under the frontend's own accounting.
    mm = MemoryModel(a.model, 0, 0, a.tp * a.pp * a.ep, a.tp, hbm_bytes / GB_TO_BYTE, 1, kv["block_size"], fp,
                     False, False, None, None, ep_size=a.ep, pp_size=a.pp,
                     kv_cache_dtype=a.kv_cache_dtype, npu_memory_utilization=1.0)
    kv_bytes = int(kv["num_gpu_blocks"]) * int(kv["block_size"]) * mm._bytes_per_token
    workspace = hbm_bytes - mm.weight - kv_bytes
    util = kv.get("gpu_memory_utilization")
    print(f"hbm per rank        {hbm_bytes / GB_TO_BYTE:10.3f} GiB")
    print(f"weights per rank    {mm.weight / GB_TO_BYTE:10.3f} GiB  (frontend accounting, tp={a.tp} ep={a.ep} pp={a.pp})")
    print(f"vLLM KV capacity    {kv_bytes / GB_TO_BYTE:10.3f} GiB  ({kv['num_gpu_blocks']} blocks x {kv['block_size']} tokens x "
          f"{mm._bytes_per_token} B/token/rank" + (f", at gpu_memory_utilization {util}" if util is not None else "") + ")")
    print(f"workspace_gib       {workspace / GB_TO_BYTE:10.3f}")
    if util is not None:
        implied = hbm_bytes * (1 - float(util))
        print(f"  of which vLLM's own headroom (1-util) {implied / GB_TO_BYTE:.3f} GiB; activation/context/workspace "
              f"{(workspace - implied) / GB_TO_BYTE:.3f} GiB")
    if workspace < 0:
        print("NOTE: negative workspace: the frontend counts more weight bytes than vLLM allocated "
              "(check dtype/ep/pp); the value still reproduces vLLM's block count exactly", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
