#!/usr/bin/env python3
"""Compile smoke test for the HTC pilot vLLM (dummy weights, a few seconds of decode).

Boots vLLM once with torch.compile + CUDA graphs (the NVTX-label import hook in
bench/compat/htc_pilot_vllm must be on PYTHONPATH) or, with --eager, eagerly;
decodes a fixed batch and prints one JSON line: mode, compilation settings,
boot time, decode tokens/s. Two runs (compiled, eager) give the eager launch
overhead per decode step directly.

  python bench/compat/smoke_compile.py --tp 2 [--ep] [--eager] [--batch 64 --out-toks 128]
"""
import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--ep", action="store_true")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--nccl-only", action="store_true",
                    help="as the hardware-validation bench: no FlashInfer all-reduce + RMSNorm fusion")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--in-toks", type=int, default=128)
    ap.add_argument("--out-toks", type=int, default=128)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, enable_expert_parallel=a.ep, load_format="dummy",
              max_model_len=4096, max_num_seqs=max(a.batch, 8), max_num_batched_tokens=8192,
              enforce_eager=a.eager, disable_custom_all_reduce=True, seed=0, gpu_memory_utilization=0.85,
              **({"compilation_config": {"pass_config": {"fuse_allreduce_rms": False}}} if a.nccl_only else {}))
    boot = time.time() - t0
    cc = llm.llm_engine.vllm_config.compilation_config
    comp = {k: str(getattr(cc, k)) for k in ("mode", "level", "cudagraph_mode") if hasattr(cc, k)}
    sp = SamplingParams(max_tokens=a.out_toks, min_tokens=a.out_toks, ignore_eos=True, temperature=0.0)
    prompts = [{"prompt_token_ids": [1000 + (i * 7 + j) % 5000 for j in range(a.in_toks)]} for i in range(a.batch)]
    llm.generate(prompts[:4], SamplingParams(max_tokens=8, ignore_eos=True))  # warm-up
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t1
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    hook = sys.modules.get("sitecustomize")
    print("SMOKE " + json.dumps({
        "mode": "eager" if a.eager else "compiled", "tp": a.tp, "ep": a.ep, "nccl_only": a.nccl_only, "compilation": comp,
        "boot_s": round(boot, 1), "batch": a.batch, "out_toks": a.out_toks, "decode_wall_s": round(dt, 3),
        "tok_per_s": round(ntok / dt, 1), "approx_step_ms": round(dt / a.out_toks * 1e3, 2),
        "hook_rewrites": len(getattr(hook, "REWRITTEN", []) or []),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
