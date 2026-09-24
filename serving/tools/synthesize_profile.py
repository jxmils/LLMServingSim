"""Synthesize a profile bundle for non-GPU hardware from a HardwareSpec + ModelSpec.

The simulator's hardware interface is the CSV bundle under
`profiler/perf/<HARDWARE>/<MODEL>/<variant>/` (docs: profiler/output-bundle).
For hardware vLLM cannot run on, the documented path is to produce that
bundle from another source. This tool produces it from a roofline:

    time_us = max(flops / (peak_dense_tflops * achieved_fraction),
                  bytes / hbm_bandwidth_gbps) + kernel_floor_us

with FLOPs and bytes derived from the ModelSpec's shapes at every operating
point the measured RTX bundles sweep (so the simulator's interpolation covers
the same points and extrapolates nowhere new). Every number is
`analytically_estimated`; the label, the formula and the spec hashes go into
meta.yaml and the simulator prints the label with its results.

Nothing here charges communication, DMA or offloading: the HardwareSpec's
`timings_include` says so, and the backend charges those itself.

Usage:
  python -m serving.tools.synthesize_profile --hardware-spec configs/specs/hardware/b200-like.json \
      --model-spec configs/specs/model/qwen3-235b-a22b.json --tp 1 --tp 8 --tp 64 \
      [--grid-from profiler/perf/RTXPRO6000/Qwen/Qwen3-30B-A3B-Instruct-2507/bf16/tp1] [--perf-root profiler/perf]
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import yaml

from serving.core.specs import load_spec

BYTES = {"bfloat16": 2, "float16": 2, "fp8": 1}

# Grids of the measured RTX bundles (profiler/perf/RTXPRO6000/*/bf16/tp1).
DENSE_TOKENS = list(range(1, 17)) + list(range(20, 65, 4)) + list(range(80, 2049, 16))
PER_SEQUENCE = list(range(1, 17)) + list(range(20, 65, 4)) + list(range(80, 257, 16))
MOE_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
MOE_ACTIVATED = [8, 16, 32, 64, 128]
ATTN_PC = [0, 16, 24, 32, 36, 54, 64, 81, 122, 128, 182, 256, 273, 410, 512, 615, 923, 1024, 1384, 2048]
ATTN_KV = [0, 16, 32, 64, 128, 256, 512, 768, 1024, 1152, 1728, 2048, 2592, 3888, 4096, 5832, 8192, 8748, 13122, 16384]
ATTN_ND = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]

DENSE_LAYERS = ["embedding", "layernorm", "qkv_proj", "qk_norm", "rotary_emb", "o_proj", "final_layernorm"]
TP_STABLE = {"layernorm", "qk_norm", "final_layernorm", "sampler"}


class Roofline:
    def __init__(self, hw):
        c = hw["compute"]
        self.peak_flops = float(c["peak_dense_tflops"]) * 1e12 * float(c.get("achieved_fraction", 1.0))
        self.bw = float(hw["hbm_bandwidth_gbps"]) * 1e9
        self.floor_us = float(c["kernel_floor_us"])

    def us(self, flops: float, nbytes: float) -> float:
        t = max(flops / self.peak_flops, nbytes / self.bw)
        return t * 1e6 + self.floor_us


class Shapes:
    """Per-rank operator shapes for a TP degree, from the ModelSpec."""

    def __init__(self, model, tp: int):
        self.d = int(model["hidden_size"])
        self.tp = tp
        heads, kv_heads, hd = int(model["num_attention_heads"]), int(model["num_key_value_heads"]), int(model["head_dim"])
        self.hd = hd
        self.heads_local = max(heads // tp, 1)
        # vLLM replicates KV heads when tp exceeds them
        self.kv_heads_local = max(kv_heads // tp, 1)
        self.q_local = self.heads_local * hd
        self.kv_local = self.kv_heads_local * hd
        self.vocab_local = int(model["vocab_size"]) // tp
        self.vocab = int(model["vocab_size"])
        self.fp = BYTES[model["torch_dtype"]]
        moe = model.get("moe")
        self.moe = moe
        if moe:
            self.d_ff = int(moe["moe_intermediate_size"])
            self.experts = int(moe["num_experts"])
        self.attention_kind = model["attention"]["kind"]
        if self.attention_kind == "mla":
            att = model["attention"]
            self.kv_lora_rank = int(att["kv_lora_rank"])
            self.rope_dim = int(att["qk_rope_head_dim"])

    # ---- dense (tokens) ----
    def dense(self, layer: str, t: int) -> Tuple[float, float]:
        d, fp = self.d, self.fp
        if layer == "embedding":
            return 0.0, t * d * fp
        if layer in ("layernorm", "final_layernorm"):
            return 5.0 * t * d, 2 * t * d * fp
        if layer == "qkv_proj":
            out = self.q_local + 2 * self.kv_local
            return 2.0 * t * d * out, (d * out + t * (d + out)) * fp
        if layer == "qk_norm":
            n = self.q_local + self.kv_local
            return 5.0 * t * n, 2 * t * n * fp
        if layer == "rotary_emb":
            n = self.q_local + self.kv_local
            return 6.0 * t * n, 2 * t * n * fp
        if layer == "o_proj":
            return 2.0 * t * self.q_local * d, (self.q_local * d + t * (self.q_local + d)) * fp
        raise KeyError(layer)

    # ---- per-sequence (sequences) ----
    def per_sequence(self, layer: str, s: int) -> Tuple[float, float]:
        if layer == "lm_head":
            return 2.0 * s * self.d * self.vocab_local, (self.d * self.vocab_local + s * (self.d + self.vocab_local)) * self.fp
        if layer == "sampler":
            return 2.0 * s * self.vocab, s * self.vocab * self.fp
        raise KeyError(layer)

    # ---- attention (prefill_chunk, kv_prefill, n_decode, kv_decode) ----
    def attention(self, pc: int, kvp: int, nd: int, kvd: int) -> Tuple[float, float]:
        hd = self.hd
        if self.attention_kind == "gqa":
            kv_bytes_per_tok = 2 * self.kv_local * self.fp
            prefill_pairs = pc * (kvp + pc / 2.0)
            decode_pairs = nd * kvd
            flops = 4.0 * self.heads_local * hd * (prefill_pairs + decode_pairs)
            nbytes = kv_bytes_per_tok * ((kvp + pc) * (1 if pc else 0) + nd * kvd) \
                + (pc + nd) * self.q_local * self.fp * 2
            return flops, nbytes
        # MLA: absorbed latent of kv_lora_rank + rope part is read per token; heads share it
        latent = (self.kv_lora_rank + self.rope_dim) * self.fp
        prefill_pairs = pc * (kvp + pc / 2.0)
        decode_pairs = nd * kvd
        flops = 4.0 * self.heads_local * (self.kv_lora_rank + self.rope_dim) * (prefill_pairs + decode_pairs)
        nbytes = latent * ((kvp + pc) * (1 if pc else 0) + nd * kvd) + (pc + nd) * self.q_local * self.fp * 2
        return flops, nbytes

    # ---- MoE (local tokens, activated experts), profiled at tp=1 ----
    def moe_block(self, tokens: int, activated: int) -> Tuple[float, float]:
        d, dff, fp = self.d, self.d_ff, self.fp
        flops = 6.0 * d * dff * tokens             # gate, up, down per token-expert assignment
        nbytes = activated * 3 * d * dff * fp + tokens * (d + dff) * 2 * fp
        return flops, nbytes


def _write_csv(path: str, header: List[str], rows: Iterable[Tuple]) -> int:
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
            n += 1
    return n


def _grid_from(bundle_tp1: str):
    """Reuse a measured bundle's attention and MoE operating points."""
    att, moe = [], []
    p = os.path.join(bundle_tp1, "attention.csv")
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                att.append((int(r["prefill_chunk"]), int(r["kv_prefill"]), int(r["n_decode"]), int(r["kv_decode"])))
    p = os.path.join(bundle_tp1, "moe.csv")
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                moe.append((int(r["tokens"]), int(r["activated_experts"])))
    return att, moe


def _default_attention_grid():
    pts = set()
    for pc in ATTN_PC:
        for kvp in ATTN_KV:
            for nd in ATTN_ND:
                for kvd in ATTN_KV:
                    if pc == 0 and nd == 0:
                        continue
                    if pc == 0 and kvp != 0:
                        continue
                    if nd == 0 and kvd != 0:
                        continue
                    pts.add((pc, kvp, nd, kvd))
    return sorted(pts)


def synthesize(hw, model, tps: List[int], perf_root: str, variant: str = "bf16",
               grid_from: str = None) -> str:
    rl = Roofline(hw.data)
    hardware = hw["frontend_hardware_label"]
    model_name = model["checkpoint"]
    root = os.path.join(perf_root, hardware, model_name, variant)
    os.makedirs(root, exist_ok=True)
    att_grid, moe_grid = _grid_from(grid_from) if grid_from else ([], [])
    if not att_grid:
        att_grid = _default_attention_grid()
    if not moe_grid:
        moe_grid = [(t, a) for t in MOE_TOKENS for a in MOE_ACTIVATED]
    counts = {}
    tp_stable_ref = Shapes(model.data, 1)
    for tp in sorted(set(tps) | {1}):
        sh = Shapes(model.data, tp)
        d = os.path.join(root, f"tp{tp}")
        os.makedirs(d, exist_ok=True)
        rows = []
        for layer in DENSE_LAYERS:
            src = tp_stable_ref if layer in TP_STABLE else sh
            for t in DENSE_TOKENS:
                rows.append((layer, t, f"{rl.us(*src.dense(layer, t)):.4f}"))
        counts[f"tp{tp}/dense"] = _write_csv(os.path.join(d, "dense.csv"), ["layer", "tokens", "time_us"], rows)
        rows = []
        for layer in ("lm_head", "sampler"):
            src = tp_stable_ref if layer in TP_STABLE else sh
            for s in PER_SEQUENCE:
                rows.append((layer, s, f"{rl.us(*src.per_sequence(layer, s)):.4f}"))
        counts[f"tp{tp}/per_sequence"] = _write_csv(os.path.join(d, "per_sequence.csv"),
                                                    ["layer", "sequences", "time_us"], rows)
        rows = [(pc, kvp, nd, kvd, f"{rl.us(*sh.attention(pc, kvp, nd, kvd)):.4f}") for (pc, kvp, nd, kvd) in att_grid]
        counts[f"tp{tp}/attention"] = _write_csv(os.path.join(d, "attention.csv"),
                                                 ["prefill_chunk", "kv_prefill", "n_decode", "kv_decode", "time_us"], rows)
        if sh.moe and tp == 1:
            rows = [(t, a, f"{rl.us(*sh.moe_block(t, a)):.4f}") for (t, a) in moe_grid if a <= sh.experts]
            counts["tp1/moe"] = _write_csv(os.path.join(d, "moe.csv"), ["tokens", "activated_experts", "time_us"], rows)
    meta = {
        "profiler_version": "synthesized-1.0",
        "provenance": hw["provenance"],
        "provenance_note": ("Every time_us in this bundle is computed from a roofline, not measured. "
                            "Report it as analytically estimated."),
        "roofline": {"formula": hw["compute"].get("model"),
                     "peak_dense_tflops": hw["compute"]["peak_dense_tflops"],
                     "achieved_fraction": hw["compute"].get("achieved_fraction", 1.0),
                     "kernel_floor_us": hw["compute"]["kernel_floor_us"],
                     "hbm_bandwidth_gbps": hw["hbm_bandwidth_gbps"]},
        "hardware_spec": {"name": hw.name, "sha256": hw.sha256, "source": hw.source},
        "model_spec": {"name": model.name, "sha256": model.sha256, "source": model.source},
        "hardware": hardware,
        "model": model_name,
        "variant": variant,
        "architecture": "qwen3_moe" if model.get("moe") else "qwen3",
        "tp_degrees": sorted(set(tps) | {1}),
        "engine_effective": {"max_num_seqs": max(PER_SEQUENCE), "max_num_batched_tokens": max(DENSE_TOKENS),
                             "tensor_parallel_size": 1, "dtype": model["torch_dtype"]},
        "attention_grid": {"source": grid_from or "default cross product", "points": len(att_grid)},
        "skew_fit": {"enabled": False, "note": "no skew sweep for synthesized bundles; alpha = 0"},
        "timings_include": hw["timings_include"],
    }
    with open(os.path.join(root, "meta.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    counts["root"] = root
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware-spec", required=True)
    ap.add_argument("--model-spec", required=True)
    ap.add_argument("--tp", type=int, action="append", required=True)
    ap.add_argument("--variant", default="bf16")
    ap.add_argument("--perf-root", default="profiler/perf")
    ap.add_argument("--grid-from", default=None, help="a measured tp1 bundle dir whose attention/moe points to reuse")
    args = ap.parse_args(argv)
    hw = load_spec("hardware", args.hardware_spec)
    model = load_spec("model", args.model_spec)
    counts = synthesize(hw, model, args.tp, args.perf_root, args.variant, args.grid_from)
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
