"""Independent byte/shape checks for a model port (plan §4, G4 acceptance).

Computes, from a ModelSpec and a PlacementSpec alone:
  * total parameter bytes and the per-rank share under the placement's TP/EP,
  * KV bytes per token (whole model) and per rank,
  * per-layer operator shapes (q/kv widths per rank, expert width),
  * expert -> rank ownership counts,
  * the TP all-reduce payload per layer for a given batch of tokens
    (hidden * tokens * dtype bytes, twice per layer: o_proj and the MoE/MLP output),
and compares them with what the frontend reports in a run log
("NPU: model weight ... loaded", "KV cache ... blocks") when a log is given,
and with a campaign roster's `logical_tp_allreduce_size_B` when given.

Nothing here is a latency. It exists so a wrong shape or byte count is found
before a number is reported.
"""

import argparse
import json
import re
import sys
from typing import Dict, Optional

from serving.core.specs import load_spec

BYTES = {"bfloat16": 2, "float16": 2, "fp8": 1}


def model_bytes(model) -> Dict[str, float]:
    m = model.data
    d, L, V = m["hidden_size"], m["num_hidden_layers"], m["vocab_size"]
    fp = BYTES[m["torch_dtype"]]
    H, Hkv, hd = m["num_attention_heads"], m["num_key_value_heads"], m["head_dim"]
    q_dim, kv_dim = H * hd, Hkv * hd
    attn_params = d * (q_dim + 2 * kv_dim) + q_dim * d       # qkv_proj + o_proj
    norm_params = 2 * d + (2 * hd if True else 0)             # two RMSNorms (+ qk_norm weights, tiny)
    moe = m.get("moe")
    moe_layers = set(moe["moe_layers"]) if moe else set()
    dense_mlp = 3 * d * m["intermediate_size"]
    expert_params = (3 * d * moe["moe_intermediate_size"]) if moe else 0
    total = V * d * (1 if m.get("tie_word_embeddings") else 2) + d  # embeddings, lm_head, final norm
    per_layer = []
    for layer in range(L):
        p = attn_params + norm_params
        if layer in moe_layers:
            p += moe["num_experts"] * expert_params + d * moe["num_experts"]  # experts + router
        else:
            p += dense_mlp
        per_layer.append(p)
    total += sum(per_layer)
    kv_per_token_per_layer = 2 * kv_dim * fp if m["attention"]["kind"] == "gqa" else \
        (m["attention"]["kv_lora_rank"] + m["attention"]["qk_rope_head_dim"]) * fp
    return {
        "params": total,
        "param_bytes": total * fp,
        "expert_params_each": expert_params,
        "kv_bytes_per_token": kv_per_token_per_layer * L,
        "kv_bytes_per_token_per_layer": kv_per_token_per_layer,
        "q_dim": q_dim, "kv_dim": kv_dim, "dtype_bytes": fp,
        "moe_layers": len(moe_layers), "dense_layers": L - len(moe_layers),
    }


def per_rank(model, placement, inst_idx: int = 0) -> Dict[str, float]:
    m, b = model.data, model_bytes(model)
    inst = placement["instances"][inst_idx]
    tp, ep = int(inst["tp"]), int(inst["ep"])
    fp = b["dtype_bytes"]
    d, L, V = m["hidden_size"], m["num_hidden_layers"], m["vocab_size"]
    Hkv, hd = m["num_key_value_heads"], m["head_dim"]
    kv_local = max(Hkv // tp, 1) * hd
    moe = m.get("moe")
    n_moe = b["moe_layers"]
    # dense parts sharded by TP; experts by EP; embeddings/lm_head by TP
    dense_per_rank = (b["params"] - n_moe * moe["num_experts"] * b["expert_params_each"]) / tp if moe else b["params"] / tp
    experts_per_rank = (moe["num_experts"] / ep) if moe else 0
    expert_bytes_per_rank = n_moe * experts_per_rank * b["expert_params_each"] * fp
    owners = placement.get("expert_owner")
    counts = {}
    if owners:
        for o in owners:
            counts[o] = counts.get(o, 0) + 1
    return {
        "tp": tp, "ep": ep,
        "weight_bytes_per_rank": dense_per_rank * fp + expert_bytes_per_rank,
        "expert_bytes_per_rank": expert_bytes_per_rank,
        "experts_per_rank": experts_per_rank,
        "expert_owner_counts": counts,
        "kv_bytes_per_token_per_rank": 2 * kv_local * fp * L if m["attention"]["kind"] == "gqa" else b["kv_bytes_per_token"],
        "kv_replication": max(1, tp // max(Hkv, 1)) if m["attention"]["kind"] == "gqa" else 1,
    }


def tp_allreduce_bytes(model, tokens: int) -> int:
    m = model.data
    return tokens * m["hidden_size"] * BYTES[m["torch_dtype"]]


def read_frontend_log(path: str) -> Dict[str, Optional[float]]:
    weight_mb = kv_blocks = kv_tokens = kv_mb = None
    with open(path, errors="replace") as f:
        for line in f:
            mw = re.search(r"model weight ([0-9.]+)\s*MB", line)
            if mw:
                weight_mb = float(mw.group(1))
            mk = re.search(r"KV cache (\d+) blocks \((\d+) tokens, ([0-9.]+)MB\)", line)
            if mk:
                kv_blocks, kv_tokens, kv_mb = int(mk.group(1)), int(mk.group(2)), float(mk.group(3))
    return {"weight_mb": weight_mb, "kv_blocks": kv_blocks, "kv_tokens": kv_tokens, "kv_mb": kv_mb}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-spec", required=True)
    ap.add_argument("--placement-spec", required=True)
    ap.add_argument("--frontend-log", default=None)
    ap.add_argument("--roster", default=None, help="campaign moe_inference_workloads_backend_v2.json")
    ap.add_argument("--roster-model", default="qwen3")
    ap.add_argument("--tokens", type=int, default=1024, help="batch tokens for the TP all-reduce payload check")
    args = ap.parse_args(argv)
    model = load_spec("model", args.model_spec)
    placement = load_spec("placement", args.placement_spec)
    b = model_bytes(model)
    r = per_rank(model, placement)
    out = {"model": model.name, "placement": placement.name, "model_totals": b, "per_rank": r,
           "tp_allreduce_bytes_per_layer_at_tokens": {str(args.tokens): tp_allreduce_bytes(model, args.tokens)},
           "checks": []}
    if args.frontend_log:
        fl = read_frontend_log(args.frontend_log)
        out["frontend_log"] = fl
        if fl["weight_mb"] is not None:
            expected_mb = r["weight_bytes_per_rank"] / 1e6
            rel = abs(fl["weight_mb"] - expected_mb) / expected_mb
            out["checks"].append({"weight_bytes_per_rank": {"expected_MB": round(expected_mb, 1),
                                  "frontend_MB": fl["weight_mb"], "rel_err": round(rel, 4),
                                  "pass": rel < 0.02}})
        if fl["kv_tokens"] and fl["kv_mb"]:
            per_tok = fl["kv_mb"] * 1e6 / fl["kv_tokens"]
            expected = r["kv_bytes_per_token_per_rank"]
            rel = abs(per_tok - expected) / expected
            out["checks"].append({"kv_bytes_per_token_per_rank": {"expected": expected,
                                  "frontend": round(per_tok, 1), "rel_err": round(rel, 4), "pass": rel < 0.02}})
    if args.roster:
        roster = json.load(open(args.roster))
        entry = next((m for m in roster["models"] if m.get("model_id") == args.roster_model), None)
        if entry:
            sizes = entry.get("logical_tp_allreduce_size_B", {})
            out["roster_tp_allreduce_size_B"] = sizes
            for key, val in sizes.items():
                try:
                    toks = int(re.sub(r"[^0-9]", "", key)) if re.search(r"\d", key) else None
                except ValueError:
                    toks = None
                if toks:
                    out["checks"].append({f"tp_allreduce_{key}": {"roster_B": val,
                                          "spec_B": tp_allreduce_bytes(model, toks),
                                          "pass": int(val) == tp_allreduce_bytes(model, toks)}})
    print(json.dumps(out, indent=2))
    failed = [c for c in out["checks"] if not list(c.values())[0].get("pass", True)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
