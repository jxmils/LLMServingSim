"""Versioned, independent specifications for a serving run (plan §3).

Six kinds — model, hardware, fabric, memory_pool, placement, serving_policy —
each a JSON file with `spec_version` and `name`, validated on load against a
fixed key set (unknown keys are refused, never dropped: a misspelt key would
silently change an experiment), and written back *resolved* into the run's
inputs root with the SHA-256 of every source file. A spec never implies
another: a HardwareSpec carries no fabric, a PlacementSpec's rank count is
checked against the FabricSpec's node count rather than assumed.

Units are part of key names (`*_gib`, `*_gbps`, `*_ns`, `*_bytes`), so a
value is never read under a different unit than it was written in.

`FabricSpec` keeps its own loader in `panel_backend.py` (it renders the
backend's flags); this module registers it so a run's resolved bundle lists
all six kinds in one place.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SPEC_VERSION = 1

# Per kind: required keys and the full allowed key set. `description`,
# `source` and `notes` are documentation and allowed everywhere.
_COMMON = {"spec_version", "name", "description", "source", "notes"}

_SCHEMA: Dict[str, Dict[str, set]] = {
    "model": {
        "required": {"checkpoint", "config_sha256", "hidden_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "head_dim",
                     "intermediate_size", "vocab_size", "torch_dtype", "attention"},
        "allowed": {"checkpoint", "config_sha256", "hidden_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "head_dim",
                    "intermediate_size", "vocab_size", "torch_dtype", "attention",
                    "moe", "tie_word_embeddings", "max_position_embeddings",
                    "frontend_model_config"},
    },
    "hardware": {
        "required": {"provenance", "compute", "hbm_capacity_gib", "hbm_bandwidth_gbps",
                     "precisions", "timings_include"},
        "allowed": {"provenance", "compute", "hbm_capacity_gib", "hbm_bandwidth_gbps",
                    "precisions", "timings_include", "execution_resources", "dma",
                    "frontend_hardware_label"},
    },
    "fabric": {  # validated by panel_backend.FabricSpec; registered here only
        "required": {"nodes"},
        "allowed": None,
    },
    "memory_pool": {
        "required": {"devices"},
        "allowed": {"devices", "access_mode", "allocation_granularity_bytes"},
    },
    "placement": {
        "required": {"num_ranks", "instances"},
        "allowed": {"num_ranks", "instances", "expert_owner", "kv_objects", "placement_epoch",
                    "rank_permutation"},
    },
    "serving_policy": {
        "required": {"admission", "batching", "reservation"},
        "allowed": {"admission", "batching", "prefill_decode", "reservation", "eviction",
                    "prefetch", "network_aware"},
    },
}

PROVENANCE_LABELS = ("measured", "externally_modeled", "analytically_estimated")


@dataclass(frozen=True)
class Spec:
    kind: str
    name: str
    data: Dict[str, Any]
    source: str
    sha256: str

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_spec(kind: str, path: str) -> Spec:
    if kind not in _SCHEMA:
        raise ValueError(f"unknown spec kind {kind!r}; known: {sorted(_SCHEMA)}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a spec is a JSON object")
    if raw.get("spec_version") != SPEC_VERSION:
        raise ValueError(f"{path}: {kind} spec_version must be {SPEC_VERSION}")
    schema = _SCHEMA[kind]
    missing = sorted(schema["required"] - set(raw))
    if missing:
        raise ValueError(f"{path}: {kind} spec is missing {missing}")
    if schema["allowed"] is not None:
        unknown = sorted(set(raw) - schema["allowed"] - _COMMON)
        if unknown:
            raise ValueError(f"{path}: {kind} spec has unknown keys {unknown}")
    if kind == "hardware":
        _check_hardware(raw, path)
    if kind == "model":
        _check_model(raw, path)
    return Spec(kind=kind, name=str(raw.get("name", os.path.basename(path))),
                data=raw, source=os.path.abspath(path), sha256=_sha256(path))


def _check_hardware(raw: dict, path: str) -> None:
    label = raw.get("provenance")
    if label not in PROVENANCE_LABELS:
        raise ValueError(f"{path}: hardware provenance must be one of {PROVENANCE_LABELS}, "
                         f"got {label!r}")
    comp = raw["compute"]
    for key in ("peak_dense_tflops", "kernel_floor_us"):
        if key not in comp:
            raise ValueError(f"{path}: hardware.compute needs {key}")
    inc = raw["timings_include"]
    for key in ("communication", "dma", "offload"):
        if not isinstance(inc.get(key), bool):
            raise ValueError(f"{path}: hardware.timings_include.{key} must be true/false "
                             "(the backend must not charge these twice)")


def _check_model(raw: dict, path: str) -> None:
    att = raw["attention"]
    kind = att.get("kind")
    if kind not in ("gqa", "mla"):
        raise ValueError(f"{path}: model.attention.kind must be 'gqa' or 'mla'")
    if kind == "mla":
        for key in ("kv_lora_rank", "qk_rope_head_dim"):
            if key not in att:
                raise ValueError(f"{path}: MLA attention needs {key}")
    moe = raw.get("moe")
    if moe is not None:
        for key in ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "moe_layers"):
            if key not in moe:
                raise ValueError(f"{path}: model.moe needs {key}")
        layers = moe["moe_layers"]
        if not isinstance(layers, list) or not all(isinstance(i, int) for i in layers):
            raise ValueError(f"{path}: model.moe.moe_layers must list the MoE layer indices explicitly")
        if any(i < 0 or i >= raw["num_hidden_layers"] for i in layers):
            raise ValueError(f"{path}: model.moe.moe_layers outside 0..num_hidden_layers-1")


def cross_check(specs: Dict[str, Spec]) -> List[str]:
    """Consistency rules between kinds. Returns a list of violations."""
    problems = []
    fabric, placement = specs.get("fabric"), specs.get("placement")
    if fabric is not None and placement is not None:
        if int(placement["num_ranks"]) != int(fabric["nodes"]):
            problems.append(f"placement num_ranks {placement['num_ranks']} != fabric nodes {fabric['nodes']}")
    model, placement = specs.get("model"), specs.get("placement")
    if model is not None and placement is not None and model.get("moe"):
        owners = placement.get("expert_owner")
        if owners is not None and len(owners) != int(model["moe"]["num_experts"]):
            problems.append(f"placement expert_owner has {len(owners)} entries for "
                            f"{model['moe']['num_experts']} experts")
    return problems


def emit_resolved(specs: Dict[str, Spec], out_dir: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """Write the fully resolved specs of a run, with source hashes."""
    os.makedirs(out_dir, exist_ok=True)
    bundle = {
        "resolved_spec_version": SPEC_VERSION,
        "specs": {k: {"name": s.name, "source": s.source, "sha256": s.sha256, "data": s.data}
                  for k, s in sorted(specs.items())},
        "cross_check": cross_check(specs),
    }
    if extra:
        bundle["run"] = extra
    path = os.path.join(out_dir, "resolved_specs.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, sort_keys=True)
    return path
