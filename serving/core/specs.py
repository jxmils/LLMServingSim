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
                    "frontend_hardware_label", "workspace_gib"},
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
    if kind == "memory_pool":
        _check_memory_pool(raw, path)
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
    # workspace_gib: the per-rank bytes held outside weights and KV (activation
    # peak, CUDA context, kernel workspaces). When present it replaces the
    # utilization fraction: KV = hbm - weights - workspace (G5 design 2.3).
    if "workspace_gib" in raw:
        ws = raw["workspace_gib"]
        if isinstance(ws, bool) or not isinstance(ws, (int, float)) or ws < 0:
            raise ValueError(f"{path}: hardware.workspace_gib must be a non-negative number of GiB")
        if ws >= float(raw["hbm_capacity_gib"]):
            raise ValueError(f"{path}: hardware.workspace_gib ({ws}) must be below hbm_capacity_gib "
                             f"({raw['hbm_capacity_gib']})")


def _check_memory_pool(raw: dict, path: str) -> None:
    """A pool is a physical endpoint with a capacity model and an access-cost
    model (plan §6): capacity, bank service rate, controller latency and at
    least one attachment to the fabric with its own link rate."""
    mode = raw.get("access_mode", "staging")
    if mode != "staging":
        raise ValueError(f"{path}: memory_pool.access_mode {mode!r} is not implemented (staging only)")
    devices = raw["devices"]
    if not isinstance(devices, list) or not devices:
        raise ValueError(f"{path}: memory_pool.devices must be a non-empty list")
    seen = set()
    for dev in devices:
        for key in ("id", "capacity_gib", "bank_service_gbps", "attachments"):
            if key not in dev:
                raise ValueError(f"{path}: memory_pool device needs {key}")
        if dev["id"] in seen:
            raise ValueError(f"{path}: duplicate memory_pool device id {dev['id']!r}")
        seen.add(dev["id"])
        if float(dev["capacity_gib"]) <= 0 or float(dev["bank_service_gbps"]) <= 0:
            raise ValueError(f"{path}: memory_pool device {dev['id']} needs positive capacity and bank rate")
        if not dev["attachments"]:
            raise ValueError(f"{path}: memory_pool device {dev['id']} needs at least one attachment")
        for att in dev["attachments"]:
            if "via" not in att or "link_gbps" not in att:
                raise ValueError(f"{path}: memory_pool attachment needs via and link_gbps")


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


RESERVATION_POLICIES = ("early", "just_in_time", "transfer_time")


def serving_policy_overrides(spec: Spec) -> Dict[str, Any]:
    """Frontend settings a ServingPolicySpec dictates (G6).

    reservation.policy: 'early' reserves the whole sequence's KV at admission
    (vLLM's reserve_full_isl); 'just_in_time' grows the reservation block by
    block as tokens are computed; 'transfer_time' reserves a staged block when
    its transfer is issued -- in this frontend a recall is issued at admission
    of the resuming/hitting request, so capacity-wise it behaves like
    just_in_time and is labelled as such in the resolved specs.
    admission.reserve_full_isl, when present, must agree.
    """
    reservation = spec.get("reservation", {}) or {}
    policy = reservation.get("policy", "early")
    if policy not in RESERVATION_POLICIES:
        raise ValueError(f"{spec.source}: reservation.policy must be one of {RESERVATION_POLICIES}, got {policy!r}")
    reserve_full = policy == "early"
    admission = spec.get("admission", {}) or {}
    if "reserve_full_isl" in admission and bool(admission["reserve_full_isl"]) != reserve_full:
        raise ValueError(f"{spec.source}: admission.reserve_full_isl={admission['reserve_full_isl']} contradicts "
                         f"reservation.policy={policy!r}")
    out = {"reserve_full_isl": reserve_full, "reservation_policy": policy}
    for key in ("max_num_seqs", "max_num_batched_tokens"):
        if key in admission and admission[key] is not None:
            out[key] = int(admission[key])
    return out


def cross_check(specs: Dict[str, Spec]) -> List[str]:
    """Consistency rules between kinds. Returns a list of violations."""
    problems = []
    fabric, placement = specs.get("fabric"), specs.get("placement")
    if fabric is not None and placement is not None:
        if int(placement["num_ranks"]) != int(fabric["nodes"]):
            problems.append(f"placement num_ranks {placement['num_ranks']} != fabric nodes {fabric['nodes']}")
    model = specs.get("model")
    if model is not None and model.get("frontend_model_config"):
        # The frontend executes configs/model/<name>.json; the spec was made
        # from the exact HF file. They must be the same bytes.
        rel = model["frontend_model_config"]
        found = None
        base = os.path.dirname(model.source)
        for _ in range(6):
            cand = os.path.join(base, rel)
            if os.path.isfile(cand):
                found = cand
                break
            base = os.path.dirname(base)
        if found is None:
            problems.append(f"model frontend_model_config {rel} not found near {model.source}")
        elif _sha256(found) != model["config_sha256"]:
            problems.append(f"model frontend_model_config {rel} sha256 {_sha256(found)[:12]}… differs "
                            f"from the spec's config_sha256 {model['config_sha256'][:12]}…")
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
