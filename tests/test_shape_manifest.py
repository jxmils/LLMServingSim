"""Shape manifest: the trace generator's profile lookups are recorded per
distinct cell with the way each resolved against the profiled grid."""
import json

from serving.core import shape_manifest
from serving.core import trace_generator as tg


def _perf_db():
    """Minimal bundle: one dense layer, one per-sequence layer, a 2x2x2x2
    attention grid and a MoE table, all at tp=1."""
    attn_rows = []
    for pc in (0, 16):
        for nd in (0, 2):
            for kp in (0, 512):
                for kd in (0, 1024):
                    attn_rows.append((pc, nd, kp, kd, 1000 + pc + nd + kp + kd))
    slices = {}
    for pc, nd, kp, kd, lat in attn_rows:
        slices.setdefault((pc, nd), {}).setdefault(kp, {})[kd] = lat
    attn = {"pc_vals": [0, 16], "nd_vals": [0, 2], "pc_nd_pairs": sorted(slices),
            "slices": {k: {"kv_prefill_vals": sorted(v),
                           "rows": [{"keys": sorted(v[kp]), "values": [v[kp][kd] for kd in sorted(v[kp])]}
                                    for kp in sorted(v)]} for k, v in slices.items()}}
    return {
        "hardware": "TESTHW", "model": "test/model", "variant": "bf16", "root": "/nonexistent",
        "available_tps": [1],
        "architecture": {"catalog": {"dense": {"q_proj": {}}, "per_sequence": {"norm": {}}}},
        "tables": {1: {
            "dense": {"q_proj": {"keys": [16, 32, 64], "values": [10, 20, 40]}},
            "per_sequence": {"norm": {"keys": [1, 8], "values": [5, 40]}},
            "attention": attn,
            "moe": {"activated_experts_vals": [1, 8],
                    "rows": [{"keys": [16, 64], "values": [100, 400]}, {"keys": [16, 64], "values": [200, 800]}]},
        }},
    }


def test_modes_and_counts(tmp_path):
    db = _perf_db()
    m = shape_manifest.enable()
    try:
        assert tg._lookup_dense(db, "q_proj", 1, 32) == 20          # exact
        assert tg._lookup_dense(db, "q_proj", 1, 32) == 20          # same cell again
        assert tg._lookup_dense(db, "q_proj", 1, 48) == 30          # interpolated
        assert tg._lookup_dense(db, "q_proj", 1, 128) == 80         # extrapolated
        assert tg._lookup_dense(db, "q_proj", 1, 4) == 10           # clamped below
        assert tg._lookup_per_sequence(db, "norm", 1, 8) == 40
        tg._lookup_attention(db, 1, 16, 512, 2, 1024)               # exact on all axes
        tg._lookup_attention(db, 1, 8, 256, 1, 512)                 # interpolated
        tg._lookup_attention(db, 1, 32, 512, 2, 4096)               # extrapolated
        tg._lookup_moe(db, 32, 4)                                   # interpolated both axes
        m.bundle(db)
    finally:
        shape_manifest.disable()
    s = m.summary()
    assert s["lookups"] == 10 and s["distinct_cells"] == 9
    dense = {tuple(r["query"].values()): r for r in m.cells.values() if r["category"] == "dense"}
    assert dense[(32,)]["mode"] == "exact" and dense[(32,)]["count"] == 2
    assert dense[(48,)]["mode"] == "interpolated" and dense[(48,)]["axes"]["tokens"] == {"mode": "interpolated", "lo": 32, "hi": 64}
    assert dense[(128,)]["mode"] == "extrapolated"
    assert dense[(4,)]["mode"] == "clamped"
    attn = [r for r in m.cells.values() if r["category"] == "attention"]
    assert [r["mode"] for r in attn] == ["exact", "interpolated", "extrapolated"]
    moe = [r for r in m.cells.values() if r["category"] == "moe"][0]
    assert moe["mode"] == "interpolated" and moe["query"] == {"tokens": 32, "activated_experts": 4}
    assert s["by_category"]["dense"] == {"cells": 4, "lookups": 5, "exact": 1, "interpolated": 1,
                                         "extrapolated": 1, "clamped": 1, "missing": 0,
                                         "lookups_extrapolated": 1}
    out = m.write(str(tmp_path / "run.manifest.json"), header={"run_id": "t"})
    doc = json.load(open(out))
    assert doc["schema_version"] == 1 and doc["header"]["run_id"] == "t"
    assert doc["bundles"][0]["hardware"] == "TESTHW" and doc["bundles"][0]["available_tps"] == [1]
    assert doc["cells"][0]["count"] == 2      # sorted by count, the repeated dense cell first
    assert "9 distinct cells over 10 lookups" in m.one_line()


def test_disabled_is_free():
    db = _perf_db()
    assert shape_manifest.get() is None
    assert tg._lookup_dense(db, "q_proj", 1, 48) == 30
    assert shape_manifest.get() is None
