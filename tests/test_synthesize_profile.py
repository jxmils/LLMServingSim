import csv
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from serving.core.specs import load_spec  # noqa: E402
from serving.tools.synthesize_profile import Roofline, Shapes, synthesize  # noqa: E402

MODEL = REPO_ROOT / "configs/specs/model/qwen3-235b-a22b.json"
B200 = REPO_ROOT / "configs/specs/hardware/b200-like.json"


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_roofline_and_shapes_hand_values():
    hw = load_spec("hardware", str(B200))
    model = load_spec("model", str(MODEL))
    rl = Roofline(hw.data)
    sh = Shapes(model.data, 1)
    # MoE block, 1 token, 8 activated experts: weight-bandwidth bound.
    # weights = 8 * 3 * 4096 * 1536 * 2 B = 302,  a bit; / 8 TB/s = 37.75 us; + 5 us floor
    flops, nbytes = sh.moe_block(1, 8)
    assert nbytes == 8 * 3 * 4096 * 1536 * 2 + 1 * (4096 + 1536) * 2 * 2
    assert abs(rl.us(flops, nbytes) - (nbytes / 8e12 * 1e6 + 5.0)) < 1e-6
    # qkv_proj at TP64: q_local = 64*128/64 = 128, kv_local = max(4//64,1)*128 = 128
    sh64 = Shapes(model.data, 64)
    assert sh64.q_local == 128 and sh64.kv_local == 128
    flops, _ = sh64.dense("qkv_proj", 2048)
    assert flops == 2.0 * 2048 * 4096 * (128 + 2 * 128)
    # KV bytes per token per layer from the spec matches the GQA shape
    assert model["attention"]["kv_bytes_per_token_per_layer"] == 2 * 4 * 128 * 2


def test_synthesized_bundle_layout_and_monotonicity(tmp_path):
    hw = load_spec("hardware", str(B200))
    model = load_spec("model", str(MODEL))
    counts = synthesize(hw, model, [8], str(tmp_path / "perf"))
    root = Path(counts["root"])
    assert root == tmp_path / "perf/B200LIKE/Qwen/Qwen3-235B-A22B/bf16"
    assert (root / "tp1/moe.csv").exists() and not (root / "tp8/moe.csv").exists()
    meta = yaml.safe_load(open(root / "meta.yaml"))
    assert meta["provenance"] == "analytically_estimated"
    assert meta["skew_fit"]["enabled"] is False
    assert meta["tp_degrees"] == [1, 8]
    assert meta["hardware_spec"]["sha256"] == hw.sha256
    dense = _rows(root / "tp8/dense.csv")
    qkv = [(int(r["tokens"]), float(r["time_us"])) for r in dense if r["layer"] == "qkv_proj"]
    assert qkv == sorted(qkv) and all(b[1] >= a[1] for a, b in zip(qkv, qkv[1:]))
    # tp-stable layers are identical across tp folders
    d1 = {(r["layer"], r["tokens"]): r["time_us"] for r in _rows(root / "tp1/dense.csv") if r["layer"] == "layernorm"}
    d8 = {(r["layer"], r["tokens"]): r["time_us"] for r in dense if r["layer"] == "layernorm"}
    assert d1 == d8
    att = _rows(root / "tp8/attention.csv")
    assert {"prefill_chunk", "kv_prefill", "n_decode", "kv_decode", "time_us"} <= set(att[0])
    moe = _rows(root / "tp1/moe.csv")
    assert all(int(r["activated_experts"]) <= 128 for r in moe)
