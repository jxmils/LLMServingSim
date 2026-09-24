import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from serving.core.specs import (  # noqa: E402
    Spec, cross_check, emit_resolved, load_spec,
)

MODEL = REPO_ROOT / "configs/specs/model/qwen3-235b-a22b.json"
B200 = REPO_ROOT / "configs/specs/hardware/b200-like.json"
IRONWOOD = REPO_ROOT / "configs/specs/hardware/ironwood-like.json"


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def test_shipped_specs_load_and_hash():
    m = load_spec("model", str(MODEL))
    assert m.kind == "model" and m.name == "qwen3-235b-a22b" and len(m.sha256) == 64
    assert m["moe"]["moe_layers"] == list(range(94))
    assert m["attention"]["kv_bytes_per_token_per_layer"] == 2048
    # the frontend model config the spec points at is the exact HF config
    cfg = json.load(open(REPO_ROOT / m["frontend_model_config"]))
    assert cfg["moe_intermediate_size"] == m["moe"]["moe_intermediate_size"] == 1536
    assert cfg["num_hidden_layers"] == m["num_hidden_layers"] == 94
    for p in (B200, IRONWOOD):
        h = load_spec("hardware", str(p))
        assert h["provenance"] == "analytically_estimated"
        assert h["timings_include"] == {"communication": False, "dma": False, "offload": False}


def test_unknown_keys_and_missing_required_are_refused(tmp_path):
    good = json.load(open(B200))
    bad = dict(good, hbm_bandwith_gbps=1)  # misspelt
    with pytest.raises(ValueError, match="unknown keys"):
        load_spec("hardware", _write(tmp_path, "h.json", bad))
    missing = {k: v for k, v in good.items() if k != "timings_include"}
    with pytest.raises(ValueError, match="missing"):
        load_spec("hardware", _write(tmp_path, "m.json", missing))
    with pytest.raises(ValueError, match="provenance"):
        load_spec("hardware", _write(tmp_path, "p.json", dict(good, provenance="vendor")))
    with pytest.raises(ValueError, match="spec_version"):
        load_spec("hardware", _write(tmp_path, "v.json", dict(good, spec_version=2)))


def test_model_moe_layers_must_be_explicit(tmp_path):
    good = json.load(open(MODEL))
    bad = json.loads(json.dumps(good)); bad["moe"]["moe_layers"] = "all"
    with pytest.raises(ValueError, match="explicitly"):
        load_spec("model", _write(tmp_path, "m.json", bad))
    mla = json.loads(json.dumps(good)); mla["attention"] = {"kind": "mla"}
    with pytest.raises(ValueError, match="kv_lora_rank"):
        load_spec("model", _write(tmp_path, "mla.json", mla))


def test_cross_check_and_resolved_emission(tmp_path):
    model = load_spec("model", str(MODEL))
    fabric = Spec("fabric", "f", {"spec_version": 1, "nodes": 64}, "/f.json", "0" * 64)
    placement = load_spec("placement", _write(tmp_path, "p.json", {
        "spec_version": 1, "name": "p", "num_ranks": 8,
        "instances": [{"id": 0, "ranks": list(range(8))}],
        "expert_owner": list(range(64))}))
    problems = cross_check({"model": model, "fabric": fabric, "placement": placement})
    assert any("num_ranks 8 != fabric nodes 64" in p for p in problems)
    assert any("64 entries for 128 experts" in p for p in problems)
    out = emit_resolved({"model": model, "fabric": fabric}, str(tmp_path / "specs"),
                        extra={"run_id": "t"})
    bundle = json.load(open(out))
    assert set(bundle["specs"]) == {"fabric", "model"}
    assert bundle["specs"]["model"]["sha256"] == model.sha256
    assert bundle["cross_check"] == []
    assert bundle["run"] == {"run_id": "t"}


def test_cross_check_refuses_edited_frontend_config(tmp_path):
    """The frontend runs configs/model/<name>.json; the spec's hash is of the exact HF file."""
    import json, os, shutil
    from serving.core import specs
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec_src = os.path.join(root, "configs", "specs", "model", "qwen3-235b-a22b.json")
    ok = specs.cross_check({"model": specs.load_spec("model", spec_src)})
    assert ok == []
    # copy the tree shape: <tmp>/configs/specs/model/spec.json + <tmp>/configs/model/Qwen/... edited
    os.makedirs(tmp_path / "configs" / "specs" / "model")
    os.makedirs(tmp_path / "configs" / "model" / "Qwen")
    shutil.copy(spec_src, tmp_path / "configs" / "specs" / "model" / "qwen3-235b-a22b.json")
    cfg = json.load(open(os.path.join(root, "configs", "model", "Qwen", "Qwen3-235B-A22B.json")))
    cfg["num_hidden_layers"] = 93
    json.dump(cfg, open(tmp_path / "configs" / "model" / "Qwen" / "Qwen3-235B-A22B.json", "w"))
    bad = specs.cross_check({"model": specs.load_spec("model", str(tmp_path / "configs" / "specs" / "model" / "qwen3-235b-a22b.json"))})
    assert bad and "differs" in bad[0]


def test_serving_policy_overrides_and_contradiction(tmp_path):
    import json, os
    from serving.core import specs
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    early = specs.load_spec("serving_policy", os.path.join(root, "configs", "specs", "serving_policy", "frontend-default.json"))
    jit = specs.load_spec("serving_policy", os.path.join(root, "configs", "specs", "serving_policy", "just-in-time.json"))
    assert specs.serving_policy_overrides(early)["reserve_full_isl"] is True
    o = specs.serving_policy_overrides(jit)
    assert o["reserve_full_isl"] is False and o["reservation_policy"] == "just_in_time" and o["max_num_seqs"] == 128
    bad = json.load(open(os.path.join(root, "configs", "specs", "serving_policy", "just-in-time.json")))
    bad["admission"]["reserve_full_isl"] = True
    p = tmp_path / "bad.json"; json.dump(bad, open(p, "w"))
    import pytest
    with pytest.raises(ValueError, match="contradicts"):
        specs.serving_policy_overrides(specs.load_spec("serving_policy", str(p)))
