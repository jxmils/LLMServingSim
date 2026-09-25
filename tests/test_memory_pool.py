import json
import os
import shutil

import pytest

from serving.core import specs
from serving.core.panel_backend import FabricSpec, pool_configuration_path
from serving.tools.compose_fabric import compose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, "configs", "specs", "memory_pool", "pool-a-custom4.json")
BASE = os.path.join(ROOT, "configs", "fabric", "custom4_shared.json")


def test_memory_pool_spec_loads_and_validates(tmp_path):
    s = specs.load_spec("memory_pool", POOL)
    assert s["devices"][0]["bank_service_gbps"] == 400
    bad = json.load(open(POOL))
    bad["devices"][0]["attachments"] = []
    p = tmp_path / "bad.json"
    json.dump(bad, open(p, "w"))
    with pytest.raises(ValueError, match="attachment"):
        specs.load_spec("memory_pool", str(p))
    bad["devices"][0]["attachments"] = [{"via": "switch:4", "link_gbps": 100}]
    bad["access_mode"] = "direct"
    json.dump(bad, open(p, "w"))
    with pytest.raises(ValueError, match="staging"):
        specs.load_spec("memory_pool", str(p))


def test_compose_appends_endpoint_and_bank_past_highest_id(tmp_path):
    shutil.copy(BASE, tmp_path / "base.json")
    shutil.copy(os.path.join(ROOT, "configs", "fabric", "custom4_shared.graph"), tmp_path / "custom4_shared.graph")
    out = tmp_path / "composed.json"
    r = compose(str(tmp_path / "base.json"), POOL, str(out))
    assert r["pools"][0]["endpoint"] == 6 and r["pools"][0]["bank"] == 7   # base ids 0..5
    lines = [ln.split() for ln in open(tmp_path / "composed.graph") if ln.startswith("E ")]
    edges = {(int(a), int(b)): rest for _, a, b, *rest in lines}
    assert (4, 6) in edges and (6, 4) in edges          # attachment via switch 4
    assert (6, 7) in edges and (7, 6) in edges          # bank behind the endpoint
    assert abs(float(edges[(6, 7)][0]) - 400e9 / 2**30) < 1e-6   # bank rate in GiB/s
    assert float(edges[(6, 7)][1]) == 300.0             # controller latency ns
    assert abs(float(edges[(4, 6)][0]) - 100e9 / 2**30) < 1e-6
    spec = json.load(open(out))
    assert spec["graph"] == "composed.graph" and spec["memory_pool_spec"].endswith("pool-a-custom4.json")
    assert spec["maxwin"] == 2097152 and spec["q"] == 90000   # large transfers under -nocc
    cfg = json.load(open(tmp_path / "composed.pool.json"))
    assert cfg["tensor_loc_pool"] == {"CXL": "pool0"} and cfg["pools"][0]["capacity_bytes"] == 512 * 2**30
    # the composed fabric loads as a FabricSpec and names its backend pool config
    fs = FabricSpec.load(str(out))
    assert fs.nodes == 4
    assert pool_configuration_path(fs) == str(tmp_path / "composed.pool.json")
    assert "-graph" in fs.htsim_opts() and "memory_pool_spec" not in " ".join(fs.htsim_opts())


def test_fabric_without_pool_has_no_pool_configuration():
    fs = FabricSpec.load(BASE)
    assert pool_configuration_path(fs) is None


def test_cxl_tier_accepted_only_with_a_pool(tmp_path):
    import json
    from serving.core.panel_backend import translate_memory_config
    mem = {"remote_mem": {"memory-type": "PER_NODE_MEMORY_EXPANSION", "mem-bw": 256, "mem-latency": 0, "num-devices": 1},
           "cxl_mem": {"memory-type": "MEMORY_POOL", "mem-bw": 100, "mem-latency": 300, "mem-size": 512 * 10**9, "num-devices": 1}}
    src = tmp_path / "memory_expansion.json"
    json.dump(mem, open(src, "w"))
    import pytest
    with pytest.raises(ValueError, match="memory pool"):
        translate_memory_config(str(src), 1, 4)
    pool = os.path.join(ROOT, "configs", "fabric", "custom4_shared_pool_a.pool.json")
    out = translate_memory_config(str(src), 1, 4, pool_config=pool)
    assert json.load(open(out))["remote-mem-bw"] == 256
