"""G5 design 2.3: an explicit workspace replaces the utilization fraction in
the KV budget, in the memory model and in the HardwareSpec."""
import json

import pytest

from serving.core import specs
from serving.core.memory_model import MemoryModel
from serving.core.memory_model import GB_TO_BYTE

MODEL = "meta-llama/Llama-3.1-8B"


def _mm(**kw):
    args = dict(model=MODEL, instance_id=0, node_id=0, num_npus=1, tp_size=1, npu_mem=96, cpu_mem=1,
                block_size=16, fp=16, enable_prefix_caching=False, enable_prefix_sharing=False,
                prefix_pool=None, prefix_storage=None)
    args.update(kw)
    return MemoryModel(**args)


def test_workspace_budget_is_exact_and_ignores_util():
    base = _mm(npu_memory_utilization=0.9)
    ws = _mm(npu_memory_utilization=0.5, npu_workspace_bytes=int(8 * GB_TO_BYTE))
    hbm = 96 * GB_TO_BYTE
    assert ws.npu_pool.num_blocks == (hbm - ws.weight - 8 * GB_TO_BYTE) // ws._npu_bytes_per_block
    assert base.npu_pool.num_blocks == (int(hbm * 0.9) - base.weight) // base._npu_bytes_per_block
    assert ws.npu_pool.num_blocks > base.npu_pool.num_blocks     # 8 GiB reserved < 9.6 GiB headroom
    assert ws.budget_label().startswith("with workspace 8.00 GiB")
    assert base.budget_label() == "at util 0.90"


def test_workspace_too_large_is_refused():
    with pytest.raises(RuntimeError, match="workspace"):
        _mm(npu_workspace_bytes=int(95 * GB_TO_BYTE))


def test_hardware_spec_workspace_validation(tmp_path):
    spec = {
        "spec_version": 1, "name": "t", "provenance": "analytically_estimated",
        "compute": {"peak_dense_tflops": 1000, "kernel_floor_us": 5.0},
        "hbm_capacity_gib": 74.5, "hbm_bandwidth_gbps": 3350, "precisions": ["bf16"],
        "timings_include": {"communication": False, "dma": False, "offload": False},
        "workspace_gib": 6.5,
    }
    p = tmp_path / "hw.json"
    p.write_text(json.dumps(spec))
    assert specs.load_spec("hardware", str(p)).data["workspace_gib"] == 6.5
    for bad in (-1, 80, "6", True):
        spec["workspace_gib"] = bad
        p.write_text(json.dumps(spec))
        with pytest.raises(ValueError, match="workspace_gib"):
            specs.load_spec("hardware", str(p))
