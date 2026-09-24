import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from serving.core.panel_backend import (  # noqa: E402
    FabricSpec, build_backend_args, logical_npu_count, resolve_binary,
)


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


HYBRID = {
    "spec_version": 1, "name": "hybrid-4d2o-n64", "nodes": 64,
    "panel": "hybrid", "linkGiBps": 200, "latencyNs": 1000,
    "policy": "directpref", "ocs": True, "reconfNs": 10,
    "q": 90000, "nocc": True, "maxwin": 2097152, "seed": 1, "nolog": True,
}


def test_htsim_opts_render_order_and_flags(tmp_path):
    spec = FabricSpec.load(_write(tmp_path, "f.json", HYBRID))
    assert spec.htsim_opts() == [
        "--htsim_opts", "-nodes", "64", "-panel", "hybrid", "-linkGiBps", "200",
        "-latencyNs", "1000", "-policy", "directpref", "-seed", "1", "-q", "90000",
        "-maxwin", "2097152", "-reconfNs", "10", "-nocc", "-ocs", "-nolog",
    ]


def test_extents_list_and_extra_passthrough(tmp_path):
    raw = dict(HYBRID, nodes=256, extents=[16, 16], extra=["-planeLatencyNs", "55"])
    spec = FabricSpec.load(_write(tmp_path, "f.json", raw))
    opts = spec.htsim_opts()
    assert opts[opts.index("-extents") + 1] == "16,16"
    assert opts[-2:] == ["-planeLatencyNs", "55"]


def test_unknown_key_is_refused(tmp_path):
    raw = dict(HYBRID, linkGibps=200)  # misspelt
    with pytest.raises(ValueError, match="unknown FabricSpec keys"):
        FabricSpec.load(_write(tmp_path, "f.json", raw))


def test_boolean_flag_must_be_boolean(tmp_path):
    spec = FabricSpec.load(_write(tmp_path, "f.json", dict(HYBRID, nocc=1)))
    with pytest.raises(ValueError, match="boolean flag"):
        spec.htsim_opts()


def test_backend_args_check_rank_count(tmp_path):
    net = tmp_path / "network.yml"
    net.write_text("topology: [FullyConnected, FullyConnected]\nnpus_count: [8, 8]\n"
                   "bandwidth: [200, 200]\nlatency: [500, 500]\n", encoding="utf-8")
    assert logical_npu_count(str(net)) == 64
    spec = FabricSpec.load(_write(tmp_path, "f.json", HYBRID))
    args = build_backend_args("/bin/true", spec, "/w/llm", "/s.json", str(net),
                              "/m.json", start_npu_ids="0", end_npu_ids="63")
    assert args[:3] == ["/bin/true", "--serving", "--chakra-send-admission=serialized"]
    assert "--remote-memory-configuration=/m.json" in args
    with pytest.raises(ValueError, match="chakra_send_admission"):
        build_backend_args("/bin/true", spec, "/w/llm", "/s.json", str(net), "/m.json",
                           chakra_send_admission="parallel")
    assert "--memory-configuration=/m.json" not in args
    assert args[args.index("--htsim_opts") - 1] == "--end-npu-ids=63"
    assert args[-1] == "-nolog"

    wrong = FabricSpec.load(_write(tmp_path, "g.json", dict(HYBRID, nodes=16)))
    with pytest.raises(ValueError, match="16 nodes but the cluster config resolves to 64"):
        build_backend_args("/bin/true", wrong, "/w/llm", "/s.json", str(net), "/m.json")


def test_resolve_binary_requires_a_path(monkeypatch):
    monkeypatch.delenv("PANEL_ASTRA_HTSIM", raising=False)
    with pytest.raises(FileNotFoundError, match="PANEL_ASTRA_HTSIM"):
        resolve_binary(None)
    monkeypatch.setenv("PANEL_ASTRA_HTSIM", sys.executable)
    assert resolve_binary(None) == sys.executable
    assert resolve_binary(sys.executable) == sys.executable
    with pytest.raises(FileNotFoundError, match="not executable"):
        resolve_binary(os.devnull)
