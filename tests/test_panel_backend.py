import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from serving.core.panel_backend import (  # noqa: E402
    FabricSpec, build_backend_args, logical_npu_count, resolve_binary,
    translate_memory_config,
)

FRONTEND_MEMORY = {"remote_mem": {"memory-type": "PER_NODE_MEMORY_EXPANSION",
                                  "mem-bw": 256, "mem-latency": 0, "num-devices": 1}}


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
    mem = _write(tmp_path, "memory_expansion.json", FRONTEND_MEMORY)
    sysc = _write(tmp_path, "s.json", {"scheduling-policy": "LIFO",
                                        "all-reduce-implementation": ["ring", "ring"]})
    spec = FabricSpec.load(_write(tmp_path, "f.json", HYBRID))
    args = build_backend_args("/bin/true", spec, "/w/llm", sysc, str(net),
                              mem, start_npu_ids="0", end_npu_ids="63")
    assert args[:4] == ["/bin/true", "--serving", "--chakra-send-admission=serialized",
                        "--chakra-runtime-unit=ns"]
    panel_mem = str(tmp_path / "memory_expansion.panel.json")
    assert "--remote-memory-configuration=" + panel_mem in args
    assert "--system-configuration=" + str(tmp_path / "s.panel.json") in args
    assert json.load(open(tmp_path / "s.panel.json"))["all-reduce-implementation"] == ["ring"]
    assert "--network-configuration=" + str(tmp_path / "network.panel.yml") in args
    assert not any(a.startswith("--memory-configuration=") for a in args)
    with pytest.raises(ValueError, match="chakra_send_admission"):
        build_backend_args("/bin/true", spec, "/w/llm", sysc, str(net), mem,
                           chakra_send_admission="parallel")
    assert args[args.index("--htsim_opts") - 1] == "--end-npu-ids=63"
    assert args[-1] == "-nolog"

    wrong = FabricSpec.load(_write(tmp_path, "g.json", dict(HYBRID, nodes=16)))
    with pytest.raises(ValueError, match="16 nodes but the cluster config resolves to 64"):
        build_backend_args("/bin/true", wrong, "/w/llm", sysc, str(net), mem)
    # a single-NPU topology is refused by the backend's network parser
    one = tmp_path / "net1.yml"
    one.write_text("topology: [FullyConnected]\nnpus_count: [1]\nbandwidth: [16]\nlatency: [500]\n",
                   encoding="utf-8")
    lone = FabricSpec.load(_write(tmp_path, "l.json", {"spec_version": 1, "nodes": 1}))
    with pytest.raises(ValueError, match="at least 2 logical NPUs"):
        build_backend_args("/bin/true", lone, "/w/llm", sysc, str(one), mem)


def test_memory_config_translation(tmp_path):
    src = _write(tmp_path, "memory_expansion.json", FRONTEND_MEMORY)
    out = translate_memory_config(src, num_nodes=1, npus_per_node=8)
    assert out == str(tmp_path / "memory_expansion.panel.json")
    assert json.loads(open(out).read()) == {
        "memory-type": "PER_NODE_MEMORY_EXPANSION", "num-nodes": 1,
        "num-npus-per-node": 8, "remote-mem-latency": 0, "remote-mem-bw": 256,
    }
    none = _write(tmp_path, "none.json", {})
    assert json.loads(open(translate_memory_config(none, 1, 8)).read()) == {
        "memory-type": "NO_MEMORY_EXPANSION"}
    for extra in ("cxl_mem", "local_mem"):
        bad = _write(tmp_path, f"{extra}.json", dict(FRONTEND_MEMORY, **{extra: {"mem-bw": 1}}))
        with pytest.raises(ValueError, match="not supported"):
            translate_memory_config(bad, 1, 8)
    pim = dict(remote_mem=dict(FRONTEND_MEMORY["remote_mem"], **{"pim-channels": 4}))
    with pytest.raises(ValueError, match="PIM"):
        translate_memory_config(_write(tmp_path, "pim.json", pim), 1, 8)


def test_file_keys_resolve_relative_and_env(tmp_path, monkeypatch):
    (tmp_path / "fx").mkdir()
    (tmp_path / "fx" / "t.topo").write_text("Nodes 8\n", encoding="utf-8")
    monkeypatch.setenv("FIXTURES", str(tmp_path / "fx"))
    spec = FabricSpec.load(_write(tmp_path, "f.json", {
        "spec_version": 1, "nodes": 8, "topo": "${FIXTURES}/t.topo"}))
    assert spec.options["topo"] == str(tmp_path / "fx" / "t.topo")
    spec = FabricSpec.load(_write(tmp_path, "g.json", {
        "spec_version": 1, "nodes": 8, "topo": "fx/t.topo"}))
    assert spec.options["topo"] == str(tmp_path / "fx" / "t.topo")
    with pytest.raises(FileNotFoundError, match="topo file not found"):
        FabricSpec.load(_write(tmp_path, "h.json", {
            "spec_version": 1, "nodes": 8, "topo": "missing.topo"}))


def test_et_transcode_remaps_node_types_once(tmp_path):
    from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
    from chakra.src.third_party.utils.protolib import decodeMessage, encodeMessage
    from serving.core.panel_backend import transcode_et_for_panel

    def write(path, types):
        with open(path, "wb") as f:
            encodeMessage(f, GlobalMetadata(version="0.0.4"))
            for i, t in enumerate(types):
                n = Node(id=i, name=f"n{i}")
                n.type = t
                encodeMessage(f, n)

    def read(path):
        with open(path, "rb") as f:
            gm = GlobalMetadata(); decodeMessage(f, gm)
            out = []
            while True:
                n = Node()
                if not decodeMessage(f, n):
                    break
                out.append(n.type)
            return gm, out

    p = str(tmp_path / "llm.0.et")
    write(p, [2, 5, 8, 6, 7, 3])          # MEM_LOAD COMP COLL SEND RECV MEM_STORE (frontend)
    assert transcode_et_for_panel(p) is True
    gm, types = read(p)
    assert types == [2, 4, 7, 5, 6, 3]     # panel numbering
    assert any(a.name == "panel_backend_schema" for a in gm.attr)
    assert transcode_et_for_panel(p) is False   # tagged: left alone
    assert read(p)[1] == [2, 4, 7, 5, 6, 3]

    pim = str(tmp_path / "llm.1.et")
    write(pim, [2, 4, 3])                  # PIM_COMP_NODE in the frontend schema
    with pytest.raises(ValueError, match="PIM_COMP_NODE"):
        transcode_et_for_panel(pim)


def test_resolve_binary_requires_a_path(monkeypatch):
    monkeypatch.delenv("PANEL_ASTRA_HTSIM", raising=False)
    with pytest.raises(FileNotFoundError, match="PANEL_ASTRA_HTSIM"):
        resolve_binary(None)
    monkeypatch.setenv("PANEL_ASTRA_HTSIM", sys.executable)
    assert resolve_binary(None) == sys.executable
    assert resolve_binary(sys.executable) == sys.executable
    with pytest.raises(FileNotFoundError, match="not executable"):
        resolve_binary(os.devnull)


def test_network_config_is_flattened_to_one_dimension(tmp_path):
    import yaml
    from serving.core.panel_backend import flatten_network_config, logical_npu_count
    src = tmp_path / "network.yml"
    src.write_text("topology: [FullyConnected, FullyConnected]\nnpus_count: [1, 3]\nbandwidth: [16.0, 16.0]\nlatency: [20000.0, 20000.0]\n")
    out = flatten_network_config(str(src))
    flat = yaml.safe_load(open(out))
    assert flat["npus_count"] == [3] and flat["topology"] == ["FullyConnected"]
    assert flat["bandwidth"] == [16.0] and flat["latency"] == [20000.0]
    assert logical_npu_count(out) == 3


def test_system_config_collective_lists_follow_the_flattened_network(tmp_path):
    from serving.core.panel_backend import flatten_system_config
    src = tmp_path / "system.json"
    json.dump({"scheduling-policy": "LIFO", "all-reduce-implementation": ["ring", "ring"],
               "all-gather-implementation": ["ring", "ring"], "local-mem-bw": 50}, open(src, "w"))
    out = json.load(open(flatten_system_config(str(src))))
    assert out["all-reduce-implementation"] == ["ring"] and out["all-gather-implementation"] == ["ring"]
    assert out["local-mem-bw"] == 50 and out["scheduling-policy"] == "LIFO"
