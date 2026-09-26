"""Launch arguments for the panel-scale HTSim backend (`--network-backend htsim`).

The backend is the `AstraSim_HTSim` binary from Panel-Scale-Systems'
astra-sim fork, run in its serving mode. It speaks the same stdin/stdout
protocol as the analytical backend (see docs/docs/simulator/architecture.mdx),
so the Controller is unchanged; what differs is how the process is started:

* the remote-memory flag is `--remote-memory-configuration` (upstream
  ASTRA-Sim naming), not `--memory-configuration`;
* the physical fabric is described by a FabricSpec (JSON) rendered into the
  `--htsim_opts ...` tail that the backend's protocol implementation parses;
* the packet layer reads only `npus_count` from network.yml, so the fabric's
  node count must equal the product of the logical dimensions, or ranks
  silently run on a fabric of the wrong size. That is a hard error here.

Nothing in this module models time. Bandwidth and latency semantics are the
backend's (`-linkGiBps X` is X * 2^30 B/s; `-latencyNs` is per link unless the
fabric uses the custom-latency loader). See
Panel-Scale-Systems/docs/SERVING_INTEGRATION_G0.md for the unit contract.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Flag names the backend's protocol implementation accepts, in the order they
# are rendered. Booleans render as bare flags when true; None means "omit".
_SCALAR_OPTS = (
    ("panel", "-panel"),
    ("planes", "-planes"),
    ("linkGiBps", "-linkGiBps"),
    ("planeGiBps", "-planeGiBps"),
    ("latencyNs", "-latencyNs"),
    ("planeLatencyNs", "-planeLatencyNs"),
    ("policy", "-policy"),
    ("permute", "-permute"),
    ("extents", "-extents"),
    ("seed", "-seed"),
    ("q", "-q"),
    ("maxwin", "-maxwin"),
    ("reconfNs", "-reconfNs"),
    ("redGBps", "-redGBps"),
    ("ocsplan", "-ocsplan"),
    ("graph", "-graph"),
    # legacy fat-tree path (no -panel): topology file, subflows, log name
    ("topo", "-topo"),
    ("sub", "-sub"),
    ("o", "-o"),
)
_FLAG_OPTS = (
    ("nocc", "-nocc"),
    ("ocs", "-ocs"),
    ("nolog", "-nolog"),
    ("preconnected", "-preconnected"),   # flows skip the TCP handshake (persistent channels)
)
# Calibration keys the frontend consumes itself (hardware validation, 2026-09-26):
# they become a backend flag before --htsim_opts or system.json overrides,
# not --htsim_opts entries.
_FRONTEND_KEYS = {
    "recv_flow_finish",        # bool: --recv-flow-finish (receive completes on last byte)
    "collective_launch_ns",    # int:  system.json collective-launch-delay-ns
    "dataset_split_bytes",     # int:  system.json dataset-split-bytes
    "max_dataset_splits",      # int:  system.json preferred-dataset-splits (the maximum with split bytes)
    "active_chunks",           # int:  system.json active-chunks-per-dimension (splits in flight)
    "step_latency_ns",         # int:  system.json collective-step-latency-ns (per ring step, Simple protocol)
    "step_latency_min_bytes",  # int:  system.json collective-step-latency-min-bytes (LL -> Simple switch)
    "collective_bw_efficiency",  # {all-reduce|all-gather|reduce-scatter|all-to-all: (0, 1]}
}
_BW_EFF_COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all")
_KNOWN_KEYS = {k for k, _ in _SCALAR_OPTS} | {k for k, _ in _FLAG_OPTS} | _FRONTEND_KEYS | {
    "spec_version", "name", "nodes", "extra", "description", "source",
    # written by compose_fabric.py: the MemoryPoolSpec this graph was composed
    # with; the backend's pool configuration is the sibling <spec>.pool.json
    "memory_pool_spec",
}
_FILE_KEYS = ("topo", "graph", "ocsplan")


@dataclass(frozen=True)
class FabricSpec:
    """Physical fabric description, loaded from JSON.

    `nodes` is the physical endpoint count (`-nodes`). Every other field maps
    one-to-one onto an `--htsim_opts` flag; `extra` is appended verbatim for
    flags this table does not know yet, so a new backend option never needs a
    frontend release to be usable.
    """

    name: str
    nodes: int
    options: Dict[str, object] = field(default_factory=dict)
    extra: List[str] = field(default_factory=list)
    source: Optional[str] = None

    @staticmethod
    def load(path: str) -> "FabricSpec":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if raw.get("spec_version") != 1:
            raise ValueError(f"{path}: FabricSpec spec_version must be 1")
        # File-valued flags: expand ${VAR} and resolve relative to the spec
        # file, so a spec can point at a fixture in another checkout
        # (${PANEL_ROOT}/src/topos/...) without an absolute path in the repo.
        spec_dir = os.path.dirname(os.path.abspath(path))
        for key in _FILE_KEYS:
            value = raw.get(key)
            if isinstance(value, str):
                value = os.path.expandvars(value)
                if not os.path.isabs(value):
                    value = os.path.join(spec_dir, value)
                if not os.path.exists(value):
                    raise FileNotFoundError(f"{path}: {key} file not found: {value}")
                raw[key] = value
        unknown = sorted(set(raw) - _KNOWN_KEYS)
        if unknown:
            # Refuse rather than drop: a misspelt flag would silently change
            # the fabric. Unknown backend flags go through "extra".
            raise ValueError(f"{path}: unknown FabricSpec keys {unknown}; "
                             "pass new backend flags via \"extra\"")
        nodes = raw.get("nodes")
        if not isinstance(nodes, int) or nodes <= 0:
            raise ValueError(f"{path}: \"nodes\" must be a positive integer")
        extra = raw.get("extra", [])
        if not isinstance(extra, list) or not all(isinstance(e, str) for e in extra):
            raise ValueError(f"{path}: \"extra\" must be a list of strings")
        options = {k: v for k, v in raw.items()
                   if k not in ("spec_version", "name", "nodes", "extra", "description", "source")}
        return FabricSpec(name=str(raw.get("name", os.path.basename(path))),
                          nodes=nodes, options=options, extra=list(extra),
                          source=os.path.abspath(path))

    def htsim_opts(self) -> List[str]:
        """Render the `--htsim_opts` tail, `-nodes` first."""
        out = ["--htsim_opts", "-nodes", str(self.nodes)]
        for key, flag in _SCALAR_OPTS:
            value = self.options.get(key)
            if value is None:
                continue
            if isinstance(value, bool):
                raise ValueError(f"FabricSpec {self.name}: {key} takes a value, not a boolean")
            if isinstance(value, (list, tuple)):
                value = ",".join(str(v) for v in value)
            out += [flag, str(value)]
        for key, flag in _FLAG_OPTS:
            value = self.options.get(key)
            if value is None or value is False:
                continue
            if value is not True:
                raise ValueError(f"FabricSpec {self.name}: {key} is a boolean flag")
            out.append(flag)
        out += self.extra
        return out

    def frontend_flags(self) -> List[str]:
        """Backend flags that go before `--htsim_opts`."""
        v = self.options.get("recv_flow_finish")
        if v is None or v is False:
            return []
        if v is not True:
            raise ValueError(f"FabricSpec {self.name}: recv_flow_finish is a boolean flag")
        return ["--recv-flow-finish"]

    def system_overrides(self) -> Dict[str, Any]:
        """system.json keys this fabric pins (collective calibration)."""
        out: Dict[str, Any] = {}
        for key, sys_key in (("collective_launch_ns", "collective-launch-delay-ns"),
                             ("dataset_split_bytes", "dataset-split-bytes"),
                             ("max_dataset_splits", "preferred-dataset-splits"),
                             ("active_chunks", "active-chunks-per-dimension"),
                             ("step_latency_ns", "collective-step-latency-ns"),
                             ("step_latency_min_bytes", "collective-step-latency-min-bytes")):
            v = self.options.get(key)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"FabricSpec {self.name}: {key} must be a non-negative integer")
            if key == "active_chunks" and v < 1:
                raise ValueError(f"FabricSpec {self.name}: active_chunks must be >= 1")
            out[sys_key] = v
        eff = self.options.get("collective_bw_efficiency")
        if eff is not None:
            if not isinstance(eff, dict):
                raise ValueError(f"FabricSpec {self.name}: collective_bw_efficiency must be an object "
                                 f"keyed by {', '.join(_BW_EFF_COLLECTIVES)}")
            for k, v in eff.items():
                if k not in _BW_EFF_COLLECTIVES:
                    raise ValueError(f"FabricSpec {self.name}: collective_bw_efficiency.{k}: unknown collective "
                                     f"(expected one of {', '.join(_BW_EFF_COLLECTIVES)})")
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 < float(v) <= 1.0):
                    raise ValueError(f"FabricSpec {self.name}: collective_bw_efficiency.{k} must be in (0, 1]")
            out["collective-bw-efficiency"] = {k: float(v) for k, v in eff.items()}
        return out


def logical_npu_count(network_config_path: str) -> int:
    """Product of `npus_count` in the network.yml config_builder wrote."""
    import yaml
    with open(network_config_path, "r", encoding="utf-8") as f:
        topo = yaml.safe_load(f)
    dims = topo["npus_count"]
    total = 1
    for d in dims:
        total *= int(d)
    return total


def kept_dimensions(npus_count) -> List[int]:
    """Indices of the network dimensions the panel backend keeps: every
    dimension of size > 1. The system layer builds each collective's ring
    from these dimensions (dimension 0 is an instance's tensor-parallel
    group, dimension 1 the instances), so they must be preserved -- a
    two-instance TP2 cluster flattened to one dimension of 4 made every
    TP all-reduce a 4-rank ring across both instances, and the instances
    deadlocked waiting on each other's graphs. A dimension of size 1 (the
    frontend writes [1, 3] for a prefill/decode cluster) is dropped because
    astra-network-analytical's parser refuses it and no collective can run
    on it."""
    dims = [int(d) for d in npus_count]
    keep = [i for i, d in enumerate(dims) if d > 1]
    return keep if keep else [0]


def flatten_network_config(network_config_path: str, out_path: Optional[str] = None) -> str:
    """Write the panel backend's copy of network.yml without size-1
    dimensions (see kept_dimensions)."""
    import yaml
    with open(network_config_path, "r", encoding="utf-8") as f:
        topo = yaml.safe_load(f)
    keep = kept_dimensions(topo["npus_count"])
    flat = dict(topo)
    for key in ("topology", "npus_count", "bandwidth", "latency"):
        if isinstance(topo.get(key), list) and topo[key]:
            flat[key] = [topo[key][i] for i in keep if i < len(topo[key])]
    if out_path is None:
        base, _ = os.path.splitext(network_config_path)
        out_path = base + ".panel.yml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(flat, f, sort_keys=False)
    return out_path


def flatten_system_config(system_config_path: str, out_path: Optional[str] = None,
                          keep: Optional[List[int]] = None,
                          overrides: Optional[Dict[str, Any]] = None) -> str:
    """Write the panel backend's copy of system.json whose per-dimension
    collective implementation lists match the kept network dimensions (the
    system layer asserts that they do not exceed the dimension count)."""
    with open(system_config_path, "r", encoding="utf-8") as f:
        sysc = json.load(f)
    flat = dict(sysc)
    for key, val in sysc.items():
        if key.endswith("-implementation") and isinstance(val, list) and len(val) > 1:
            idx = keep if keep is not None else list(range(len(val)))
            flat[key] = [val[i] for i in idx if i < len(val)] or [val[0]]
    if overrides:
        flat.update(overrides)
    if out_path is None:
        base, _ = os.path.splitext(system_config_path)
        out_path = base + ".panel.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(flat, f, indent=2)
    return out_path


SEND_ADMISSION_MODES = ("serialized", "concurrent")


def translate_memory_config(frontend_path: str, num_nodes: int, npus_per_node: int,
                            out_path: Optional[str] = None,
                            pool_config: Optional[str] = None) -> str:
    """Write the panel backend's remote-memory config from the frontend's.

    config_builder emits the casys-kaist multi-level layout
    (`{"remote_mem": {"memory-type", "mem-bw", "mem-latency", "num-devices"},
    "cxl_mem": ..., "local_mem": ...}`); the panel's AnalyticalRemoteMemory
    reads upstream ASTRA-Sim's flat layout (`memory-type`, `num-nodes`,
    `num-npus-per-node`, `remote-mem-latency`, `remote-mem-bw`). Bandwidth
    is GB/s and latency ns in both. Only the CPU (remote) tier exists on the
    panel side in G2: a config with `cxl_mem`, `local_mem` or PIM channels
    is refused rather than silently flattened.
    """
    with open(frontend_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # The CXL tier is served by a physical memory pool of the fabric (G5):
    # its MEM_LOAD/MEM_STORE nodes become flows to the pool's bank. Without a
    # pool in the fabric there is nothing on the panel side to model it.
    unsupported = [k for k in ("cxl_mem", "local_mem") if k in raw]
    if "cxl_mem" in unsupported and pool_config:
        with open(pool_config, "r", encoding="utf-8") as f:
            pool = json.load(f)
        if pool.get("tensor_loc_pool", {}).get("CXL") is None:
            raise ValueError(f"{pool_config}: the fabric's memory pool does not serve the CXL "
                             "tensor location; compose the fabric with a pool for it")
        cxl = raw["cxl_mem"]
        cap = sum(int(p.get("capacity_bytes", 0)) for p in pool.get("pools", []))
        want = float(cxl.get("mem-size", cxl.get("mem_size", 0))) * (1 if cxl.get("mem-size", 0) > 1e6 else 1e9)
        if want and cap and abs(cap - want) / max(cap, want) > 0.10:
            raise ValueError(f"{frontend_path}: cxl_mem size {want:.0f} B differs from the pool "
                             f"capacity {cap} B by more than 10%; size the cluster from the pool spec")
        unsupported.remove("cxl_mem")
    if unsupported:
        raise ValueError(f"{frontend_path}: memory tiers {unsupported} are not supported by the "
                         "panel backend (the CPU/remote tier is analytical; a CXL tier needs a "
                         "memory pool in the fabric)")
    remote = raw.get("remote_mem")
    if remote is None:
        out = {"memory-type": "NO_MEMORY_EXPANSION"}
    else:
        if "pim-channels" in remote:
            raise ValueError(f"{frontend_path}: PIM channels are not supported by the panel backend")
        out = {
            "memory-type": remote.get("memory-type", "PER_NODE_MEMORY_EXPANSION"),
            "num-nodes": int(remote.get("num-devices", num_nodes)),
            "num-npus-per-node": int(npus_per_node),
            "remote-mem-latency": remote["mem-latency"],
            "remote-mem-bw": remote["mem-bw"],
        }
    if out_path is None:
        base, _ = os.path.splitext(frontend_path)
        out_path = base + ".panel.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out_path


def pool_configuration_path(fabric: FabricSpec) -> Optional[str]:
    """The backend's memory-pool configuration for a fabric composed with a
    MemoryPoolSpec (compose_fabric.py writes it next to the spec as
    <spec>.pool.json); None for a fabric without a pool."""
    if not fabric.options.get("memory_pool_spec"):
        return None
    if not fabric.source:
        raise ValueError("a fabric with memory_pool_spec must be loaded from a file")
    path = os.path.splitext(fabric.source)[0] + ".pool.json"
    if not os.path.isfile(path):
        raise ValueError(f"fabric {fabric.name} names a memory pool spec but {path} is missing; "
                         "rerun serving/tools/compose_fabric.py")
    return path


def _kept_dims(network_config_path: str) -> List[int]:
    import yaml
    with open(network_config_path, "r", encoding="utf-8") as f:
        return kept_dimensions(yaml.safe_load(f)["npus_count"])


def build_backend_args(binary: str, fabric: FabricSpec, workload: str,
                       system_config: str, network_config: str,
                       memory_config: str, start_npu_ids: str = "",
                       end_npu_ids: str = "",
                       chakra_send_admission: str = "serialized",
                       num_nodes: int = 1) -> List[str]:
    """Full argv for the serving-mode panel backend.

    `chakra_send_admission` is the backend's per-NPU send gate: "serialized"
    (one in-flight Chakra send per NPU, ASTRA-Sim's HardwareResource default,
    the setting every retained campaign row was produced under) or
    "concurrent". The backend has no default and refuses to start without it.

    `--htsim_opts` must be last: the backend hands everything after it to the
    protocol implementation's own parser.
    """
    if chakra_send_admission not in SEND_ADMISSION_MODES:
        raise ValueError(f"chakra_send_admission must be one of {SEND_ADMISSION_MODES}, "
                         f"got {chakra_send_admission!r}")
    logical = logical_npu_count(network_config)
    if logical < 2:
        # The htsim frontend reads network.yml through astra-network-analytical's
        # parser, which exits with "npus_count (1) should be larger than 1".
        raise ValueError("the panel backend needs at least 2 logical NPUs (its network parser "
                         "refuses a single-NPU topology); use a tp_size >= 2 cluster")
    if logical != fabric.nodes:
        raise ValueError(
            f"FabricSpec {fabric.name} has {fabric.nodes} nodes but the cluster "
            f"config resolves to {logical} logical NPUs ({network_config}); "
            "the packet backend cannot map ranks onto a fabric of a different size")
    if num_nodes <= 0 or logical % num_nodes != 0:
        raise ValueError(f"num_nodes={num_nodes} does not divide the {logical} logical NPUs")
    pool_cfg = pool_configuration_path(fabric)
    panel_memory = translate_memory_config(memory_config, num_nodes, logical // num_nodes,
                                           pool_config=pool_cfg)
    # The frontend's trace_generator writes COMP durations in nanoseconds
    # (profiler time_us * 1000); the backend's default reads microseconds.
    args = [binary, "--serving",
            "--chakra-send-admission=" + chakra_send_admission,
            "--chakra-runtime-unit=ns",
            "--workload-configuration=" + workload,
            "--system-configuration=" + flatten_system_config(system_config, keep=_kept_dims(network_config),
                                                              overrides=fabric.system_overrides()),
            "--network-configuration=" + flatten_network_config(network_config),
            "--remote-memory-configuration=" + panel_memory]
    if start_npu_ids != "":
        args.append("--start-npu-ids=" + start_npu_ids)
    if end_npu_ids != "":
        args.append("--end-npu-ids=" + end_npu_ids)
    if pool_cfg:
        # Pool MEM_LOAD/MEM_STORE nodes become flows to/from the pool's bank
        # device; REMOTE (CPU) locations keep the analytical remote memory.
        args.append("--memory-pool-configuration=" + pool_cfg)
    args += fabric.frontend_flags()
    args += fabric.htsim_opts()
    return args


# Chakra node-type numbering: the casys-kaist schema the frontend writes
# inserts PIM_COMP_NODE = 4, shifting COMP/SEND/RECV/COLL to 5/6/7/8; the
# panel backend's feeder is compiled against upstream's 4/5/6/7. The two
# schemas are otherwise identical (11-line diff, same Node fields, same
# CollectiveCommType values). Read as the wrong type, a COMP node becomes a
# zero-byte send and the first ALLREDUCE never issues.
_FRONTEND_TO_PANEL_NODE_TYPE = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 6: 5, 7: 6, 8: 7}
_FRONTEND_PIM_COMP_NODE = 4
_PANEL_SCHEMA_MARK = "panel_backend_schema"


def transcode_et_for_panel(path: str) -> bool:
    """Rewrite one Chakra .et from the frontend's node numbering to the panel's.

    Idempotent: the file's GlobalMetadata is tagged, and a tagged file is
    left alone (a second remap would turn COMP=4 into the PIM type). Returns
    True when the file was rewritten. Raises on a PIM compute node, which
    the panel backend does not model.
    """
    from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
    from chakra.src.third_party.utils.protolib import decodeMessage, encodeMessage

    with open(path, "rb") as f:
        gm = GlobalMetadata()
        decodeMessage(f, gm)
        if any(a.name == _PANEL_SCHEMA_MARK for a in gm.attr):
            return False
        nodes = []
        while True:
            node = Node()
            if not decodeMessage(f, node):
                break
            nodes.append(node)
    for node in nodes:
        if node.type == _FRONTEND_PIM_COMP_NODE:
            raise ValueError(f"{path}: node {node.id} ({node.name}) is a PIM_COMP_NODE; "
                             "PIM offload is not supported by the panel backend")
        try:
            node.type = _FRONTEND_TO_PANEL_NODE_TYPE[node.type]
        except KeyError:
            raise ValueError(f"{path}: node {node.id} has unknown node type {node.type}")
    mark = gm.attr.add()
    mark.name = _PANEL_SCHEMA_MARK
    mark.int64_val = 1
    tmp = path + ".panel.tmp"
    with open(tmp, "wb") as f:
        encodeMessage(f, gm)
        for node in nodes:
            encodeMessage(f, node)
    os.replace(tmp, path)
    return True


def resolve_binary(cli_value: Optional[str]) -> str:
    """The backend binary: `--panel-backend-binary`, else $PANEL_ASTRA_HTSIM."""
    path = cli_value or os.environ.get("PANEL_ASTRA_HTSIM")
    if not path:
        raise FileNotFoundError(
            "--network-backend htsim needs the panel backend binary: pass "
            "--panel-backend-binary or set PANEL_ASTRA_HTSIM to "
            ".../extern/astra-sim/build/astra_htsim/build/bin/AstraSim_HTSim")
    if not os.access(path, os.X_OK):
        raise FileNotFoundError(f"panel backend binary is not executable: {path}")
    return path
