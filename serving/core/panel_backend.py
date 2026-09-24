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
from typing import Dict, List, Optional


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
)
_KNOWN_KEYS = {k for k, _ in _SCALAR_OPTS} | {k for k, _ in _FLAG_OPTS} | {
    "spec_version", "name", "nodes", "extra", "description", "source",
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


SEND_ADMISSION_MODES = ("serialized", "concurrent")


def translate_memory_config(frontend_path: str, num_nodes: int, npus_per_node: int,
                            out_path: Optional[str] = None) -> str:
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
    unsupported = [k for k in ("cxl_mem", "local_mem") if k in raw]
    if unsupported:
        raise ValueError(f"{frontend_path}: memory tiers {unsupported} are not supported by the "
                         "panel backend (G2 models the CPU/remote tier only)")
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
    if logical != fabric.nodes:
        raise ValueError(
            f"FabricSpec {fabric.name} has {fabric.nodes} nodes but the cluster "
            f"config resolves to {logical} logical NPUs ({network_config}); "
            "the packet backend cannot map ranks onto a fabric of a different size")
    if num_nodes <= 0 or logical % num_nodes != 0:
        raise ValueError(f"num_nodes={num_nodes} does not divide the {logical} logical NPUs")
    panel_memory = translate_memory_config(memory_config, num_nodes, logical // num_nodes)
    args = [binary, "--serving",
            "--chakra-send-admission=" + chakra_send_admission,
            "--workload-configuration=" + workload,
            "--system-configuration=" + system_config,
            "--network-configuration=" + network_config,
            "--remote-memory-configuration=" + panel_memory]
    if start_npu_ids != "":
        args.append("--start-npu-ids=" + start_npu_ids)
    if end_npu_ids != "":
        args.append("--end-npu-ids=" + end_npu_ids)
    args += fabric.htsim_opts()
    return args


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
