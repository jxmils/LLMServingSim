#!/usr/bin/env python3
"""Compose a fabric graph with memory-pool devices (G5 design §2.1).

A pool device from a MemoryPoolSpec becomes two graph devices with ids past
the compute ranks: the pool endpoint P (attached to the fabric over its
attachment edges at link_gbps) and, behind it, the bank B reached only over
P->B / B->P at bank_service_gbps with controller_latency_ns. Transfers to
the pool terminate at B and transfers from it originate at B, so the bank
rate, the controller latency, the attachment port and the shared path each
bottleneck where they should.

Usage:
  compose_fabric.py --base configs/fabric/custom4_shared.json \
      --pool configs/specs/memory_pool/pool-a.json \
      --out configs/fabric/custom4_shared_pool_a.json
Writes <out>.graph next to <out> and a FabricSpec that references it, plus
the backend's memory-pool configuration (<out>.pool.json) mapping pool ids to
endpoint/bank device ids.
"""
import argparse
import json
import os
import sys

GIB = 2 ** 30


def _gbps_to_gibps(gbps: float) -> float:
    # decimal GB/s on the spec side, GiB/s on the htsim command line
    return gbps * 1e9 / GIB


def compose(base_spec_path: str, pool_spec_path: str, out_spec_path: str) -> dict:
    with open(base_spec_path) as f:
        base = json.load(f)
    with open(pool_spec_path) as f:
        pool = json.load(f)
    if base.get("panel") != "custom":
        raise ValueError("compose_fabric: the base fabric must be a custom graph (panel: custom); "
                         "plane-only pool endpoints on hybrid fabrics are G5 step 5")
    base_dir = os.path.dirname(os.path.abspath(base_spec_path))
    graph_path = os.path.join(base_dir, base["graph"])
    with open(graph_path) as f:
        base_lines = [ln.rstrip("\n") for ln in f]
    nodes = int(base["nodes"])
    max_id = nodes - 1
    for ln in base_lines:
        parts = ln.split()
        if parts and parts[0] == "E":
            max_id = max(max_id, int(parts[1]), int(parts[2]))
    next_id = max_id + 1

    out_lines = list(base_lines)
    out_lines.append(f"# memory pool devices from {os.path.basename(pool_spec_path)} "
                     f"(compose_fabric.py): endpoint P then bank B per pool")
    backend_pools = []
    for dev in pool["devices"]:
        p_id, b_id = next_id, next_id + 1
        next_id += 2
        bank_gibps = _gbps_to_gibps(float(dev["bank_service_gbps"]))
        ctrl_ns = float(dev.get("controller_latency_ns", 0))
        out_lines.append(f"# pool {dev['id']}: P={p_id} B={b_id}")
        for att in dev["attachments"]:
            via = att["via"]
            kind, _, idx = via.partition(":")
            if kind not in ("switch", "rank") or not idx.isdigit():
                raise ValueError(f"attachment via must be 'switch:<id>' or 'rank:<id>', got {via!r}")
            peer = int(idx)
            if peer > max_id:
                raise ValueError(f"attachment {via} is not a device of the base graph (max id {max_id})")
            link_gibps = _gbps_to_gibps(float(att["link_gbps"]))
            lat = att.get("link_latency_ns")
            tail = f" {link_gibps:.6f}" + (f" {float(lat):.3f}" if lat is not None else "")
            out_lines.append(f"E {peer} {p_id}{tail}")
            out_lines.append(f"E {p_id} {peer}{tail}")
        out_lines.append(f"E {p_id} {b_id} {bank_gibps:.6f} {ctrl_ns:.3f}")
        out_lines.append(f"E {b_id} {p_id} {bank_gibps:.6f} {ctrl_ns:.3f}")
        backend_pools.append({"id": dev["id"], "endpoint": p_id, "bank": b_id,
                              "capacity_bytes": int(float(dev["capacity_gib"]) * GIB),
                              "allocation_granularity_bytes": int(dev.get("allocation_granularity_bytes",
                                                                          pool.get("allocation_granularity_bytes", 65536)))})

    out_graph = os.path.splitext(out_spec_path)[0] + ".graph"
    with open(out_graph, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    spec = dict(base)
    spec["name"] = f"{base['name']}+{pool['name']}"
    spec["description"] = (f"{base.get('description', '')} Composed with memory pool spec "
                           f"'{pool['name']}' by compose_fabric.py: pool endpoint/bank devices appended "
                           f"past the base graph's highest id.")
    spec["graph"] = os.path.basename(out_graph)
    # Pool transfers are tens of MiB. Under -nocc the window is the whole
    # message unless capped, and a window larger than the queue loses
    # packets, whose retransmit timeouts (0.25 s, doubling) stall a serving
    # run for simulated centuries. Pin the hybrid fixtures' 2 MiB window and
    # 90k-packet queue on every composed fabric.
    spec.setdefault("maxwin", 2097152)
    spec["q"] = max(int(spec.get("q", 0)), 90000)
    spec["memory_pool_spec"] = os.path.relpath(os.path.abspath(pool_spec_path), os.path.dirname(os.path.abspath(out_spec_path)))
    with open(out_spec_path, "w") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")
    backend_cfg = {"access_mode": pool.get("access_mode", "staging"), "pools": backend_pools,
                   "tensor_loc_pool": {"CXL": pool["devices"][0]["id"]} if pool["devices"] else {}}
    backend_path = os.path.splitext(out_spec_path)[0] + ".pool.json"
    with open(backend_path, "w") as f:
        json.dump(backend_cfg, f, indent=2)
        f.write("\n")
    return {"graph": out_graph, "spec": out_spec_path, "backend_config": backend_path, "pools": backend_pools}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = compose(a.base, a.pool, a.out)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    sys.exit(main())
