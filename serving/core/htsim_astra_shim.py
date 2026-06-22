#!/usr/bin/env python3
"""ASTRA-Sim protocol compatibility shim for the HTSim backend.

This is intentionally a protocol stub.  It lets ``python -m serving`` launch a
non-ASTRA process while reusing the existing ``Controller`` stdin/stdout
contract.  Later milestones should replace the fixed cycle model below with:

1. workload/trace path mapping,
2. HTSim flow export,
3. Broadcom HTSim execution,
4. parsed HTSim completion cycles.

The shim must keep stdout compatible with ``serving.core.controller.Controller``:

* print ``Waiting`` whenever the process can accept another command;
* for a workload command, print
  ``sys[N] iteration I finished, C cycles, exposed communication X cycles.``;
* then print ``Waiting`` again;
* after ``exit``, print ``All Request Has Been Exited`` so ``check_end`` can
  terminate cleanly.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


DEFAULT_FIXED_CYCLES = 1000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTSim ASTRA protocol shim")
    parser.add_argument("--workload-configuration", default=None)
    parser.add_argument("--system-configuration", default=None)
    parser.add_argument("--network-configuration", default=None)
    parser.add_argument("--memory-configuration", default=None)
    parser.add_argument("--start-npu-ids", default="")
    parser.add_argument("--end-npu-ids", default="")
    parser.add_argument("--logical-topology-configuration", default=None)
    parser.add_argument(
        "--fixed-cycles",
        type=int,
        default=DEFAULT_FIXED_CYCLES,
        help="temporary deterministic completion latency returned for every workload",
    )
    parser.add_argument(
        "--shim-log",
        default=None,
        help="optional path for debug logs; stdout stays ASTRA-protocol clean",
    )
    return parser.parse_args()


def _parse_sys_id(workload: str, start_npu_ids: str) -> int:
    """Best-effort system-id extraction for ASTRA-compatible completion lines.

    LLMServingSim usually embeds ``instance<N>`` in independent workload paths.
    DP-shared workload folders may not encode the active sys id, so for the
    first protocol milestone we safely fall back to the first start NPU id or 0.
    """

    match = re.search(r"instance(\d+)_batch", workload)
    if match:
        return int(match.group(1))

    if start_npu_ids:
        first = start_npu_ids.split(",")[0].strip()
        if first.isdigit():
            return int(first)

    return 0


def _parse_iteration_id(workload: str) -> int:
    match = re.search(r"batch(\d+)", workload)
    if match:
        return int(match.group(1))
    return 0


def _log(log_path: str | None, message: str) -> None:
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def _print_waiting() -> None:
    print("Waiting", flush=True)


def _print_completion(sys_id: int, iteration_id: int, cycles: int) -> None:
    print(
        f"sys[{sys_id}] iteration {iteration_id} finished, "
        f"{cycles} cycles, exposed communication {cycles} cycles.",
        flush=True,
    )


def _print_end_marker() -> None:
    print("All Request Has Been Exited", flush=True)


def main() -> int:
    args = _parse_args()
    fixed_cycles = max(0, args.fixed_cycles)

    _log(args.shim_log, "htsim_astra_shim starting")
    _log(args.shim_log, f"argv_workload={args.workload_configuration}")
    _log(args.shim_log, f"system={args.system_configuration}")
    _log(args.shim_log, f"network={args.network_configuration}")
    _log(args.shim_log, f"memory={args.memory_configuration}")

    _print_waiting()

    for raw in sys.stdin:
        command = raw.strip()
        _log(args.shim_log, f"stdin={command}")

        if command == "":
            _print_waiting()
            continue

        if command == "exit":
            _log(args.shim_log, "exit")
            _print_end_marker()
            return 0

        if command in {"pass", "done"}:
            _print_waiting()
            continue

        # Treat every other line as an ASTRA workload path.  We deliberately do
        # not require the path to exist yet; ns-3 and analytical ASTRA receive
        # workload paths, and this milestone only validates the protocol.
        workload = os.path.normpath(command)
        sys_id = _parse_sys_id(workload, args.start_npu_ids)
        iteration_id = _parse_iteration_id(workload)
        _print_completion(sys_id, iteration_id, fixed_cycles)
        _print_waiting()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
