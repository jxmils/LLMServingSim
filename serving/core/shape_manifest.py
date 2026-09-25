"""Shape manifest: every profile-bundle cell a run asked the trace generator for.

The trace generator turns each emitted layer into a lookup in the profile
bundle (dense / per-sequence / attention / MoE tables, see
``trace_generator._lookup_*``). This module records those lookups so a run
can say, afterwards, exactly which (op, shape) cells it depended on and how
each was resolved: an exact grid hit, an interpolation between two profiled
points, an extrapolation past the profiled range, or a clamp below it.

Two consumers:

* profile-bundle coverage — a run whose cells are mostly extrapolated is
  standing on a bundle that was not profiled for it (hardware validation
  tier 3 checks this for the H100 bundle);
* the tracing request for a cycle-level simulator (Accel-Sim figure, step
  1): the distinct cells are the shapes that need silicon traces.

Recording is a dict update per lookup; a run with millions of lookups keeps
one entry per distinct cell. Enabled by ``python -m serving`` for the whole
run and written to ``<output>.manifest.json``; never fails a run.
"""
import bisect
import json
import os
import threading
from collections import OrderedDict

_lock = threading.Lock()
_ACTIVE = None


def _mode_1d(keys, query):
    """How a 1-D query resolves against a sorted key axis."""
    if not keys:
        return "missing", None, None
    if len(keys) == 1:
        return ("exact" if query == keys[0] else "clamped"), keys[0], keys[0]
    idx = bisect.bisect_right(keys, query)
    if idx == 0:
        return "clamped", keys[0], keys[0]
    if keys[idx - 1] == query:
        return "exact", query, query
    if idx >= len(keys):
        return "extrapolated", keys[-2], keys[-1]
    return "interpolated", keys[idx - 1], keys[idx]


def combine_modes(modes):
    """Worst-of across axes: extrapolated > clamped > interpolated > exact."""
    order = ("missing", "extrapolated", "clamped", "interpolated", "exact")
    for m in order:
        if m in modes:
            return m
    return "exact"


class ShapeManifest:
    def __init__(self):
        self.cells = OrderedDict()   # key tuple -> record dict
        self.bundles = OrderedDict()  # (hardware, model, variant) -> info
        self.lookups = 0

    # -- recording ---------------------------------------------------------
    def bundle(self, perf_db):
        key = (perf_db.get("hardware"), perf_db.get("model"), perf_db.get("variant"))
        if key not in self.bundles:
            self.bundles[key] = {
                "hardware": key[0], "model": key[1], "variant": key[2],
                "root": perf_db.get("root"),
                "available_tps": list(perf_db.get("available_tps") or []),
            }

    def record(self, category, name, tp, query, axes, value_ns):
        """``query``: ordered dict of axis -> integer query value.
        ``axes``: dict axis -> (mode, lo, hi) as returned by ``_mode_1d``.
        """
        key = (category, name, int(tp)) + tuple(int(v) for v in query.values())
        with _lock:
            self.lookups += 1
            rec = self.cells.get(key)
            if rec is None:
                mode = combine_modes([m for (m, _, _) in axes.values()])
                rec = {
                    "category": category, "layer": name, "tp": int(tp),
                    "query": {k: int(v) for k, v in query.items()},
                    "mode": mode,
                    "axes": {k: {"mode": m, "lo": lo, "hi": hi} for k, (m, lo, hi) in axes.items()},
                    "time_ns": int(value_ns), "count": 0,
                }
                self.cells[key] = rec
            rec["count"] += 1

    # -- reporting ---------------------------------------------------------
    def summary(self):
        by_cat = {}
        for rec in self.cells.values():
            c = by_cat.setdefault(rec["category"], {"cells": 0, "lookups": 0, "exact": 0,
                                                     "interpolated": 0, "extrapolated": 0,
                                                     "clamped": 0, "missing": 0,
                                                     "lookups_extrapolated": 0})
            c["cells"] += 1
            c["lookups"] += rec["count"]
            c[rec["mode"]] += 1
            if rec["mode"] == "extrapolated":
                c["lookups_extrapolated"] += rec["count"]
        return {"lookups": self.lookups, "distinct_cells": len(self.cells), "by_category": by_cat}

    def to_dict(self, header=None):
        cells = sorted(self.cells.values(), key=lambda r: (-r["count"], r["category"], r["layer"]))
        return {
            "schema_version": 1,
            "header": header or {},
            "bundles": list(self.bundles.values()),
            "summary": self.summary(),
            "cells": cells,
        }

    def write(self, path, header=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(header), f, indent=1)
        return path

    def one_line(self):
        s = self.summary()
        parts = []
        for cat, c in sorted(s["by_category"].items()):
            parts.append(f"{cat} {c['cells']} cells ({c['exact']} exact, {c['interpolated']} interp, "
                         f"{c['extrapolated']} extrap, {c['clamped']} clamped)")
        return f"{s['distinct_cells']} distinct cells over {s['lookups']} lookups: " + "; ".join(parts)


# -- module-level switch ----------------------------------------------------

def enable():
    global _ACTIVE
    _ACTIVE = ShapeManifest()
    return _ACTIVE


def disable():
    global _ACTIVE
    m, _ACTIVE = _ACTIVE, None
    return m


def get():
    return _ACTIVE


def mode_1d(keys, query):
    return _mode_1d(keys, query)
