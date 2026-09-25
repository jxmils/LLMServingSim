#!/usr/bin/env python3
"""Synthetic prefix-reuse workload for the pooling-capacity experiment (plan §9 B).

G prefix groups; every request of a group shares the group's prefix (P
tokens, a multiple of the lower tier's 256-token coarse block so a recall can
hit) followed by U unique tokens, and generates O output tokens. Requests are
issued round by round: round r issues one request per group in group order,
so a group's prefix is needed again after all the other groups' prefixes have
passed through the caches -- the lower tier's capacity (in prefixes) is what
decides whether that return is a recall or a recompute.

Instance pinning (``--pin``) writes an ``instance_id`` per request for the
CUSTOM routing policy: ``balanced`` alternates groups between instances,
``skew:<f>`` sends fraction f of the groups to instance 0 (imbalance), ``none``
writes no field (LOAD/RR decide).

Usage:
  python -m serving.tools.gen_prefix_workload --groups 8 --per-group 3 --prefix 512 --unique 128 \
      --output-toks 8 --spacing-ms 5 --pin skew:0.75 --out workloads/expB_g8x3_512p_skew75.jsonl
"""
import argparse
import json
import random
import sys

COARSE = 256


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--per-group", type=int, default=3, help="requests per group (= rounds)")
    ap.add_argument("--prefix", type=int, default=512, help="shared prefix tokens (multiple of 256)")
    ap.add_argument("--unique", type=int, default=128)
    ap.add_argument("--output-toks", type=int, default=8)
    ap.add_argument("--spacing-ms", type=float, default=5.0, help="inter-arrival spacing")
    ap.add_argument("--pin", default="none", help="none | balanced | skew:<fraction of groups on instance 0>")
    ap.add_argument("--vocab", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.prefix % COARSE:
        sys.exit(f"--prefix must be a multiple of {COARSE} (lower-tier coarse block)")
    rnd = random.Random(a.seed)
    prefixes = [[rnd.randrange(a.vocab) for _ in range(a.prefix)] for _ in range(a.groups)]
    if a.pin == "none":
        owner = [None] * a.groups
    elif a.pin == "balanced":
        owner = [g % 2 for g in range(a.groups)]
    elif a.pin.startswith("skew:"):
        f = float(a.pin.split(":", 1)[1])
        n0 = int(round(f * a.groups))
        owner = [0 if g < n0 else 1 for g in range(a.groups)]
    else:
        sys.exit("--pin must be none, balanced or skew:<fraction>")
    rows = []
    t = 0
    for r in range(a.per_group):
        for g in range(a.groups):
            ids = prefixes[g] + [rnd.randrange(a.vocab) for _ in range(a.unique)]
            row = {"input_toks": len(ids), "output_toks": a.output_toks, "arrival_time_ns": int(t),
                   "input_tok_ids": ids, "group": g, "round": r}
            if owner[g] is not None:
                row["instance_id"] = owner[g]
            rows.append(row)
            t += a.spacing_ms * 1e6
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} requests: {a.groups} groups x {a.per_group} rounds, prefix {a.prefix} + {a.unique} unique, "
          f"pin={a.pin} -> {a.out}")


if __name__ == "__main__":
    main()
