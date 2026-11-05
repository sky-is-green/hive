#!/usr/bin/env python3
"""Compare the direct Scion Q6_K lane with the legacy Q2 bridge on the same 30 tasks.

The two inputs are the ``--out`` JSONs from
``experiments/ab_qwen27_flashnext.py`` (one arm each).  Both arms must have run
with the same task split, samples, max_tokens and thinking budget; the script
refuses to compare different task sets.

It reports the headline accuracy and, because the earlier Q2 result was
mostly a budget-starvation result, the finish reasons, completion tokens and
truncation rate per arm, plus every task where the two disagree.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> tuple[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    models = data.get("models") or {}
    if len(models) != 1:
        raise SystemExit(f"{path}: expected exactly one model, got {sorted(models)}")
    name = next(iter(models))
    return name, {"summary": models[name], "rows": data.get("rows") or []}


def reason_counts(rows: list[dict[str, Any]]) -> Counter:
    return Counter((row.get("finish_reason") or "none") for row in rows)


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direct", type=Path, help="the Q6_K arm JSON")
    parser.add_argument("bridge", type=Path, help="the Q2 bridge arm JSON")
    args = parser.parse_args()

    direct_name, direct = load(args.direct)
    bridge_name, bridge = load(args.bridge)

    d_ids = [row["task_id"] for row in direct["rows"]]
    b_ids = [row["task_id"] for row in bridge["rows"]]
    if d_ids != b_ids:
        raise SystemExit("the two runs did not use the same task order/set; rerun with the same split")

    print(f"direct: {direct_name}  ({args.direct})")
    print(f"bridge: {bridge_name}  ({args.bridge})")
    print()
    print(f"{'metric':28s} {direct_name:>16s} {bridge_name:>16s}")
    for key in ("n", "accuracy", "errors", "median_s", "p95_s", "total_completion_tokens"):
        dv = direct["summary"].get(key)
        bv = bridge["summary"].get(key)
        if key == "accuracy" and isinstance(dv, (int, float)) and isinstance(bv, (int, float)):
            dv, bv = f"{100 * dv:.1f}%", f"{100 * bv:.1f}%"
        print(f"{key:28s} {str(dv):>16s} {str(bv):>16s}")

    print()
    for label, run in (("direct", direct), ("bridge", bridge)):
        rows = run["rows"]
        counts = reason_counts(rows)
        trunc = sum(1 for row in rows if row.get("finish_reason") == "length")
        print(f"{label:8s} finish_reason: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        print(f"{label:8s} truncated: {trunc}/{len(rows)} ({pct(trunc, len(rows))})")
    print()

    print("per benchmark (accuracy):")
    for bench in sorted({row.get("benchmark") for row in direct["rows"]}):
        d = [row for row in direct["rows"] if row.get("benchmark") == bench]
        b = [row for row in bridge["rows"] if row.get("benchmark") == bench]
        dok = sum(1 for row in d if row.get("ok"))
        bok = sum(1 for row in b if row.get("ok"))
        print(f"  {bench:12s} direct {dok:2d}/{len(d):2d}  bridge {bok:2d}/{len(b):2d}")

    flips = [
        (d, b) for d, b in zip(direct["rows"], bridge["rows"])
        if bool(d.get("ok")) != bool(b.get("ok"))
    ]
    print()
    if not flips:
        print("no disagreements: every task has the same pass/fail on both arms")
    else:
        print(f"disagreements ({len(flips)}):")
        for d, b in flips:
            print(f"  {d['task_id']:16s} direct={'PASS' if d.get('ok') else 'FAIL':4s} "
                  f"bridge={'PASS' if b.get('ok') else 'FAIL':4s} "
                  f"direct_tokens={(d.get('usage') or {}).get('completion_tokens')} "
                  f"bridge_tokens={(b.get('usage') or {}).get('completion_tokens')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
