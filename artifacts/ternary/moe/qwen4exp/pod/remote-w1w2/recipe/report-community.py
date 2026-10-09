#!/usr/bin/env python3
"""Print the community-eval report from the fetched artifacts.

    python report-community.py <dir with scion-flashnext-*.json(l)>

prerecorded: gen samples (3/task) -> pass@1 / pass@3 per split and bucket, the
thinking-hard arm (1/task), and the arm table the eval phase wrote.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line]


def pct(a: float) -> str:
    return f"{100.0 * a:.1f}%"


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    samples = load_jsonl(d / "scion-flashnext-samples.jsonl")
    hard = load_jsonl(d / "scion-flashnext-hard.jsonl")
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for r in samples:
        by_task[r["task_id"]].append(r)

    def table(rows: list[dict], per_task: bool) -> None:
        if not rows:
            print("  (no records)")
            return
        groups: dict[tuple, list[dict]] = collections.defaultdict(list)
        if per_task:
            for tid, recs in by_task.items():
                groups[(recs[0]["split"], recs[0]["bucket"])].append(
                    {"ok": sum(bool(r["ok"]) for r in recs) / max(1, len(recs)),
                     "any": any(bool(r["ok"]) for r in recs)})
        else:
            for r in rows:
                groups[(r["split"], r["bucket"])].append({"ok": float(bool(r["ok"])), "any": float(bool(r["ok"]))})
        total = {"ok": [], "any": []}
        for (split, bucket) in sorted(groups):
            g = groups[(split, bucket)]
            p1 = sum(x["ok"] for x in g) / len(g)
            pa = sum(x["any"] for x in g) / len(g)
            total["ok"] += [x["ok"] for x in g]
            total["any"] += [x["any"] for x in g]
            print(f"  {split:5s} {bucket:9s} n={len(g):4d}  pass@1 {pct(p1)}  pass@3 {pct(pa)}")
        if total["ok"]:
            print(f"  {'all':5s} {'':9s} n={len(total['ok']):4d}  pass@1 {pct(sum(total['ok']) / len(total['ok']))}"
                  f"  pass@3 {pct(sum(total['any']) / len(total['any']))}")

    print("== gen (3 samples/task)")
    table(samples, per_task=True)
    print("== hard (thinking, 1/task)")
    table(hard, per_task=False)

    ev = d / "scion-flashnext-eval.json"
    if ev.exists():
        r = json.loads(ev.read_text())
        print("== arms")
        for name, a in (r.get("arms") or {}).items():
            print(f"  {name}: {json.dumps(a)[:200]}")
        print(f"== n_tasks {r.get('n_tasks')} n_samples {r.get('n_samples')} "
              f"primary {r.get('primary_candidate')}")
    else:
        print("== no eval json (arms not computed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
