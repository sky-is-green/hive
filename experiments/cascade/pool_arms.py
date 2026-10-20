#!/usr/bin/env python3
"""Pool the repeated runs of one cascade arm into its effect size.

A single draw per task is a *direction*, not a measurement: the
deterministic verifier's catch rate on the same tasks and the same model
ranges from 0 to 4 out of 28 across draws, so a one-draw arm can move by
four items for reasons that have nothing to do with the arm.

The stable regime is the multi-sample one: the same tasks, drawn
repeatedly, pooled.  The published 66-item number is 28 tasks x 3 samples
(the hedges are dropped), and this pools any number of runs of the same
arm.

Usage::

    python -m experiments.cascade.pool_arms 'experiments/cascade/results/cascade-arm-repair-auto-2026*'
    python -m experiments.cascade.pool_arms 'results/*-s1' 'results/*-s2' --label 'the arm'
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]


def load_reports(patterns: list[str]) -> list[dict]:
    reports: list[dict] = []
    for pattern in patterns:
        for path in sorted(Path().glob(pattern)):
            report = path / "report.json"
            if report.is_file():
                reports.append({"path": path, "data": json.loads(report.read_text())})
    return reports


def pool(reports: list[dict]) -> dict:
    """The pooled records and the summary over them."""
    records: list[dict] = []
    for report in reports:
        for record in report["data"].get("records") or []:
            item = dict(record)
            item["_run"] = report["path"].name
            records.append(item)
    out: dict = {"runs": [r["path"].name for r in reports], "items": len(records)}
    if not records:
        return out
    served = [r for r in records if r.get("cascade_ok")]
    out["served_correct"] = len(served)
    out["served_rate"] = round(len(served) / len(records), 4)
    # The runner's own ``false_accept``/``false_reject`` flags are the
    # *ladder's* miss/escalation bookkeeping, not the verifier's: a
    # rejected item counts as a "false accept" there, and a "false
    # reject" only exists once something escalated.  The verifier's own
    # numbers are computed here from the verdict and the local answer's
    # correctness, which is what the pre-registered budgets talk about.
    rejected = [r for r in records if r.get("verdict") == "reject"]
    out["verifier_rejections"] = len(rejected)
    out["verifier_catches"] = sum(1 for r in rejected if not r.get("scion_ok"))
    out["verifier_false_rejects"] = sum(1 for r in rejected if r.get("scion_ok"))
    out["misses"] = sum(1 for r in records
                        if not r.get("cascade_ok") and not r.get("escalated"))
    out["verdicts"] = dict(Counter(r.get("verdict") for r in records))
    out["repairs_attempted"] = sum(1 for r in records if r.get("repair"))
    out["repairs_cleared"] = sum(1 for r in records
                                 if (r.get("repair") or {}).get("cleared"))
    out["gate_scored"] = sum(1 for r in records if r.get("gate_scored"))
    out["gate_fires"] = sum(1 for r in records if r.get("gate_fired"))
    out["gate_cleared"] = sum(1 for r in records if r.get("gate_cleared"))
    # the gate's false flags: it fired on an answer that was right
    out["gate_false_flags"] = sum(1 for r in records
                                  if r.get("gate_fired") and r.get("scion_ok"))
    out["gate_true_catches"] = sum(1 for r in records
                                   if r.get("gate_fired") and not r.get("scion_ok"))
    out["escalated"] = sum(1 for r in records if r.get("escalated"))
    out["flagged"] = sum(1 for r in records if r.get("flagged"))
    out["truncated"] = sum(1 for r in records if r.get("truncated"))
    out["mean_ms"] = round(statistics.mean(
        [r["scion_ms"] for r in records if r.get("scion_ms")]), 1) if any(
        r.get("scion_ms") for r in records) else None
    # per bucket, and per task (the tasks repeat across the runs)
    per_bucket: dict[str, list[int]] = {}
    for r in records:
        bucket = per_bucket.setdefault(r.get("bucket") or "?", [0, 0])
        bucket[0] += 1
        bucket[1] += 1 if r.get("cascade_ok") else 0
    out["per_bucket"] = {k: {"n": v[0], "correct": v[1],
                             "rate": round(v[1] / v[0], 4)}
                         for k, v in sorted(per_bucket.items())}
    per_task: dict[str, list[int]] = {}
    for r in records:
        task = per_task.setdefault(r.get("id") or "?", [0, 0])
        task[0] += 1
        task[1] += 1 if r.get("cascade_ok") else 0
    out["tasks"] = len(per_task)
    out["tasks_always_right"] = sum(1 for v in per_task.values()
                                    if v[1] == v[0])
    out["tasks_never_right"] = sum(1 for v in per_task.values() if v[1] == 0)
    out["tasks_mixed"] = sum(1 for v in per_task.values() if 0 < v[1] < v[0])
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="+",
                        help="the run directories to pool (globs)")
    parser.add_argument("--label", default="the arm")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    reports = load_reports(args.patterns)
    if not reports:
        print("no reports matched", file=sys.stderr)
        return 2
    summary = pool(reports)
    print(f"=== {args.label}: {len(reports)} runs, {summary['items']} items "
          f"over {summary.get('tasks')} tasks")
    print(f"  served-correct   {summary['served_correct']}/{summary['items']} "
          f"({summary['served_rate'] * 100:.1f}%)")
    print(f"  the misses       {summary['misses']}")
    print(f"  the verifier     rejected {summary['verifier_rejections']} "
          f"(caught {summary['verifier_catches']}, "
          f"FALSE rejects {summary['verifier_false_rejects']})")
    print(f"  the repair leg   {summary['repairs_cleared']}/"
          f"{summary['repairs_attempted']} cleared")
    print(f"  the gate         scored {summary['gate_scored']}, "
          f"fired {summary['gate_fires']} "
          f"(true catches {summary['gate_true_catches']}, "
          f"FALSE flags {summary['gate_false_flags']}), "
          f"cleared {summary['gate_cleared']}")
    print(f"  escalated        {summary['escalated']}   flagged {summary['flagged']}")
    print(f"  the verdicts     {summary['verdicts']}")
    if summary.get("mean_ms"):
        print(f"  the mean         {summary['mean_ms'] / 1000:.1f} s an item")
    print(f"  the tasks        always right {summary['tasks_always_right']}, "
          f"mixed {summary['tasks_mixed']}, never right {summary['tasks_never_right']}")
    for bucket, v in summary["per_bucket"].items():
        print(f"    {bucket:<12} {v['correct']}/{v['n']} ({v['rate'] * 100:.0f}%)")
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
