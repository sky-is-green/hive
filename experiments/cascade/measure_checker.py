#!/usr/bin/env python
"""Measure the execution checker on the same 28 organic tasks.

The candidate answers are the 35B's, the gold labels are the campaign's,
and the head's per-item p come from the judge-gate artifact.  The checker
sees the runtime spec only, never ``checker.expect``; scoring is post hoc.

    .venv/bin/python experiments/cascade/measure_checker.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.cascade.cascade import runtime_spec  # noqa: E402
from experiments.cascade.verifier import verify  # noqa: E402

TASKS = REPO / "experiments/cascade/tasks-organic-28.json"
ANSWERS = REPO / "experiments/cascade/results/cascade-arm-escalation-20261020/report.json"
GATE = REPO / "experiments/cascade/results/judge-gate-20261021/judge-gate.json"
OUT = REPO / "experiments/cascade/results/checker-v0-20261022"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(q * len(ordered)) - 1))
    return ordered[index]


def main() -> int:
    tasks = {task["id"]: task for task in json.loads(TASKS.read_text())["tasks"]}
    records = json.loads(ANSWERS.read_text())["records"]
    head = next(arm for arm in json.loads(GATE.read_text())["the_arms"]
                if "176B" in arm["the_judge"])
    p_correct = head["the_p_correct"]

    rows = []
    for record in records:
        task = tasks.get(record["id"])
        if task is None:
            continue
        verdict = verify(runtime_spec(task), record.get("scion_answer") or "")
        rows.append({
            "id": record["id"],
            "bucket": task.get("bucket"),
            "gold_ok": bool(record.get("scion_ok")),
            "head_p": p_correct.get(record["id"]),
            "p97": bool(not record.get("scion_ok")
                        and (p_correct.get(record["id"]) or 0) >= 0.97),
            **verdict,
        })

    correct = [row for row in rows if row["gold_ok"]]
    wrong = [row for row in rows if not row["gold_ok"]]
    p97 = [row for row in rows if row["p97"]]
    false_rejects = [row for row in correct if row["verdict"] == "fail"]
    caught = [row for row in wrong if row["verdict"] == "fail"]
    unknown = [row for row in wrong if row["verdict"] == "unknown"]
    passes = [row for row in rows if row["verdict"] == "pass"]
    mutation_caught = [row for row in passes if row["mutated_ok"]]
    latencies = [row["latency_ms"] for row in rows]

    summary = {
        "n": len(rows),
        "verdicts": dict(Counter(row["verdict"] for row in rows)),
        "pass_correct": sum(1 for row in correct if row["verdict"] == "pass"),
        "unknown_correct": sum(1 for row in correct if row["verdict"] == "unknown"),
        "correct_total": len(correct),
        "errors_total": len(wrong),
        "errors_caught": len(caught),
        "errors_unknown": len(unknown),
        "errors_passed": sum(1 for row in wrong if row["verdict"] == "pass"),
        "false_rejects": len(false_rejects),
        "false_reject_ids": [row["id"] for row in false_rejects],
        "p97_total": len(p97),
        "p97_caught": sum(1 for row in p97 if row["verdict"] == "fail"),
        "p97_ids": [row["id"] for row in p97],
        "p97_verdicts": {row["id"]: row["verdict"] for row in p97},
        "mutation_trials": len(passes),
        "mutation_caught": len(mutation_caught),
        "mutation_rate": (len(mutation_caught) / len(passes)) if passes else None,
        "latency_median_ms": round(statistics.median(latencies), 3),
        "latency_mean_ms": round(statistics.mean(latencies), 3),
        "latency_p95_ms": round(percentile(latencies, 0.95), 3),
        "latency_max_ms": round(max(latencies), 3),
        "head_mean_ms": head["the_mean_ms"],
        "head_errors_caught": 3,
        "head_correct_sent_up": 0,
    }
    payload = {
        "name": "checker-v0-20261022",
        "protocol": (
            "same 28 organic tasks and 35B answers; the runtime spec only "
            "(kind, task tests, prelude/footer); gold used after the verdict; "
            "code in bwrap, arithmetic by AST; every pass mutation-tested"
        ),
        "answers": str(ANSWERS.relative_to(REPO)),
        "gate": str(GATE.relative_to(REPO)),
        "summary": summary,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'item':18s} {'gold':5s} {'p':>7s}  {'verdict':7s} {'check':10s} "
          f"{'mut':5s} {'ms':>7s}  evidence")
    for row in rows:
        print(f"{row['id']:18s} {str(row['gold_ok']):5s} {row['head_p']!s:>7s}  "
              f"{row['verdict']:7s} {row['check']:10s} "
              f"{str(row['mutated_ok']):5s} {row['latency_ms']:7.2f}  "
              f"{row['evidence'][:70]!r}")
    print()
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT / 'report.json'}")
    return 0 if (summary["p97_caught"] == summary["p97_total"]
                 and summary["false_rejects"] == 0
                 and summary["mutation_rate"] == 1.0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
