#!/usr/bin/env python
"""The controller with the API in the 27B slot.

Same measured 35B/head arms and same checker as the scripted loop; the six
checker failures and the one head-rejected choice go straight to the live API.
No local escalation model is loaded.  Gold is applied after each decision.

    .venv/bin/python experiments/cascade/api_replaces_27b.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.cascade.cascade import runtime_spec  # noqa: E402
from experiments.cascade.checkers import check  # noqa: E402 - gold, post hoc
from experiments.cascade.controller_loop import (  # noqa: E402
    ANSWERS, GATE, TASKS, ask_api,
)
from experiments.cascade.verifier import verify  # noqa: E402

OUT = REPO / "experiments/cascade/results/controller-20261022/api-replaces-27b.json"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(0, min(len(ordered) - 1, int(q * len(ordered)) - 1))]


def main() -> int:
    tasks = {task["id"]: task for task in json.loads(TASKS.read_text())["tasks"]}
    answers = {row["id"]: row for row in json.loads(ANSWERS.read_text())["records"]}
    head = next(arm for arm in json.loads(GATE.read_text())["the_arms"]
                if "176B" in arm["the_judge"])
    p_correct = head["the_p_correct"]
    threshold = float(head["the_best_net"]["the_threshold"])

    rows, steps = [], []
    for task_id, task in tasks.items():
        answer = answers[task_id]
        text = answer.get("scion_answer") or ""
        spec = runtime_spec(task)
        p = p_correct.get(task_id)
        v35 = verify(spec, text)
        steps.append({"task": task_id, "phase": "verify35", "verdict": v35["verdict"],
                      "check": v35["check"], "evidence": v35["evidence"]})
        gold35 = bool(check(task, text))
        gold_final = gold35
        routed = "35B"
        t_head = t_api = 0.0
        a_api = None
        v_api = None
        escalate = False

        if v35["verdict"] == "fail":
            escalate = True
        elif v35["verdict"] == "unknown":
            t_head = 2.52 * 1000.0
            steps.append({"task": task_id, "phase": "score", "p": p,
                          "threshold": threshold,
                          "accept": bool(p is not None and p >= threshold)})
            escalate = not (p is not None and p >= threshold)

        if not escalate:
            if v35["verdict"] == "unknown":
                routed = "35B+head"
            steps.append({"task": task_id, "phase": "accept", "routed": routed,
                          "verdict": v35["verdict"]})
        else:
            steps.append({"task": task_id, "phase": "escalate", "to": "API"})
            got = ask_api(task["prompt"])
            a_api = got["text"]
            t_api = got["seconds"] * 1000.0
            v_api = verify(spec, a_api)
            steps.append({"task": task_id, "phase": "verify_api",
                          "verdict": v_api["verdict"], "check": v_api["check"],
                          "evidence": v_api["evidence"],
                          "seconds": round(got["seconds"], 3),
                          "tokens": got.get("tokens"), "cost": got.get("cost")})
            routed = "API"
            gold_final = bool(check(task, a_api))
            steps.append({"task": task_id, "phase": "accept", "routed": routed,
                          "verdict": v_api["verdict"]})

        rows.append({
            "id": task_id, "bucket": task.get("bucket"),
            "gold35": gold35, "head_p": p,
            "p97": bool(not gold35 and (p or 0) >= 0.97),
            "v35": v35, "v_api": v_api, "a_api": a_api,
            "routed": routed, "gold_final": gold_final,
            "correct_sent_up": bool(gold35 and routed not in ("35B", "35B+head")),
            "t35_ms": round(float(answer.get("scion_ms") or 0.0), 3),
            "t_head_ms": round(t_head, 3), "t_api_ms": round(t_api, 3),
            "total_ms": round(float(answer.get("scion_ms") or 0.0) + t_head + t_api
                              + v35["latency_ms"], 3),
        })

    correct = [row for row in rows if row["gold35"]]
    wrong = [row for row in rows if not row["gold35"]]
    p97 = [row for row in rows if row["p97"]]
    api_rows = [row for row in rows if row["routed"] == "API"]
    times = [row["total_ms"] for row in rows]
    summary = {
        "policy": "checker-api (API replaces 27B)",
        "n": len(rows),
        "correct": sum(row["gold_final"] for row in rows),
        "accuracy": round(sum(row["gold_final"] for row in rows) / len(rows), 4),
        "correct_35b": len(correct),
        "escalations": len(api_rows),
        "api_calls": len(api_rows),
        "api_ids": [row["id"] for row in api_rows],
        "served_by": dict(Counter(row["routed"] for row in rows)),
        "correct_sent_up": sum(1 for row in rows if row["correct_sent_up"]),
        "correct_sent_up_ids": [row["id"] for row in rows if row["correct_sent_up"]],
        "p97_total": len(p97),
        "p97_caught": sum(1 for row in p97 if row["routed"] == "API"),
        "p97_ids": [row["id"] for row in p97],
        "false_rejects": sum(1 for row in correct if row["v35"]["verdict"] == "fail"),
        "wrong_served_by_35b": sum(1 for row in wrong
                                   if row["routed"] in ("35B", "35B+head")),
        "mean_ms": round(statistics.mean(times), 3),
        "median_ms": round(statistics.median(times), 3),
        "p95_ms": round(percentile(times, 0.95), 3),
        "conditional_ms": round(statistics.mean(row["total_ms"] for row in api_rows), 3)
        if api_rows else None,
        "total_s": round(sum(times) / 1000.0, 2),
    }
    payload = {
        "name": "api-replaces-27b-20261022",
        "protocol": (
            "same measured 35B/head arms and checker; checker fail or "
            "unconvincing head score goes directly to the live free OpenCode "
            "API; no 27B; gold post hoc"
        ),
        "threshold": threshold,
        "summary": summary,
        "rows": rows,
        "steps": steps,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    return 0 if summary["correct_sent_up"] == 0 and summary["p97_caught"] == summary["p97_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
