#!/usr/bin/env python
"""Repeat the API-slot controller for DeepSeek and GLM.

The 35B/head/checker decisions are fixed from the measured arms. Each run
repeats the seven API escalations, so the six executable answers and the one
unverifiable choice are both sampled. No 27B is loaded.

    .venv/bin/python experiments/cascade/stability_api_slot.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.cascade.cascade import runtime_spec  # noqa: E402
from experiments.cascade.checkers import check  # noqa: E402 - gold, post hoc
from experiments.cascade.controller_loop import (  # noqa: E402
    ANSWERS, GATE, TASKS, ask_api,
)
from experiments.cascade.verifier import verify  # noqa: E402

OUT = REPO / "experiments/cascade/results/controller-20261022/stability-api-slot.json"
RUNS = 5
MODELS = {
    "DeepSeek V4.1 Flash Free": {"id": "deepseek-v4.1-flash-free", "providerID": "opencode"},
    "GLM 5.3 Flash Free": {"id": "glm-5.3-flash-free", "providerID": "opencode"},
}


def one_api(prompt: str, model: dict, attempts: int = 2) -> dict:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return ask_api(prompt, model)
        except Exception as exc:  # noqa: BLE001 - a provider failure is a sample
            last = exc
            if attempt + 1 < attempts:
                time.sleep(2 + 2 * attempt)
    raise last if last else RuntimeError("API call failed")


def main() -> int:
    tasks = {task["id"]: task for task in json.loads(TASKS.read_text())["tasks"]}
    answers = {row["id"]: row for row in json.loads(ANSWERS.read_text())["records"]}
    head = next(arm for arm in json.loads(GATE.read_text())["the_arms"]
                if "176B" in arm["the_judge"])
    p_correct = head["the_p_correct"]
    threshold = float(head["the_best_net"]["the_threshold"])

    plan = {}
    for task_id, task in tasks.items():
        spec = runtime_spec(task)
        text35 = answers[task_id].get("scion_answer") or ""
        v35 = verify(spec, text35)
        p = p_correct.get(task_id)
        gold35 = bool(check(task, text35))
        if v35["verdict"] == "pass":
            route = "35B"
        elif v35["verdict"] == "unknown" and p is not None and p >= threshold:
            route = "35B+head"
        else:
            route = "API"
        plan[task_id] = {
            "task": task, "spec": spec, "answer35": text35, "v35": v35,
            "gold35": gold35, "p": p, "route": route,
            "t35": float(answers[task_id].get("scion_ms") or 0.0),
        }
    api_ids = [task_id for task_id, item in plan.items() if item["route"] == "API"]
    print(f"API tasks ({len(api_ids)}): {api_ids}")

    rows, runs = [], []
    for model_label, model in MODELS.items():
        for run in range(1, RUNS + 1):
            started = time.perf_counter()
            run_rows = []
            for task_id, item in plan.items():
                gold = item["gold35"]
                routed = item["route"]
                t_api = 0.0
                a_api = None
                v_api = None
                error = None
                if routed == "API":
                    try:
                        got = one_api(item["task"]["prompt"], model)
                        a_api = got["text"]
                        t_api = got["seconds"] * 1000.0
                        v_api = verify(item["spec"], a_api)
                        gold = bool(check(item["task"], a_api))
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"
                        gold = False
                total = (item["t35"] + (2.52 * 1000.0 if routed == "35B+head" else 0.0)
                         + t_api + item["v35"]["latency_ms"])
                run_rows.append({
                    "id": task_id, "route": routed, "p": item["p"],
                    "gold35": item["gold35"], "gold_final": gold,
                    "correct_sent_up": bool(item["gold35"] and routed == "API"),
                    "a_api": a_api, "v_api": v_api, "error": error,
                    "t35_ms": round(item["t35"], 3),
                    "t_api_ms": round(t_api, 3),
                    "total_ms": round(total, 3),
                })
            correct = sum(row["gold_final"] for row in run_rows)
            api_rows = [row for row in run_rows if row["route"] == "API"]
            runs.append({
                "model": model_label, "run": run,
                "correct": correct, "n": len(run_rows),
                "api_answers": {row["id"]: (row["v_api"] or {}).get("verdict")
                                if row["a_api"] else "error"
                                for row in api_rows},
                "api_gold": {row["id"]: row["gold_final"] for row in api_rows},
                "api_errors": [row["id"] for row in api_rows if row["error"]],
                "correct_sent_up": sum(row["correct_sent_up"] for row in run_rows),
                "false_rejects": sum(1 for row in run_rows
                                     if row["gold35"] and row["route"] != "API"
                                     and False),
                "p97_caught": sum(1 for row in run_rows
                                  if (row["p"] or 0) >= 0.97
                                  and not row["gold35"] and row["route"] == "API"),
                "p97_total": sum(1 for row in run_rows
                                 if (row["p"] or 0) >= 0.97 and not row["gold35"]),
                "total_ms": round(sum(row["total_ms"] for row in run_rows), 3),
                "mean_ms": round(statistics.mean(row["total_ms"] for row in run_rows), 3),
                "conditional_ms": round(statistics.mean(row["total_ms"] for row in api_rows), 3)
                if api_rows else None,
                "seconds": round(time.perf_counter() - started, 3),
            })
            rows.extend({"model": model_label, "run": run, **row} for row in run_rows)
            print(f"{model_label:26s} run {run}/{RUNS} correct={correct}/28 "
                  f"api={len(api_rows)} sent_up={runs[-1]['correct_sent_up']} "
                  f"mean={runs[-1]['mean_ms']/1000:.3f}s", flush=True)

    summary = {}
    for model_label in MODELS:
        model_runs = [run for run in runs if run["model"] == model_label]
        model_rows = [row for row in rows if row["model"] == model_label]
        summary[model_label] = {
            "runs": len(model_runs),
            "served": {run["run"]: run["correct"] for run in model_runs},
            "correct_total": sum(run["correct"] for run in model_runs),
            "n_total": sum(run["n"] for run in model_runs),
            "runs_28": sum(1 for run in model_runs if run["correct"] == 28),
            "runs_27": sum(1 for run in model_runs if run["correct"] == 27),
            "correct_sent_up": sum(run["correct_sent_up"] for run in model_runs),
            "p97_caught": sum(run["p97_caught"] for run in model_runs),
            "p97_total": sum(run["p97_total"] for run in model_runs),
            "api_errors": sum(len(run["api_errors"]) for run in model_runs),
            "mean_ms": round(statistics.mean(run["total_ms"] for run in model_runs), 3),
            "mean_of_means_ms": round(statistics.mean(run["mean_ms"] for run in model_runs), 3),
            "conditional_ms": round(statistics.mean(run["conditional_ms"] for run in model_runs
                                                     if run["conditional_ms"] is not None), 3),
            "total_s": round(sum(run["total_ms"] for run in model_runs) / 1000.0, 2),
            "api_answers": {
                task_id: dict(Counter(
                    (run["api_answers"].get(task_id, "missing")) for run in model_runs
                )) for task_id in api_ids
            },
            "api_gold": {
                task_id: dict(Counter(
                    run["api_gold"].get(task_id, False) for run in model_runs
                )) for task_id in api_ids
            },
        }

    payload = {
        "name": "stability-api-slot-20261022",
        "protocol": (
            "five full API-slot runs per model; fixed measured 35B/head/checker; "
            "the seven API escalations repeated; no 27B; gold post hoc"
        ),
        "runs_per_model": RUNS,
        "models": MODELS,
        "api_ids": api_ids,
        "summary": summary,
        "runs": runs,
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
