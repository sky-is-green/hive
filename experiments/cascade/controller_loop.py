#!/usr/bin/env python
"""The scripted controller: ask the 35B, verify, escalate, score, resolve.

The 35B answers and the head's per-item p are the measured campaign arms.
The 27B and the API are live.  The checker is the execution checker; it sees
the runtime spec only.  Gold is applied after each decision, never inside it.

Policies:
  checker  fail -> 27B; unknown -> head score -> 27B if unconvincing
  head     the head's p on every task -> 27B if unconvincing

After a 27B answer that still fails or is unknown, the API is the last local
resort.  Every step (pass/fail/unknown, score, escalation, accept) is logged.

    .venv/bin/python experiments/cascade/controller_loop.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.cascade.cascade import runtime_spec  # noqa: E402
from experiments.cascade.checkers import check  # noqa: E402 - gold, post hoc
from experiments.cascade.verifier import verify  # noqa: E402

TASKS = REPO / "experiments/cascade/tasks-organic-28.json"
ANSWERS = REPO / "experiments/cascade/results/cascade-arm-escalation-20261020/report.json"
GATE = REPO / "experiments/cascade/results/judge-gate-20261021/judge-gate.json"
OUT = REPO / "experiments/cascade/results/controller-20261022"

BONSAI_URL = "http://127.0.0.1:8897/v1/chat/completions"
BONSAI_MODEL = "bonsai"
OPENCODE = "http://127.0.0.1:49374"
OPENCODE_SERVICE = Path.home() / ".config/opencode/service.json"
API_MODEL = {"id": "deepseek-v4.1-flash-free", "providerID": "opencode"}


# ---------------------------------------------------------------- providers

def ask_27b(prompt: str) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        BONSAI_URL,
        json={"model": BONSAI_MODEL,
              "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 1024, "temperature": 0.7},
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    choice = (payload.get("choices") or [{}])[0]
    return {
        "text": (choice.get("message") or {}).get("content", ""),
        "seconds": time.perf_counter() - started,
        "finish": choice.get("finish_reason"),
        "tokens": (payload.get("usage") or {}).get("completion_tokens"),
    }


def ask_api(prompt: str) -> dict[str, Any]:
    password = json.loads(OPENCODE_SERVICE.read_text())["password"]
    session = requests.Session()
    session.auth = ("opencode", password)
    started = time.perf_counter()
    sid = None
    try:
        created = session.post(OPENCODE + "/api/session", json={}, timeout=30)
        created.raise_for_status()
        sid = created.json()["data"]["id"]
        session.post(OPENCODE + f"/api/session/{sid}/model",
                     json={"model": API_MODEL}, timeout=30).raise_for_status()
        session.post(OPENCODE + f"/api/session/{sid}/prompt",
                     json={"text": prompt, "resume": True}, timeout=30).raise_for_status()
        session.post(OPENCODE + f"/api/experimental/session/{sid}/wait",
                     timeout=300).raise_for_status()
        messages = session.get(
            OPENCODE + f"/api/session/{sid}/message",
            params={"type": "assistant", "order": "asc"}, timeout=30,
        ).json()["data"]
        text = "\n".join(
            part["text"]
            for message in messages
            for part in message.get("content", [])
            if part.get("type") == "text"
        ).strip()
        info = session.get(OPENCODE + f"/api/session/{sid}", timeout=30).json()["data"]
        return {"text": text, "seconds": time.perf_counter() - started,
                "tokens": info.get("tokens"), "cost": info.get("cost"),
                "outcome": info.get("outcome")}
    finally:
        if sid:
            try:
                session.delete(OPENCODE + f"/api/session/{sid}", timeout=30)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------- controller

class Controller:
    def __init__(self, tasks: list[dict], answers: list[dict], p_correct: dict[str, float],
                 threshold: float):
        self.tasks = {task["id"]: task for task in tasks}
        self.answers = {row["id"]: row for row in answers}
        self.p_correct = p_correct
        self.threshold = threshold
        self.cache_27b: dict[str, dict] = {}
        self.cache_api: dict[str, dict] = {}
        self.steps: list[dict] = []

    def log(self, task: str, phase: str, **fields: Any) -> None:
        self.steps.append({"task": task, "phase": phase, **fields})

    def ask_27b(self, task_id: str) -> dict:
        if task_id not in self.cache_27b:
            self.cache_27b[task_id] = ask_27b(self.tasks[task_id]["prompt"])
            self.log(task_id, "ask27", seconds=round(self.cache_27b[task_id]["seconds"], 3),
                     finish=self.cache_27b[task_id]["finish"],
                     tokens=self.cache_27b[task_id]["tokens"])
        return self.cache_27b[task_id]

    def ask_api(self, task_id: str) -> dict:
        if task_id not in self.cache_api:
            self.cache_api[task_id] = ask_api(self.tasks[task_id]["prompt"])
            self.log(task_id, "ask_api", seconds=round(self.cache_api[task_id]["seconds"], 3),
                     tokens=self.cache_api[task_id]["tokens"],
                     cost=self.cache_api[task_id]["cost"])
        return self.cache_api[task_id]

    def run(self, policy: str) -> dict:
        rows = []
        for task_id, task in self.tasks.items():
            answer = self.answers[task_id]
            source = "recorded-35b"
            text35 = answer.get("scion_answer") or ""
            spec = runtime_spec(task)
            p = self.p_correct.get(task_id)
            v35 = verify(spec, text35)
            self.log(task_id, "verify35", verdict=v35["verdict"],
                     check=v35["check"], evidence=v35["evidence"],
                     latency_ms=v35["latency_ms"])
            t35 = float(answer.get("scion_ms") or 0.0)
            t_head = 0.0
            t27 = 0.0
            t_api = 0.0
            routed = "35B"
            accepted = text35
            gold35 = bool(answer.get("scion_ok"))
            gold_final = gold35
            v27 = None
            v_api = None
            a27 = None
            a_api = None

            escalate = False
            if policy == "checker":
                if v35["verdict"] == "fail":
                    escalate = True
                elif v35["verdict"] == "unknown":
                    t_head = 2.52 * 1000.0
                    self.log(task_id, "score", p=p, threshold=self.threshold,
                             accept=bool(p is not None and p >= self.threshold))
                    escalate = not (p is not None and p >= self.threshold)
            elif policy == "head":
                t_head = 2.52 * 1000.0
                self.log(task_id, "score", p=p, threshold=self.threshold,
                         accept=bool(p is not None and p >= self.threshold))
                escalate = not (p is not None and p >= self.threshold)
            else:
                raise ValueError(policy)

            if not escalate:
                if policy == "checker" and v35["verdict"] == "unknown":
                    routed = "35B+head"
                self.log(task_id, "accept", routed=routed, verdict=v35["verdict"])
            else:
                self.log(task_id, "escalate", to="27B")
                got27 = self.ask_27b(task_id)
                t27 = got27["seconds"] * 1000.0
                a27 = got27["text"]
                v27 = verify(spec, a27)
                self.log(task_id, "verify27", verdict=v27["verdict"],
                         check=v27["check"], evidence=v27["evidence"],
                         latency_ms=v27["latency_ms"])
                if v27["verdict"] == "pass":
                    routed = "27B"
                    accepted = a27
                    gold_final = bool(check(task, a27))
                    self.log(task_id, "accept", routed=routed, verdict="pass")
                else:
                    self.log(task_id, "escalate", to="API")
                    got_api = self.ask_api(task_id)
                    t_api = got_api["seconds"] * 1000.0
                    a_api = got_api["text"]
                    v_api = verify(spec, a_api)
                    self.log(task_id, "verify_api", verdict=v_api["verdict"],
                             check=v_api["check"], evidence=v_api["evidence"])
                    routed = "API"
                    accepted = a_api
                    gold_final = bool(check(task, a_api))
                    self.log(task_id, "accept", routed=routed, verdict=v_api["verdict"])

            rows.append({
                "id": task_id, "bucket": task.get("bucket"),
                "gold35": gold35, "head_p": p,
                "p97": bool(not gold35 and (p or 0) >= 0.97),
                "v35": v35, "a27": a27, "v27": v27, "a_api": a_api, "v_api": v_api,
                "routed": routed, "gold_final": gold_final,
                "correct_sent_up": bool(gold35 and routed != "35B" and routed != "35B+head"),
                "t35_ms": round(t35, 3), "t_head_ms": round(t_head, 3),
                "t27_ms": round(t27, 3), "t_api_ms": round(t_api, 3),
                "total_ms": round(t35 + t_head + t27 + t_api
                                  + v35["latency_ms"]
                                  + (v27["latency_ms"] if v27 else 0.0), 3),
                "served_text": accepted,
            })

        correct = [row for row in rows if row["gold35"]]
        wrong = [row for row in rows if not row["gold35"]]
        escalations = [row for row in rows if row["routed"] in ("27B", "API")]
        api_calls = [row for row in rows if row["routed"] == "API"]
        p97 = [row for row in rows if row["p97"]]
        summary = {
            "policy": policy,
            "n": len(rows),
            "correct": sum(row["gold_final"] for row in rows),
            "accuracy": round(sum(row["gold_final"] for row in rows) / len(rows), 4),
            "correct_35b": len(correct),
            "escalations": len(escalations),
            "escalation_ids": [row["id"] for row in escalations],
            "escalation_27b": sum(1 for row in rows if row["routed"] == "27B"),
            "calls_27b": sum(1 for row in rows if row["t27_ms"] > 0),
            "served_27b": sum(1 for row in rows if row["routed"] == "27B"),
            "api_calls": len(api_calls),
            "calls_api": sum(1 for row in rows if row["t_api_ms"] > 0),
            "served_api": sum(1 for row in rows if row["routed"] == "API"),
            "api_ids": [row["id"] for row in api_calls],
            "correct_sent_up": sum(1 for row in rows if row["correct_sent_up"]),
            "correct_sent_up_ids": [row["id"] for row in rows if row["correct_sent_up"]],
            "p97_total": len(p97),
            "p97_caught": sum(1 for row in p97
                              if row["routed"] in ("27B", "API")),
            "p97_ids": [row["id"] for row in p97],
            "false_rejects": sum(1 for row in correct if row["v35"]["verdict"] == "fail"),
            "wrong_served_by_35b": sum(1 for row in wrong
                                       if row["routed"] in ("35B", "35B+head")),
            "served_by": dict(Counter(row["routed"] for row in rows)),
            "mean_ms": round(statistics.mean(row["total_ms"] for row in rows), 3),
            "median_ms": round(statistics.median(row["total_ms"] for row in rows), 3),
            "conditional_ms": round(statistics.mean(row["total_ms"] for row in escalations), 3)
            if escalations else None,
            "total_s": round(sum(row["total_ms"] for row in rows) / 1000.0, 2),
        }
        return {"summary": summary, "rows": rows}


def main() -> int:
    tasks = json.loads(TASKS.read_text())["tasks"]
    answers = json.loads(ANSWERS.read_text())["records"]
    head = next(arm for arm in json.loads(GATE.read_text())["the_arms"]
                if "176B" in arm["the_judge"])
    threshold = float(head["the_best_net"]["the_threshold"])
    controller = Controller(tasks, answers, head["the_p_correct"], threshold)

    print(f"head threshold {threshold}; 35B answers recorded; 27B and API live")
    print("\n=== checker policy ===")
    checker = controller.run("checker")
    print(json.dumps(checker["summary"], indent=2))
    print("\n=== head-only policy (same live 27B/API cache) ===")
    head_only = controller.run("head")
    print(json.dumps(head_only["summary"], indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "controller-20261022",
        "protocol": (
            "scripted loop; 35B answers and head p from the measured campaign "
            "arms; 27B and API live; checker sees runtime spec only; gold post hoc"
        ),
        "bonsai": BONSAI_URL,
        "api": OPENCODE,
        "threshold": threshold,
        "checker": checker,
        "head_only": head_only,
        "steps": controller.steps,
    }
    (OUT / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "steps.jsonl").write_text("\n".join(json.dumps(step) for step in controller.steps) + "\n")
    print(f"\nwrote {OUT / 'report.json'} and steps.jsonl")
    return 0 if (checker["summary"]["correct_sent_up"] == 0
                 and checker["summary"]["p97_caught"] == checker["summary"]["p97_total"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
