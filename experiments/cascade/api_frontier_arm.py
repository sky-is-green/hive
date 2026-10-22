"""Measure the OpenCode free DeepSeek model as the frontier API tier.

One fresh OpenCode session per task, the task prompt exactly as the local
arms saw it, and the task's own checker only after the answer is fixed.
The service password is read from the local OpenCode service file and never
written to the result. Sessions are deleted after each answer.

This is a one-off measurement harness, not part of the serving loop. It
requires the OpenCode service that owns this checkout and the free
``opencode/deepseek-v4.1-flash-free`` model.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.cascade.checkers import check  # noqa: E402

TASKS = REPO / "experiments/cascade/tasks-organic-28.json"
OUT = REPO / "experiments/cascade/results/api-frontier-20261022"
SERVICE = Path.home() / ".config/opencode/service.json"
BASE = "http://127.0.0.1:49374"
MODEL = {"id": "deepseek-v4.1-flash-free", "providerID": "opencode"}
AGENT = "build"


def main() -> int:
    tasks = json.loads(TASKS.read_text())["tasks"]
    password = json.loads(SERVICE.read_text())["password"]
    session = requests.Session()
    session.auth = ("opencode", password)
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.time()

    for index, task in enumerate(tasks, 1):
        sid = None
        row = {
            "id": task["id"],
            "bucket": task.get("bucket"),
            "expect": task["checker"].get("expect"),
            "answer": "",
            "correct": False,
            "elapsed_s": None,
            "total_s": None,
            "tokens": None,
            "cost": None,
            "outcome": "error",
            "error": None,
        }
        t0 = time.time()
        try:
            created = session.post(BASE + "/api/session", json={}, timeout=30)
            created.raise_for_status()
            sid = created.json()["data"]["id"]
            model = session.post(
                BASE + f"/api/session/{sid}/model", json={"model": MODEL}, timeout=30
            )
            model.raise_for_status()
            prompt = session.post(
                BASE + f"/api/session/{sid}/prompt",
                json={"text": task["prompt"], "resume": True},
                timeout=30,
            )
            prompt.raise_for_status()
            t_prompt = time.time()
            waited = session.post(
                BASE + f"/api/experimental/session/{sid}/wait", timeout=300
            )
            waited.raise_for_status()
            row["elapsed_s"] = round(time.time() - t_prompt, 3)
            messages = session.get(
                BASE + f"/api/session/{sid}/message",
                params={"type": "assistant", "order": "asc"},
                timeout=30,
            ).json()["data"]
            texts = [
                part["text"]
                for message in messages
                for part in message.get("content", [])
                if part.get("type") == "text"
            ]
            row["answer"] = "\n".join(texts).strip()
            info = session.get(BASE + f"/api/session/{sid}", timeout=30).json()["data"]
            row["tokens"] = info.get("tokens")
            row["cost"] = info.get("cost")
            row["outcome"] = info.get("outcome", "unknown")
            row["correct"] = bool(row["answer"]) and check(task, row["answer"])
            if not row["answer"]:
                row["error"] = "empty assistant answer"
        except Exception as exc:  # noqa: BLE001 - one bad task must not stop the arm
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            row["total_s"] = round(time.time() - t0, 3)
            if sid:
                try:
                    session.delete(BASE + f"/api/session/{sid}", timeout=30)
                except Exception:  # noqa: BLE001
                    pass
        results.append(row)
        print(
            f"[{index:02d}/{len(tasks)}] {row['id']:<18} "
            f"{'PASS' if row['correct'] else 'FAIL'} "
            f"{row['elapsed_s'] or 0:6.2f}s "
            f"{row['answer'][:60]!r}",
            flush=True,
        )
        latencies = [r["elapsed_s"] for r in results if r["elapsed_s"] is not None]
        payload = {
            "name": "api-frontier-opencode-free-20261022",
            "model": "opencode/deepseek-v4.1-flash-free",
            "agent": AGENT,
            "tasks": "experiments/cascade/tasks-organic-28.json",
            "protocol": (
                "one fresh OpenCode session per task; the task prompt exactly; "
                "the build agent; the task checker only after the answer is fixed; "
                "sessions deleted after collection"
            ),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "summary": {
                "n": len(results),
                "correct": sum(r["correct"] for r in results),
                "accuracy": sum(r["correct"] for r in results) / len(results),
                "median_s": round(statistics.median(latencies), 3),
                "mean_s": round(statistics.mean(latencies), 3),
                "max_s": round(max(latencies), 3),
                "tokens_input": sum((r["tokens"] or {}).get("input", 0) for r in results),
                "tokens_output": sum((r["tokens"] or {}).get("output", 0) for r in results),
                "cost_usd": 0,
            },
            "caveats": [
                "Free OpenCode tier, not a paid API; it is only available inside OpenCode.",
                "The build agent and its system prompt/tools were used; this is not a raw chat-completion endpoint.",
                "One sample per task, default agent sampling, not the local arms T=0.7.",
                "The API-driven controller is not measured; this is the API tier alone.",
            ],
            "results": results,
        }
        (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n",
                                          encoding="utf-8")

    passed = sum(row["correct"] for row in results)
    print(f"\nAPI frontier: {passed}/{len(results)} correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
