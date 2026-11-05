#!/usr/bin/env python
"""Model x task matrix for Hive model selection.

Uses the existing cascade task schema and checkers
(``experiments/cascade/tasks-*.json`` and ``checkers.check``).  The point is a
role/quality/cost scorecard, not one "best model": a model can win on math and
lose on open-ended QA.

Pilot:

    .venv/bin/python experiments/model_matrix.py --per-benchmark 2 --samples 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.cascade.checkers import check  # noqa: E402

TASKS = REPO / "experiments" / "cascade" / "tasks-bench.json"
RESULTS = REPO / "experiments" / "model-matrix"

MODELS = {
    "bonsai27": {
        "base": "http://127.0.0.1:8163/v1",
        "model": "bonsai",
        "role": "planner/face candidate",
        "extra": {},
    },
    "qwen35": {
        "base": "http://127.0.0.1:8162/v1",
        "model": "Qwen3.8-35B-A3B-IQ2_M",
        "role": "planner/face candidate",
        "extra": {},
    },
    "lfm3-1.7b": {
        "base": "http://127.0.0.1:8155/v1",
        "model": "lfm3",
        "role": "cheap/mechanics candidate",
        "extra": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "ds41": {
        "base": "http://127.0.0.1:8150/v1",
        "model": "deepseek-v4.1-flash-gsq",
        "role": "heavy/escalation candidate",
        "extra": {},
    },
    "scion35": {
        "base": "http://127.0.0.1:8082/v1",
        "model": "scion-35b-a3b-pq2_0-corr",
        "role": "controller/agency candidate",
        "extra": {},
    },
}

SYSTEM = (
    "You are a precise assistant. Answer the user's task. For code questions "
    "return only the requested Python function in a single code block."
)


def post(base: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def sample_tasks(tasks: list[dict[str, Any]], per_benchmark: int, limit: int,
                 split: str = "") -> list[dict[str, Any]]:
    if split:
        tasks = [t for t in tasks if t.get("split", "train") == split]
    if limit:
        return tasks[:limit]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[task.get("benchmark", "unknown")].append(task)
    picked = []
    for name in sorted(grouped):
        pool = grouped[name]
        picked.extend(pool[:per_benchmark] if per_benchmark > 0 else pool)
    return picked


def token_budget(task: dict[str, Any], override: int) -> int:
    if override:
        return override
    return {"reasoning": 2048, "code": 1536, "qa": 512}.get(task.get("bucket"), 768)


def run_one(model: dict[str, Any], task: dict[str, Any], max_tokens: int,
            timeout: float) -> dict[str, Any]:
    body = {
        "model": model["model"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": task["prompt"]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": False,
    }
    body.update(model.get("extra") or {})
    started = time.perf_counter()
    try:
        out = post(model["base"], "/chat/completions", body, timeout)
        wall = time.perf_counter() - started
        choice = (out.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        # Some servers put the visible answer in reasoning_content for
        # thinking models; only use it when content is empty.
        if not text.strip():
            text = message.get("reasoning_content") or ""
        timings = out.get("timings") or {}
        return {
            "ok": bool(check(task, text)),
            "answer": text[:1200],
            "wall_s": round(wall, 3),
            "finish_reason": choice.get("finish_reason"),
            "usage": out.get("usage"),
            "timings": {
                "prompt_n": timings.get("prompt_n"),
                "predicted_n": timings.get("predicted_n"),
                "prompt_per_second": timings.get("prompt_per_second"),
                "predicted_per_second": timings.get("predicted_per_second"),
            },
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - a failed call is a data point
        return {
            "ok": False,
            "answer": "",
            "wall_s": round(time.perf_counter() - started, 3),
            "finish_reason": None,
            "usage": None,
            "timings": {},
            "error": str(exc)[:300],
        }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    latencies = [r["wall_s"] for r in rows]
    tokens = [((r.get("usage") or {}).get("completion_tokens") or 0) for r in rows]
    return {
        "n": len(rows),
        "accuracy": round(sum(r["ok"] for r in rows) / len(rows), 4),
        "errors": sum(1 for r in rows if r.get("error")),
        "median_s": round(statistics.median(latencies), 3),
        "p95_s": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 3),
        "total_completion_tokens": sum(tokens),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(TASKS))
    parser.add_argument("--per-benchmark", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", default="test",
                        help="task split to use; empty = all")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="0 = task-aware budget (reasoning 2048, code 1536, qa 512)")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    tasks = json.loads(Path(args.tasks).read_text())["tasks"]
    tasks = sample_tasks(tasks, args.per_benchmark, args.limit, args.split)
    print(f"tasks {len(tasks)}: {dict(Counter(t.get('benchmark') for t in tasks))}", flush=True)

    selected = [m for m in args.models.split(",") if m in MODELS]
    if not selected:
        raise SystemExit(f"no known models in {args.models!r}")

    all_rows: list[dict[str, Any]] = []
    for model_id in selected:
        model = MODELS[model_id]
        for task in tasks:
            for sample in range(args.samples):
                result = run_one(model, task, token_budget(task, args.max_tokens), args.timeout)
                row = {
                    "model": model_id,
                    "role": model["role"],
                    "task_id": task["id"],
                    "benchmark": task.get("benchmark"),
                    "bucket": task.get("bucket"),
                    "sample": sample,
                    **result,
                }
                all_rows.append(row)
                status = "ok" if result["ok"] else "FAIL"
                print(
                    f"  {model_id:12s} {task['id']:14s} {status:4s} "
                    f"{result['wall_s']:7.2f}s {str(result.get('error') or '')[:60]}",
                    flush=True,
                )

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_bench: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_model[row["model"]].append(row)
        by_model_bench[(row["model"], row.get("benchmark") or "?")].append(row)

    summary = {
        "models": {
            model_id: {
                **summarise(rows),
                "by_benchmark": {
                    name: summarise(bench_rows)
                    for (mid, name), bench_rows in by_model_bench.items()
                    if mid == model_id
                },
            }
            for model_id, rows in by_model.items()
        },
        "tasks": [
            {"id": t["id"], "benchmark": t.get("benchmark"), "bucket": t.get("bucket")}
            for t in tasks
        ],
        "samples": args.samples,
        "rows": all_rows,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS / f"matrix-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["models"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
