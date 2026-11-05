#!/usr/bin/env python3
"""Four-arm A/B matrix: Bonsai 2 vs Qwen3.8-27B Q4 vs Scion Flash-Next.

Uses the existing 30-task test split, prompts, token budgets, and checkers from
``experiments/model_matrix.py`` so the numbers are comparable to the 2026-10-30
baseline. The arms are:

- ``bonsai2``: Ternary-Bonsai-2-27B-PQ2_0
- ``qwen27-q4``: Qwen3.8-27B UD-Q4_K_M (Unsloth)
- ``scion-q2``: Scion-FlashNext-176B-A6B Q2_0
- ``scion-q6k``: Scion-FlashNext-176B-A6B Q6_K
- ``gsq-rco-q2``: Qwen3.8-Flash-Next GSQ-RCO Q2_0 (non-ternary reference)

Only one heavy server should be live at a time on this workstation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.model_matrix import (  # noqa: E402
    TASKS,
    run_one,
    sample_tasks,
    summarise,
    token_budget,
)

ARMS: dict[str, dict[str, Any]] = {
    "bonsai2": {
        "base": "http://127.0.0.1:8163/v1",
        "model": "bonsai2",
        "role": "ternary 27B baseline",
        "extra": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "qwen27-q4": {
        "base": "http://127.0.0.1:8162/v1",
        "model": "qwen27-q4",
        "role": "non-ternary Q4 baseline",
        "extra": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "scion-q2": {
        "base": "http://127.0.0.1:8081/v1",
        "model": "scion-flashnext-176b-a6b-q2_0",
        "role": "ternary Flash-Next",
        "extra": {},
    },
    "scion-q6k": {
        "base": "http://127.0.0.1:8081/v1",
        "model": "scion-flashnext-176b-a6b-q6k",
        "role": "Q6 Flash-Next",
        "extra": {},
    },
    "gsq-rco-q2": {
        "base": "http://127.0.0.1:8083/v1",
        "model": "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0",
        "role": "non-ternary GSQ-RCO Q2_0 Flash-Next",
        "extra": {},
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(ARMS))
    parser.add_argument("--tasks", default=str(TASKS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--thinking-budget", type=int, default=0,
                        help="request-level thinking_budget_tokens for the Scion arms (0 = server default)")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())["tasks"]
    tasks = sample_tasks(tasks, 0, 0, args.split)
    selected = [name for name in args.models.split(",") if name in ARMS]
    if not selected:
        raise SystemExit(f"no known arms in {args.models!r}")

    rows: list[dict[str, Any]] = []
    for name in selected:
        arm = ARMS[name]
        if args.thinking_budget > 0 and name.startswith("scion"):
            # Upstream 0.2.10 owns the controller: llama.cpp's request field caps
            # the thinking, and answer_room (on by default) keeps the answer room.
            arm = {**arm, "extra": {**arm.get("extra", {}),
                                    "thinking_budget_tokens": args.thinking_budget}}
        for task in tasks:
            for sample in range(args.samples):
                result = run_one(
                    arm, task, token_budget(task, args.max_tokens), args.timeout,
                )
                rows.append({
                    "model": name,
                    "role": arm["role"],
                    "task_id": task["id"],
                    "benchmark": task.get("benchmark"),
                    "bucket": task.get("bucket"),
                    "sample": sample,
                    **result,
                })
                status = "ok" if result["ok"] else "FAIL"
                print(
                    f"  {name:12s} {task['id']:14s} {status:4s} "
                    f"{result['wall_s']:7.2f}s "
                    f"{str(result.get('error') or '')[:70]}",
                    flush=True,
                )

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_bench: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
        by_model_bench[(row["model"], row["benchmark"])].append(row)

    summary = {
        "models": {
            name: {
                **summarise(model_rows),
                "by_benchmark": {
                    benchmark: summarise(bench_rows)
                    for (model, benchmark), bench_rows in by_model_bench.items()
                    if model == name
                },
            }
            for name, model_rows in by_model.items()
        },
        "tasks": [
            {"id": task["id"], "benchmark": task.get("benchmark"),
             "bucket": task.get("bucket")}
            for task in tasks
        ],
        "samples": args.samples,
        "rows": rows,
    }
    out = Path(args.out) if args.out else (
        REPO / "experiments" / "model-matrix"
        / f"ab-qwen27-flashnext-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["models"], indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
