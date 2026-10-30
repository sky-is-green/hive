#!/usr/bin/env python
"""Evaluate an OpenAI-compatible endpoint on a role task file.

Used to score Scion-35B on the wider controller suite with the same prompts
and checkers as the local HF evaluator. Greedy decoding; JSON mode is enabled
for JSON-shaped checkers, matching the role-matrix benchmark.

    python experiments/eval_role_endpoint.py \
      --base http://127.0.0.1:8082 \
      --model scion-35b-a3b-pq2_0-corr \
      --tasks experiments/roles/tasks-controller.json \
      --out ../train-runs/controller-sft-v1/eval-scion.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import check  # noqa: E402
from experiments.role_matrix import SYSTEM  # noqa: E402


def post(base: str, path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    lat = [r["wall_s"] for r in rows]
    return {
        "n": len(rows),
        "accuracy": round(sum(bool(r["ok"]) for r in rows) / len(rows), 4),
        "median_s": round(statistics.median(lat), 3),
        "errors": sum(1 for r in rows if r.get("error")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--roles", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2560)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-response-format", action="store_true",
                        help="do not send response_format for JSON-shaped checkers")
    args = parser.parse_args(argv)

    tasks = json.loads(Path(args.tasks).read_text())["tasks"]
    roles = {r.strip() for r in args.roles.split(",") if r.strip()}
    if roles:
        tasks = [t for t in tasks if t["role"] in roles]
    if args.limit:
        tasks = tasks[:args.limit]

    rows: list[dict] = []
    for task in tasks:
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM.get(task["role"], "You are a Hive role.")},
                {"role": "user", "content": task["prompt"]},
            ],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": False,
        }
        if not args.no_response_format and task["checker"].get("type") in ("json", "ordered"):
            body["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        try:
            out = post(args.base, "/v1/chat/completions", body, args.timeout)
            wall = time.perf_counter() - started
            message = (out.get("choices") or [{}])[0].get("message") or {}
            answer = message.get("content") or ""
            if not str(answer).strip():
                answer = message.get("reasoning_content") or ""
            answer = str(answer)
            ok = bool(check(task, answer))
            error = None
        except Exception as exc:  # noqa: BLE001 - endpoint failure is a failed task
            wall = time.perf_counter() - started
            answer = ""
            ok = False
            error = str(exc)[:300]
        rows.append({"task_id": task["id"], "role": task["role"], "ok": ok,
                     "answer": answer[:2000], "wall_s": round(wall, 3),
                     "error": error})
        print(f"  {task['id']:16s} {'ok' if ok else 'FAIL':4s} {wall:6.2f}s "
              f"{answer[:80]!r}", flush=True)

    by_role = defaultdict(list)
    for row in rows:
        by_role[row["role"]].append(row)
    summary = {
        "base": args.base, "model": args.model, "tasks": str(Path(args.tasks).resolve()),
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "overall": summarise(rows),
        "by_role": {role: summarise(rs) for role, rs in by_role.items()},
        "rows": rows,
    }
    print("\noverall", json.dumps(summary["overall"]))
    for role, stats in summary["by_role"].items():
        print(f"  {role:12s} n={stats.get('n', 0)} acc={stats.get('accuracy', 0):.3f} "
              f"median={stats.get('median_s', 0):.2f}s")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=1) + "\n")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
