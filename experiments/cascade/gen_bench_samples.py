#!/usr/bin/env python
"""Generate bench samples for the monitor pipeline on any served model.

Mirrors correction_eval.py's generation loop (ANSWER_SYSTEM, temp 0.7,
max_tokens 700, template-default thinking) without the decision-head scoring,
so a non-ternary model (e.g. Qwen3.8-27B IQ4_XS) can produce the same-shaped
monitor dataset.  JSONL records, resumable.

Usage:
    gen_bench_samples.py --server http://127.0.0.1:8896 \
        --out results/jev-readout-20261006/qwen-monitor-samples.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from checkers import check  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
ANSWER_SYSTEM = ("You are a precise assistant. Answer the user's task. For code "
                 "questions return only the requested Python function in a "
                 "single code block.")


def post(url: str, path: str, body: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--tasks", type=Path, default=CASCADE / "tasks-bench.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--thinking", action="store_true",
                    help="force chat_template_kwargs enable_thinking=true")
    args = ap.parse_args()

    tasks = json.loads(args.tasks.read_text())["tasks"]
    done: dict[str, list] = {}
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            rec = json.loads(line)
            done.setdefault(rec["task_id"], []).append(rec)
        print(f"resume: {len(done)} tasks", flush=True)

    sf = open(args.out, "a")
    t0 = time.time()
    empty_reasoning = 0
    for n, task in enumerate(tasks, 1):
        rows = done.get(task["id"], [])
        if len(rows) >= args.samples:
            continue
        for k in range(len(rows), args.samples):
            body = {"model": "local",
                    "messages": [{"role": "system", "content": ANSWER_SYSTEM},
                                 {"role": "user", "content": task["prompt"]}],
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature}
            if args.thinking:
                body["chat_template_kwargs"] = {"enable_thinking": True}
            started = time.time()
            data = post(args.server, "/v1/chat/completions", body)
            msg = (data.get("choices") or [{}])[0].get("message", {})
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            usage = data.get("usage") or {}
            if not reasoning.strip():
                empty_reasoning += 1
            rec = {"task_id": task["id"], "bucket": task["bucket"],
                   "split": task["split"], "sample": k,
                   "content": content, "reasoning": reasoning,
                   "tokens": int(usage.get("completion_tokens") or 0),
                   "ms": round((time.time() - started) * 1000, 1),
                   "ok": bool(check(task, content))}
            rows.append(rec)
            sf.write(json.dumps(rec) + "\n")
            sf.flush()
        done[task["id"]] = rows
        print(f"[{n}/{len(tasks)}] {task['id']} "
              f"ok={sum(r['ok'] for r in rows)}/{len(rows)} "
              f"empty_reason={empty_reasoning} {time.time()-t0:.0f}s", flush=True)
    sf.close()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
