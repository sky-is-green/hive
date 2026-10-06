#!/usr/bin/env python
"""Generate the generator's own answers on the cascade bench.

Runs the bench prompts through the same chat path the harness uses
(`/v1/chat/completions`, temperature 0, 700 tokens) so the answers are
comparable with the recorded bench runs.  The checker then labels each answer
(`scion_ok`), giving the J-Bench source set: natural errors, and the
"own answers" half of the source-swap / self-preference audit.

Usage:
    gen_answers.py --server http://127.0.0.1:8897 \
        --out results/jev-readout-20261006/scion-answers.json
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

from experiments.cascade.checkers import check  # noqa: E402
from experiments.cascade.run_cascade import ANSWER_SYSTEM  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
DEFAULT_TASKS = CASCADE / "tasks-bench.json"
DEFAULT_OUT = CASCADE / "results" / "jev-readout-20261006" / "scion-answers.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=700)
    args = ap.parse_args()

    tasks = json.loads(args.tasks.read_text())["tasks"]
    if args.limit:
        tasks = tasks[:args.limit]

    records = []
    t0 = time.time()
    for n, task in enumerate(tasks, 1):
        body = {
            "model": "scion",
            "messages": [{"role": "system", "content": ANSWER_SYSTEM},
                         {"role": "user", "content": task["prompt"]}],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            args.server.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        started = time.time()
        with urllib.request.urlopen(req, timeout=1800) as r:
            data = json.load(r)
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage") or {}
        records.append({
            "id": task["id"], "bucket": task["bucket"], "split": task["split"],
            "prompt": task["prompt"], "scion_answer": content,
            "scion_ok": check(task, content),
            "scion_tokens": int(usage.get("completion_tokens") or 0),
            "scion_ms": round((time.time() - started) * 1000, 1),
        })
        print(f"[{n}/{len(tasks)}] {task['id']} ok={records[-1]['scion_ok']} "
              f"tok={records[-1]['scion_tokens']} {time.time()-t0:.0f}s",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"base": args.server, "records": records}, indent=1))
    ok = sum(1 for r in records if r["scion_ok"])
    print(f"wrote {args.out}: {ok}/{len(records)} correct "
          f"({len(records)-ok} natural negatives)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
