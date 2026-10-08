#!/usr/bin/env python
"""Build judge prompts for the slot-logit pilot from a samples JSONL.

Renders SYSTEM + order_body(task, candidate) through the server's
/apply-template (thinking disabled) exactly like correction_eval.py, and writes
one {i:04d}.txt per sample plus index.json with ids/labels/tasks/splits.

Usage (server up):
    make_judge_prompts.py --server http://127.0.0.1:8898 \
        --samples results/.../correction-samples.jsonl \
        --tasks experiments/cascade/tasks-bench.json \
        --out-dir results/.../logit-pilot-scion/prompts
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from community_eval import SYSTEM  # noqa: E402
from readout_judge import order_body  # noqa: E402


def post(url: str, path: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-tokens", type=int, default=0)
    args = ap.parse_args()

    tasks = {t["id"]: t
             for t in json.loads(args.tasks.read_text())["tasks"]}
    recs = [json.loads(line)
            for line in args.samples.read_text().splitlines()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for rec in recs:
        if rec["tokens"] < args.min_tokens:
            continue
        task = tasks[rec["task_id"]]
        prompt = post(args.server, "/apply-template", {
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": order_body(
                             {"prompt": task["prompt"],
                              "candidate": rec["content"]}, False)}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        i = len(index)
        (args.out_dir / f"{i:04d}.txt").write_text(prompt)
        index.append({"file": f"{i:04d}", "key": f"{rec['task_id']}|sample|{rec['sample']}",
                      "task_id": rec["task_id"], "bucket": rec["bucket"],
                      "split": rec["split"], "sample": rec["sample"],
                      "ok": bool(rec["ok"]), "tokens": rec["tokens"],
                      "content": rec["content"]})
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(recs)}]", flush=True)

    (args.out_dir / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} prompts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
