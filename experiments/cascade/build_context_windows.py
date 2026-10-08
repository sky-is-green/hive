#!/usr/bin/env python
"""Build generation-context windows (template + question + trace) for test-mtp-probe.

Like build_monitor_windows.py, but prefixes each window with the exact
assistant-generation prefix (system + user through /apply-template) so the
forward-pass hidden states sit in the generation context, and records
prompt_len per stream so readouts can address trace-relative positions.

Usage (server up):
    build_context_windows.py --base http://127.0.0.1:8898 \
        --samples .../correction-samples.jsonl \
        --out experiments/cascade/streams/scion-bench-monitor-ctx
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"
ANSWER_SYSTEM = ("You are a precise assistant. Answer the user's task. For code "
                 "questions return only the requested Python function in a "
                 "single code block.")


def post(url: str, path: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--samples", type=Path,
                    default=R / "correction-samples.jsonl")
    ap.add_argument("--tasks", type=Path, default=CASCADE / "tasks-bench.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq", type=int, default=1280)
    ap.add_argument("--trace-cap", type=int, default=768)
    ap.add_argument("--min-tokens", type=int, default=64)
    ap.add_argument("--thinking", action="store_true",
                    help="force chat_template_kwargs enable_thinking=true")
    args = ap.parse_args()

    tasks = {t["id"]: t
             for t in json.loads(args.tasks.read_text())["tasks"]}
    records = [json.loads(line)
               for line in args.samples.read_text().splitlines()]

    order, lengths, labels, plens, truncs, rows = [], [], [], [], [], []
    for rec in records:
        if rec["tokens"] < args.min_tokens:
            continue
        tid = rec["task_id"]
        if tid not in tasks:
            continue
        text = ((rec.get("reasoning") or "") + "\n"
                + (rec.get("content") or "")).strip()
        if not text:
            continue
        tpl = {"messages": [{"role": "system", "content": ANSWER_SYSTEM},
                            {"role": "user", "content": tasks[tid]["prompt"]}],
               "add_generation_prompt": True}
        if args.thinking:
            tpl["chat_template_kwargs"] = {"enable_thinking": True}
        prefix = post(args.base, "/apply-template", tpl)["prompt"]
        pids = [int(t) for t in post(args.base, "/tokenize", {
            "content": prefix, "add_special": True})["tokens"]]
        tids = [int(t) for t in post(args.base, "/tokenize", {
            "content": text, "add_special": False})["tokens"]]
        trunc = len(tids) > args.trace_cap
        tids = tids[:args.trace_cap]
        ids = pids + tids
        if len(ids) > args.seq:
            trunc = True
            ids = ids[:args.seq]
        if not pids:
            continue
        order.append(f"{tid}:{rec['sample']}")
        lengths.append(len(ids))
        labels.append(bool(rec["ok"]))
        plens.append(min(len(pids), len(ids)))
        truncs.append(bool(trunc))
        rows.append(ids)
        if len(rows) % 25 == 0:
            print(f"[{len(rows)}] {order[-1]} "
                  f"prefix={plens[-1]} total={lengths[-1]} "
                  f"ok={labels[-1]}", flush=True)

    arr = np.zeros((len(rows), args.seq), dtype=np.int32)
    for i, ids in enumerate(rows):
        arr[i, :len(ids)] = ids
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokens.bin").write_bytes(arr.tobytes())
    (out / "meta.json").write_text(json.dumps({
        "samples": str(args.samples),
        "order": order, "lengths": lengths, "labels": labels,
        "prompt_len": plens, "truncated": truncs,
        "seq": args.seq, "n_embd": None,
        "tasks": {t["id"]: t["split"] for t in tasks.values()},
    }, indent=1))
    print(f"wrote {out}/tokens.bin {arr.shape}, labels "
          f"{sum(labels)}/{len(labels)}, prefix max {max(plens)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
