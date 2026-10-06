#!/usr/bin/env python
"""Build the L3 monitor windows from the correction run's samples.

Reads `correction-samples.jsonl` (one JSON per generated sample), tokenizes
``reasoning + content`` with the server's own tokenizer (authoritative for the
GGUF), truncates/pads each stream to ``--seq``, and writes ``tokens.bin``
(int32 [n, seq]) plus ``meta.json`` (order, lengths, labels, task/split) for
``test-mtp-probe``.

The monitor dataset therefore costs no extra generation: it reuses the L1/L2
samples.

Usage (server up):
    build_monitor_windows.py --base http://127.0.0.1:8897 \
        --samples results/jev-readout-20261006/correction-samples.jsonl \
        --out experiments/cascade/streams/scion-bench-monitor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import urllib.request

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import check  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"


def post(url: str, path: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--samples", type=Path, default=R / "correction-samples.jsonl")
    ap.add_argument("--out", type=Path,
                    default=CASCADE / "streams" / "scion-bench-monitor")
    ap.add_argument("--seq", type=int, default=768)
    ap.add_argument("--min-tokens", type=int, default=64)
    args = ap.parse_args()

    tasks = {t["id"]: t
             for t in json.loads((CASCADE / "tasks-bench.json").read_text())["tasks"]}
    records = [json.loads(line) for line in args.samples.read_text().splitlines()]

    order, lengths, labels, rows = [], [], [], []
    for rec in records:
        if rec["tokens"] < args.min_tokens:
            continue
        text = ((rec.get("reasoning") or "") + "\n"
                + (rec.get("content") or "")).strip()
        ids = [int(t) for t in post(args.base, "/tokenize",
                                    {"content": text, "add_special": True})["tokens"]]
        if not ids:
            continue
        ids = ids[:args.seq]
        order.append(f"{rec['task_id']}:{rec['sample']}")
        lengths.append(len(ids))
        labels.append(bool(rec["ok"]))
        rows.append(ids)
        if len(rows) % 25 == 0:
            print(f"[{len(rows)}] {order[-1]} {len(ids)} tok ok={labels[-1]}",
                  flush=True)

    arr = np.zeros((len(rows), args.seq), dtype=np.int32)
    for i, ids in enumerate(rows):
        arr[i, :len(ids)] = ids
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokens.bin").write_bytes(arr.tobytes())
    (out / "meta.json").write_text(json.dumps({
        "samples": str(args.samples),
        "order": order, "lengths": lengths, "labels": labels,
        "seq": args.seq, "n_embd": None,
        "tasks": {t["id"]: t["split"] for t in tasks.values()},
    }, indent=1))
    print(f"wrote {out}/tokens.bin {arr.shape}, labels {sum(labels)}/{len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
