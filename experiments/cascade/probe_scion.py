#!/usr/bin/env python
"""Hidden-state probe for a llama-server GGUF (the Scion-35B generator).

Reads the last-token hidden state through ``/embedding`` (server started with
``--embeddings --pooling last --embd-normalize -1``), fits a closed-form ridge
head on the train split and scores the frozen test split.  Same schema as
``probe_judge.py`` (single depth = final) so ``compare_readouts.py`` picks it
up.

Usage:
    probe_scion.py --server http://127.0.0.1:8897 --name scion35b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from probe_judge import platt, ridge_fit, ridge_score  # noqa: E402
from readout_judge import (  # noqa: E402
    best_ba,
    ece,
    metrics,
    no_leak_threshold,
    order_body,
    threshold_stats,
)

DEFAULT_BENCH = REPO / "experiments/cascade/results/jev-readout-20261006/judge-bench.jsonl"
DEFAULT_OUT = REPO / "experiments/cascade/results/jev-readout-20261006"
SYSTEM = "You are a decision function. Reply with the answer label only."


def post(url: str, path: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = [json.loads(line) for line in args.bench.read_text().splitlines()]
    if args.limit:
        items = items[:args.limit]

    states: dict[str, np.ndarray] = {}
    t0 = time.time()
    dim = None
    for n, item in enumerate(items, 1):
        prompt = post(args.server, "/apply-template", {
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": order_body(item, False)}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        emb = post(args.server, "/embedding",
                   {"content": prompt, "embd_normalize": -1})
        vec = np.asarray(emb[0]["embedding"][0], dtype=np.float32)
        dim = vec.shape[0] if dim is None else dim
        states[item["item_id"]] = vec
        if n % 25 == 0 or n == len(items):
            print(f"[{n}/{len(items)}] {time.time()-t0:.0f}s dim={dim}",
                  flush=True)

    frac = "1"
    arr = np.stack([states[i["item_id"]] for i in items])
    np.savez_compressed(args.out / f"probe-states-{args.name}.npz",
                        item_ids=[i["item_id"] for i in items],
                        labels=[i["label"] for i in items],
                        splits=[i["split"] for i in items],
                        **{f"L{frac}": arr})

    train = [i for i in items if i["split"] == "train"]
    test = [i for i in items if i["split"] == "test"]
    y = np.asarray([i["label"] for i in items], dtype=np.float64)
    tr = [n for n, i in enumerate(items) if i["split"] == "train"]
    model = ridge_fit(arr[tr].astype(np.float64), y[tr])
    a, b = platt(ridge_score(arr[tr].astype(np.float64), model), y[tr])
    results = {"name": args.name, "backend": args.server, "layers": {},
               "state_file": f"probe-states-{args.name}.npz",
               "n_layers": None, "dim": dim}
    for split_name, split_items in (("train", train), ("test", test)):
        idx = [n for n, i in enumerate(items) if i["split"] == split_name]
        s = ridge_score(arr[idx].astype(np.float64), model)
        p = 1.0 / (1.0 + np.exp(-(a * s + b)))
        yy = y[idx]
        ba, thr = best_ba(list(p), list(yy))
        entry = {"layer": None, "depth": frac, "n": len(yy),
                 "auc": None, "best_ba": {"bal_acc": ba, "threshold": thr},
                 "ece": ece(list(p), list(yy)),
                 "no_leak": no_leak_threshold(list(p), list(yy)),
                 "at_0.5": threshold_stats(list(p), list(yy), 0.5)}
        results["layers"].setdefault(frac, {})[split_name] = entry
    # AUC via compare_readouts' recompute; keep a local copy too
    from readout_judge import auc
    idx = [n for n, i in enumerate(items) if i["split"] == "test"]
    s = ridge_score(arr[idx].astype(np.float64), model)
    p = 1.0 / (1.0 + np.exp(-(a * s + b)))
    yt = [items[n]["label"] for n in idx]
    results["layers"][frac]["test"]["auc"] = auc(list(p), yt)
    (args.out / f"probe-{args.name}.json").write_text(json.dumps(results, indent=1))
    print(f"=== {args.name} (final-layer probe, test n={len(yt)}) ===")
    print(f"AUC {results['layers'][frac]['test']['auc']:.3f} "
          f"BA {results['layers'][frac]['test']['best_ba']['bal_acc']:.3f} "
          f"ECE {results['layers'][frac]['test']['ece']:.3f}")
    print("wrote", args.out / f"probe-{args.name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
