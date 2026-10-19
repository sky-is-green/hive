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


def wait_healthy(url: str, timeout_s: float) -> bool:
    """Poll /health until the engine answers 200 again.

    The decision-head server SIGTERMs and auto-restarts an engine that goes silent for
    STRATA_ENGINE_STALL_S (90 s by default), which closes the open connection.  That is a
    recoverable event for the client, not a reason to lose the run.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001 — any error just means "not ready yet"
            pass
        time.sleep(3)
    return False


def _replace(src: Path, dst: Path, attempts: int = 10) -> None:
    """os.replace with a short retry.

    Windows refuses to replace a file that another handle still has open (a
    reader, an indexer, the killed process), and the failure is transient.
    """
    import os
    import time

    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--system", default=SYSTEM,
                    help="system message for the decision prompt")
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="embed states to a partial npz every N items (0 disables)")
    ap.add_argument("--retries", type=int, default=4,
                    help="re-attempts per item after an engine restart closes the connection")
    ap.add_argument("--restart-wait", type=float, default=420.0,
                    help="seconds to wait for /health after a dropped connection")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing partial checkpoint and re-embed from scratch")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in args.bench.read_text().splitlines()]
    if args.limit:
        items = items[:args.limit]

    partial = args.out / f"probe-states-{args.name}.partial.npz"

    def save_partial() -> None:
        done = [i for i in items if i["item_id"] in states]
        if not done:
            return
        tmp = partial.with_name(partial.name + ".tmp.npz")   # savez appends .npz unless it is already there
        np.savez_compressed(tmp,
                            item_ids=[i["item_id"] for i in done],
                            labels=[i["label"] for i in done],
                            splits=[i["split"] for i in done],
                            L1=np.stack([states[i["item_id"]] for i in done]))
        _replace(tmp, partial)

    states: dict[str, np.ndarray] = {}
    if partial.exists() and not args.no_resume:
        # close the archive: Windows refuses to replace a file that still has
        # an open handle, and save_partial() replaces exactly this file
        with np.load(partial, allow_pickle=True) as z:
            states = {str(i): z["L1"][n] for n, i in enumerate(z["item_ids"])}
        print(f"resumed {len(states)}/{len(items)} states from {partial.name}", flush=True)

    t0 = time.time()
    dim = next(iter(states.values())).shape[0] if states else None
    n_retried = 0
    for n, item in enumerate(items, 1):
        if item["item_id"] in states:
            continue
        body = {"messages": [{"role": "system", "content": args.system},
                             {"role": "user", "content": order_body(item, False)}],
                "add_generation_prompt": True}
        for attempt in range(args.retries + 1):
            try:
                try:
                    prompt = post(args.server, "/apply-template",
                                  {**body,
                                   "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
                except Exception:
                    # template without thinking-control support (e.g. Clef)
                    prompt = post(args.server, "/apply-template", body)["prompt"]
                emb = post(args.server, "/embedding",
                           {"content": prompt, "embd_normalize": -1})
                vec = np.asarray(emb[0]["embedding"][0], dtype=np.float32)
                break
            except Exception as exc:  # noqa: BLE001 — an engine restart looks like this
                if attempt >= args.retries:
                    raise
                n_retried += 1
                print(f"[{n}/{len(items)}] {type(exc).__name__} on {item['item_id']}; "
                      f"waiting for the engine to come back (attempt {attempt + 1})", flush=True)
                if not wait_healthy(args.server, args.restart_wait):
                    raise SystemExit(f"engine did not return within {args.restart_wait}s")
        dim = vec.shape[0] if dim is None else dim
        states[item["item_id"]] = vec
        if n % 25 == 0 or n == len(items):
            print(f"[{n}/{len(items)}] {time.time()-t0:.0f}s dim={dim} retried={n_retried}",
                  flush=True)
        if args.checkpoint_every and n % args.checkpoint_every == 0:
            save_partial()

    missing = [i["item_id"] for i in items if i["item_id"] not in states]
    if missing:
        raise SystemExit(f"{len(missing)} items have no state: {missing[:5]}")
    save_partial()

    frac = "1"
    arr = np.stack([states[i["item_id"]] for i in items])
    np.savez_compressed(args.out / f"probe-states-{args.name}.npz",
                        item_ids=[i["item_id"] for i in items],
                        labels=[i["label"] for i in items],
                        splits=[i["split"] for i in items],
                        **{f"L{frac}": arr})
    partial.unlink(missing_ok=True)

    train = [i for i in items if i["split"] == "train"]
    test = [i for i in items if i["split"] == "test"]
    y = np.asarray([i["label"] for i in items], dtype=np.float64)
    tr = [n for n, i in enumerate(items) if i["split"] == "train"]
    model = ridge_fit(arr[tr].astype(np.float64), y[tr])
    a, b = platt(ridge_score(arr[tr].astype(np.float64), model), y[tr])
    results = {"name": args.name, "backend": args.server, "layers": {},
               "state_file": f"probe-states-{args.name}.npz",
               "n_layers": None, "dim": dim,
               "items_retried_after_engine_restart": n_retried,
               "resume_checkpoint": partial.name}
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
