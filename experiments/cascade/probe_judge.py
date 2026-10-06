#!/usr/bin/env python
"""Hidden-state linear probe judge over J-Bench (T2/T3).

Extracts the last-prompt-token hidden state at requested depths, fits a
closed-form ridge head on the train split (standardised features, lambda=1),
and scores the frozen test split.  States are cached so the same heads can be
applied later to ternary bodies.

Usage:
    python experiments/cascade/probe_judge.py --hf Qwen/Qwen3.5-4B \
        --name qwen35-4b --device cuda --layers 0.5,0.75,1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_BENCH = REPO / "experiments/cascade/results/jev-readout-20261006/judge-bench.jsonl"
DEFAULT_OUT = REPO / "experiments/cascade/results/jev-readout-20261006"

sys.path.insert(0, str(REPO / "experiments" / "cascade"))
from readout_judge import (  # noqa: E402
    auc,
    best_ba,
    ece,
    no_leak_threshold,
    order_body,
    threshold_stats,
)


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> tuple[np.ndarray, float]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y)
    b = float(y.mean())
    return w, (mu, sd, b)


def ridge_score(X: np.ndarray, model) -> np.ndarray:
    w, (mu, sd, b) = model
    return ((X - mu) / sd) @ w + b


def platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit sigmoid(a*s+b) by a coarse grid (enough for ECE)."""
    best = (0.0, 0.0, 1e9)
    for a in (0.5, 1.0, 2.0, 4.0, 8.0):
        for b in np.linspace(-2, 2, 9):
            p = 1.0 / (1.0 + np.exp(-(a * scores + b)))
            nll = -np.mean(labels * np.log(p + 1e-9)
                           + (1 - labels) * np.log(1 - p + 1e-9))
            if nll < best[2]:
                best = (a, b, nll)
    return best[0], best[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--name", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", default="0.5,0.75,1.0",
                    help="comma-separated depth fractions")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=dtype)
    model = model.to(args.device).eval()

    items = [json.loads(line) for line in args.bench.read_text().splitlines()]
    if args.limit:
        items = items[:args.limit]

    def prompt(item: dict) -> str:
        body = order_body(item, swap=False)
        if getattr(tok, "chat_template", None):
            messages = [
                {"role": "system",
                 "content": "You are a decision function. Reply with the answer "
                            "label only."},
                {"role": "user", "content": body},
            ]
            return tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
        return body + "Answer:"

    states: dict[str, dict] = {}
    t0, n_layers = time.time(), None
    for n, item in enumerate(items, 1):
        ids = tok(prompt(item), return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple: embeddings + layers
        if n_layers is None:
            n_layers = len(hs) - 1
            print(f"hidden_states: {len(hs)} entries ({n_layers} layers)",
                  flush=True)
        states[item["item_id"]] = {f: hs[i][0, -1].float().cpu().numpy()
                                   for f, i in _layer_map(args.layers, n_layers)}
        if n % 25 == 0 or n == len(items):
            print(f"[{n}/{len(items)}] {time.time()-t0:.0f}s", flush=True)

    fracs = [f for f, _ in _layer_map(args.layers, n_layers)]
    layers = [i for _, i in _layer_map(args.layers, n_layers)]
    np.savez_compressed(
        args.out / f"probe-states-{args.name}.npz",
        item_ids=[i["item_id"] for i in items],
        labels=[i["label"] for i in items],
        splits=[i["split"] for i in items],
        **{f"L{f}": np.stack([states[i["item_id"]][f] for i in items])
           for f in fracs},
    )

    train = [i for i in items if i["split"] == "train"]
    test = [i for i in items if i["split"] == "test"]
    if not train or not test:
        raise SystemExit("train/test split empty (limit too small); "
                         "increase --limit or drop it")
    results = {"name": args.name, "backend": args.hf, "layers": {},
               "state_file": f"probe-states-{args.name}.npz",
               "n_layers": n_layers}
    for f in fracs:
        Xtr = np.stack([states[i["item_id"]][f] for i in train]).astype(np.float64)
        ytr = np.array([i["label"] for i in train], dtype=np.float64)
        model_r = ridge_fit(Xtr, ytr)
        str_scores = ridge_score(Xtr, model_r)
        a, b = platt(str_scores, ytr)
        for split_name, split_items in (("train", train), ("test", test)):
            X = np.stack([states[i["item_id"]][f] for i in split_items]).astype(np.float64)
            y = np.array([i["label"] for i in split_items])
            s = ridge_score(X, model_r)
            p = 1.0 / (1.0 + np.exp(-(a * s + b)))
            ba, thr = best_ba(list(p), list(y))
            entry = {
                "layer": int(dict(zip(fracs, layers))[f]),
                "depth": f,
                "n": len(y),
                "auc": auc(list(p), list(y)),
                "best_ba": {"bal_acc": ba, "threshold": thr},
                "ece": ece(list(p), list(y)),
                "no_leak": no_leak_threshold(list(p), list(y)),
                "at_0.5": threshold_stats(list(p), list(y), 0.5),
            }
            results["layers"].setdefault(f, {})[split_name] = entry
    (args.out / f"probe-{args.name}.json").write_text(json.dumps(results, indent=1))
    print(f"=== {args.name} ===")
    for f in fracs:
        e = results["layers"][f]["test"]
        print(f"depth {f:>4} (layer {e['layer']}): test AUC {e['auc']:.3f} "
              f"BA {e['best_ba']['bal_acc']:.3f} ECE {e['ece']:.3f}")
    print("wrote", args.out / f"probe-{args.name}.json")
    return 0


def _layer_map(layers_arg: str, n_layers: int) -> list[tuple[str, int]]:
    out = []
    for token in layers_arg.split(","):
        f = float(token.strip())
        idx = max(1, min(n_layers, int(round(f * n_layers))))
        out.append((f"{f:g}", idx))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
