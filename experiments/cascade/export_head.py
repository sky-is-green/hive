#!/usr/bin/env python
"""Export a fitted decision head for a llama-server GGUF model.

The head is a closed-form ridge on the last-prompt-token hidden state
(`/embedding`, `--pooling last`, `--embd-normalize -1`), plus a Platt
calibration pair.  The JSON is self-describing: model, prompt format, dims,
training provenance, weights, and test metrics.

Usage:
    export_head.py --states probe-states-scion35b.npz \
        --model SkyIsNotGreen/Scion-35B-A3B \
        --out scion-35b-a3b-decision-head.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from probe_judge import platt, ridge_fit, ridge_score  # noqa: E402
from readout_judge import auc, best_ba  # noqa: E402

R = REPO / "experiments/cascade/results/jev-readout-20261006"
PROMPT = ("system: You are a decision function. Reply with the answer label only.\n"
          "user: Question: {task}\nCandidate answer: {candidate}\n"
          "Is the candidate answer correct?\nA. Yes\nB. No\n"
          "assistant (generation prompt, thinking disabled)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states", type=Path, default=R / "probe-states-scion35b.npz")
    ap.add_argument("--out", type=Path,
                    default=R / "scion-35b-a3b-decision-head.json")
    ap.add_argument("--model", default="SkyIsNotGreen/Scion-35B-A3B")
    ap.add_argument("--name", default="scion-35b-a3b-decision-head")
    ap.add_argument("--runtime",
                    default="prism-ml-llama.cpp fork (moe-corr-runtime), "
                            "llama-server --embeddings --pooling last "
                            "--embd-normalize -1")
    ap.add_argument("--jbench", type=Path, default=R / "judge-bench.manifest.json")
    args = ap.parse_args()

    z = np.load(args.states, allow_pickle=True)
    X = z["L1"].astype(np.float64)
    labels = np.asarray(z["labels"], dtype=np.float64)
    splits = [str(s) for s in z["splits"]]
    tr = [i for i, s in enumerate(splits) if s == "train"]
    te = [i for i, s in enumerate(splits) if s == "test"]

    model = ridge_fit(X[tr], labels[tr])
    w, (mu, sd, b) = model
    a, c = platt(ridge_score(X[tr], model), labels[tr])
    s_te = ridge_score(X[te], model)
    p_te = 1.0 / (1.0 + np.exp(-(a * s_te + c)))
    y_te = labels[te]
    ba, thr = best_ba(list(p_te), list(y_te))
    jb = json.loads(args.jbench.read_text())
    head = {
        "name": args.name,
        "model": args.model,
        "runtime": args.runtime,
        "readout": {"kind": "ridge-to-labels", "position": "last prompt token",
                    "layer": "final", "dim": int(X.shape[1]),
                    "calibration": "Platt on train scores"},
        "prompt": PROMPT,
        "training": {"bench_sha256": jb.get("sha256"),
                     "bench_items": jb.get("items"),
                     "n_train": len(tr), "n_pos": int(labels[tr].sum()),
                     "n_neg": int(len(tr) - labels[tr].sum())},
        "weights": {"w": [round(float(v), 8) for v in w],
                    "mu": [round(float(v), 6) for v in mu],
                    "sd": [round(float(v), 6) for v in sd],
                    "b": float(b), "platt": [float(a), float(c)]},
        "test": {"n": len(te), "auc": auc(list(p_te), list(y_te)),
                 "best_ba": ba, "threshold": thr},
        "apply": ("p_correct = sigmoid(a*(ridge((h-mu)/sd)+b)+c) on the last "
                  "prompt-token hidden state from /embedding; threshold at "
                  "`threshold` for accept/reject"),
    }
    args.out.write_text(json.dumps(head))
    print(json.dumps({k: head[k] for k in ("name", "model", "test")}, indent=1))
    print("wrote", args.out, f"({args.out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
