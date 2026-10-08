#!/usr/bin/env python
"""Slot-logit pilot analysis: raw answer-slot signal vs the fitted head.

Inputs (from the batch harness):
  <dir>/prompts/index.json     prompt ids, keys, ok labels, splits, tokens
  <dir>/out/hidden.bin         pooled last-token states, one row per prompt
  <dir>/out/ids.txt            ids in row order
  <dir>/readout.tsv            id <TAB> n_tokens <TAB> tok=prob ...

Checks:
  - hidden states reproduce the frozen head's stored scores (validation)
  - zero-training baseline: model's own p(A)-vs-p(B) margin at the answer slot
  - hidden-only refit vs stack (blend of hidden score and raw margin)
  - best-of-4 selection arms on the held-out task split

Usage:
    analyze_logit_pilot.py --dir results/.../logit-pilot-scion \
        --samples results/.../correction-samples.jsonl \
        --head-json results/.../scion-35b-a3b-decision-head.json \
        --out results/.../logit-pilot-scion.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from probe_judge import ridge_fit, ridge_score  # noqa: E402
from readout_judge import auc  # noqa: E402


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def zscore(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (x - mu) / (sd + 1e-12)


def best_blend(s_h: np.ndarray, s_m: np.ndarray, y: np.ndarray):
    """Grid-search blend weight on standardized scores; return w and curves."""
    zh = (s_h - s_h.mean()) / (s_h.std() + 1e-12)
    zm = (s_m - s_m.mean()) / (s_m.std() + 1e-12)
    best = (-1.0, 0.0)
    curve = []
    for w in (i / 20.0 for i in range(21)):
        a = auc(list(w * zh + (1 - w) * zm), list(y))
        curve.append((round(w, 2), a))
        if a is not None and a > best[0]:
            best = (a, w)
    return best[1], curve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--head-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    index = json.loads((args.dir / "prompts" / "index.json").read_text())
    ids = [l.strip() for l in (args.dir / "out" / "ids.txt").read_text().splitlines() if l.strip()]
    assert ids == [e["file"] for e in index], "ids.txt does not match index order"

    n = len(index)
    raw = np.fromfile(args.dir / "out" / "hidden.bin", dtype=np.float32)
    d = raw.size // n
    H = raw.reshape(n, d).astype(np.float64)
    y = np.array([int(e["ok"]) for e in index], dtype=np.float64)
    split = np.array([e["split"] for e in index])
    task = [e["task_id"] for e in index]

    probs = {}
    for line in (args.dir / "readout.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        probs[parts[0]] = dict(p.split("=") for p in parts[2:])

    pA = np.array([max(float(probs[e["file"]].get("A", 0.0)),
                       float(probs[e["file"]].get(" A", 0.0))) for e in index])
    pB = np.array([max(float(probs[e["file"]].get("B", 0.0)),
                       float(probs[e["file"]].get(" B", 0.0))) for e in index])
    margin = np.log(pA + 1e-12) - np.log(pB + 1e-12)

    # validation: reproduce the frozen head scores from the dumped states
    hj = json.loads(args.head_json.read_text())
    w = np.asarray(hj["weights"]["w"], dtype=np.float64)
    mu = np.asarray(hj["weights"]["mu"], dtype=np.float64)
    sd = np.asarray(hj["weights"]["sd"], dtype=np.float64)
    b = float(hj["weights"]["b"])
    pa, pc = hj["weights"]["platt"]
    score_frozen = ((H - mu) / sd) @ w + b
    p_frozen = sigmoid(pa * score_frozen + pc)
    stored = {}
    for line in args.samples.read_text().splitlines():
        rec = json.loads(line)
        stored[f"{rec['task_id']}|sample|{rec['sample']}"] = rec
    keys = [e["key"] for e in index]
    dscore = max(abs(score_frozen[i] - stored[k]["head_score"]) for i, k in enumerate(keys))
    dp = max(abs(p_frozen[i] - stored[k]["p_correct"]) for i, k in enumerate(keys))

    tr = np.where(split == "train")[0]
    te = np.where(split == "test")[0]
    model_h = ridge_fit(H[tr], y[tr])
    s_h = ridge_score(H, model_h)
    w_blend, curve = best_blend(s_h[tr], margin[tr], y[tr])

    def arm(tids, pick):
        correct = sum(1 for t in tids if y[pick(t)] > 0.5)
        return {"n": len(tids), "correct": int(correct),
                "acc": correct / max(1, len(tids))}

    te_tasks = sorted(set(task[i] for i in te))
    pools = {t: [i for i in te if task[i] == t] for t in te_tasks}
    pools = {t: v for t, v in pools.items() if len(v) >= 2}

    def picked(scores, t):
        return max(pools[t], key=lambda i: scores[i])

    zh = zscore(s_h, s_h[tr].mean(), s_h[tr].std())
    zm = zscore(margin, margin[tr].mean(), margin[tr].std())
    s_blend = w_blend * zh + (1 - w_blend) * zm

    test_pool = [i for t in pools for i in pools[t]]
    res = {
        "n": n, "dim": int(d), "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_test_tasks": len(pools),
        "validation": {"max_abs_dscore": float(dscore), "max_abs_dp": float(dp)},
        "auc_test": {
            "raw_margin": auc(list(margin[te]), list(y[te])),
            "frozen_head": auc(list(p_frozen[te]), list(y[te])),
            "refit_hidden": auc(list(s_h[te]), list(y[te])),
            "blend": auc(list(s_blend[te]), list(y[te])),
            "blend_weight": w_blend,
        },
        "auc_test_pool": {
            "raw_margin": auc(list(margin[test_pool]), list(y[test_pool])),
            "frozen_head": auc(list(p_frozen[test_pool]), list(y[test_pool])),
            "refit_hidden": auc(list(s_h[test_pool]), list(y[test_pool])),
            "blend": auc(list(s_blend[test_pool]), list(y[test_pool])),
        },
        "arms_test_tasks": {
            "single": arm(te_tasks, lambda t: pools[t][0]),
            "best_of_4_frozen": arm(te_tasks, lambda t: picked(p_frozen, t)),
            "best_of_4_refit": arm(te_tasks, lambda t: picked(s_h, t)),
            "best_of_4_raw": arm(te_tasks, lambda t: picked(margin, t)),
            "best_of_4_blend": arm(te_tasks, lambda t: picked(s_blend, t)),
            "oracle": {"n": len(te_tasks),
                       "correct": sum(1 for t in te_tasks if any(y[i] > 0.5 for i in pools[t]))},
        },
        "mean_pA": float(pA.mean()), "mean_pB": float(pB.mean()),
    }
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    # sanity: does the raw margin even rank single samples on all data?
    print("auc all (raw margin):", round(auc(list(margin), list(y)), 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
