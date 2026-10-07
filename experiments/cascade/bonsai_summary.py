#!/usr/bin/env python
"""Bonsai full-set summary: selection frontier on all 100 tasks.

The 30-task test split is near-ceiling for Bonsai (single 28/30, oracle 28/30),
so selection headroom is only visible on the full bench.  Fits the in-domain
answer-format head on the 70 train tasks, ranks the 4-sample pools on all 100,
and reports the best-of-k frontier, gated sequential (threshold tuned on
train), oracle, and the J-Bench-head ranker for comparison.

Usage:
    bonsai_summary.py --states .../bonsai-states.npz \
        --jbench .../probe-states-bonsai27b.npz \
        --out .../bonsai-summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from probe_judge import platt, ridge_fit, ridge_score  # noqa: E402
from readout_judge import auc  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit(X, y):
    model = ridge_fit(X, y)
    a, c = platt(ridge_score(X, model), y)
    return model, (a, c)


def score(X, model, plat):
    return sigmoid(plat[0] * ridge_score(X, model) + plat[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states", type=Path, default=R / "bonsai-states.npz")
    ap.add_argument("--jbench", type=Path,
                    default=R / "probe-states-bonsai27b.npz")
    ap.add_argument("--tasks", type=Path, default=CASCADE / "tasks-bench.json")
    ap.add_argument("--out", type=Path, default=R / "bonsai-summary.json")
    args = ap.parse_args()

    z = np.load(args.states, allow_pickle=True)
    tid = [str(t) for t in z["task_id"]]
    kind = [str(k) for k in z["kind"]]
    sample = [int(s) for s in z["sample"]]
    split = [str(s) for s in z["split"]]
    label = np.asarray(z["label"], dtype=np.float64)
    toks = np.asarray(z["tokens"], dtype=np.float64)
    X = z["H"].astype(np.float64)  # answer-format candidates

    sm = [i for i, k in enumerate(kind) if k == "sample"]
    tr = [i for i in sm if split[i] == "train"]
    pools: dict[str, list[int]] = defaultdict(list)
    for i in sm:
        pools[tid[i]].append(i)
    for v in pools.values():
        v.sort(key=lambda i: sample[i])
    pools = {t: v for t, v in pools.items() if len(v) >= 4}

    model, plat = fit(X[tr], label[tr])
    p = score(X, model, plat)

    def frontier(score_vec):
        def best_of(k):
            correct = 0
            tok = 0.0
            for t, rows in pools.items():
                pick = max(rows[:k], key=lambda i: score_vec[i])
                correct += int(label[pick])
                tok += sum(float(toks[i]) for i in rows[:k])
            return {"correct": correct, "n": len(pools),
                    "acc": correct / len(pools), "tokens": tok,
                    "tok_per_correct": tok / max(1, correct)}

        def gated(thr):
            correct = used = 0
            tok = 0.0
            for t, rows in pools.items():
                pick = None
                for i in rows:
                    used += 1
                    tok += float(toks[i])
                    if score_vec[i] >= thr:
                        pick = i
                        break
                if pick is None:
                    pick = max(rows, key=lambda i: score_vec[i])
                correct += int(label[pick])
            return {"correct": correct, "n": len(pools),
                    "acc": correct / len(pools),
                    "mean_samples": round(used / len(pools), 2),
                    "tokens": tok, "tok_per_correct": tok / max(1, correct)}

        tr_tasks = [t for t, rows in pools.items() if split[rows[0]] == "train"]
        best_thr, best_key = 0.5, None
        for step in range(30, 100, 5):
            thr = step / 100
            correct = 0
            used = 0
            for t in tr_tasks:
                for i in pools[t]:
                    used += 1
                    if score_vec[i] >= thr:
                        correct += int(label[i])
                        break
                else:
                    correct += int(label[max(pools[t],
                                             key=lambda i: score_vec[i])])
            key = (correct, -used)
            if best_key is None or key > best_key:
                best_key, best_thr = key, thr
        oracle = sum(1 for t, rows in pools.items()
                     if any(label[i] for i in rows))
        out = {f"best_of_{k}": best_of(k) for k in (1, 2, 3, 4)}
        out["oracle_best_of_4"] = {"correct": oracle, "n": len(pools),
                                   "acc": oracle / len(pools)}
        out["gated_sequential"] = {
            "policy": f"accept first sample with p >= {best_thr} (tuned on "
                      "train), else best by score",
            **gated(best_thr)}
        out["threshold"] = best_thr
        # per-bucket first-sample catch at 0.5
        tasks = {t["id"]: t for t in json.loads(
            args.tasks.read_text())["tasks"]}
        buckets: dict[str, dict] = {}
        for t, rows in pools.items():
            b = tasks[t]["bucket"]
            e = buckets.setdefault(b, {"n": 0, "correct": 0})
            e["n"] += 1
            e["correct"] += int(label[rows[0]])
        out["first_sample_by_bucket"] = buckets
        return out

    result = {"in_domain_answer": frontier(p)}

    if args.jbench.exists():
        zj = np.load(args.jbench, allow_pickle=True)
        Xj = zj["L1"].astype(np.float64)
        yj = np.asarray(zj["labels"], dtype=np.float64)
        sj = [str(s) for s in zj["splits"]]
        jtr = [i for i, s in enumerate(sj) if s == "train"]
        jte = [i for i, s in enumerate(sj) if s == "test"]
        jmodel, jplat = fit(Xj[jtr], yj[jtr])
        pj = score(X, jmodel, jplat)
        result["jbench"] = frontier(pj)
        result["jbench_test_auc"] = auc(
            list(score(Xj[jte], jmodel, jplat)), list(yj[jte]))

    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
