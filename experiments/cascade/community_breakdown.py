#!/usr/bin/env python
"""Test-split, per-benchmark breakdown of the community arms.

The in-domain head is in-sample-perfect on train (1,395 samples, dim 2048), so
only the held-out test tasks are interpretable.  This recomputes the arms on
the test split per benchmark (GSM8K / MATH / HumanEval) for the in-domain and
the frozen J-Bench heads, plus per-sample AUC and the first-sample catch.

Usage:
    community_breakdown.py --prefix scion-community
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from community_eval import canonical, load_jsonl  # noqa: E402
from probe_judge import platt, ridge_fit, ridge_score  # noqa: E402
from readout_judge import auc, best_ba  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--states", type=Path, default=None)
    ap.add_argument("--samples", type=Path, default=None)
    ap.add_argument("--tasks", type=Path,
                    default=CASCADE / "tasks-community.json")
    ap.add_argument("--jbench-states", type=Path,
                    default=R / "probe-states-scion35b.npz")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args.states = args.states or R / f"{args.prefix}-states.npz"
    args.samples = args.samples or R / f"{args.prefix}-samples.jsonl"
    args.out = args.out or R / f"{args.prefix}-breakdown.json"

    z = np.load(args.states, allow_pickle=True)
    tid = [str(t) for t in z["task_id"]]
    kind = [str(k) for k in z["kind"]]
    split = [str(s) for s in z["split"]]
    sample = [int(s) for s in z["sample"]]
    label = np.asarray(z["label"], dtype=np.float64)
    toks = np.asarray(z["tokens"], dtype=np.float64)
    X = z["H"].astype(np.float64)

    tasks = {t["id"]: t for t in json.loads(args.tasks.read_text())["tasks"]}
    sm = [i for i, k in enumerate(kind) if k == "sample"]
    tr = [i for i in sm if split[i] == "train"]
    te = [i for i in sm if split[i] == "test"]
    first = [i for i in sm if sample[i] == 0]

    model = ridge_fit(X[tr], label[tr])
    plat = platt(ridge_score(X[tr], model), label[tr])
    p = sigmoid(plat[0] * ridge_score(X, model) + plat[1])

    zj = np.load(args.jbench_states, allow_pickle=True)
    Xj = zj["L1"].astype(np.float64)
    yj = np.asarray(zj["labels"], dtype=np.float64)
    sj = [str(s) for s in zj["splits"]]
    jtr = [i for i, s in enumerate(sj) if s == "train"]
    jmodel = ridge_fit(Xj[jtr], yj[jtr])
    jplat = platt(ridge_score(Xj[jtr], jmodel), yj[jtr])
    jp = sigmoid(jplat[0] * ridge_score(X, jmodel) + jplat[1])

    pools: dict[str, list[int]] = defaultdict(list)
    for i in sm:
        pools[tid[i]].append(i)
    for v in pools.values():
        v.sort(key=lambda i: sample[i])
    text: dict[str, str] = {}
    for rec in load_jsonl(args.samples):
        text[f"{rec['task_id']}|sample|{rec['sample']}"] = rec["content"]

    tr_tasks = [t for t, rows in pools.items() if split[rows[0]] == "train"]
    te_tasks = [t for t, rows in pools.items() if split[rows[0]] == "test"]

    def gated(rows, thr, score):
        used = 0
        for i in rows:
            used += 1
            if score[i] >= thr:
                return i, used
        return max(rows, key=lambda i: score[i]), used

    thr_in, key = 0.5, None
    for step in range(30, 100, 5):
        t = step / 100
        correct = sum(int(label[gated(pools[k], t, p)[0]]) for k in tr_tasks)
        used = sum(gated(pools[k], t, p)[1] for k in tr_tasks)
        if key is None or (correct, -used) > key:
            key, thr_in = (correct, -used), t
    thr_jb, key = 0.5, None
    for step in range(30, 100, 5):
        t = step / 100
        correct = sum(int(label[gated(pools[k], t, jp)[0]]) for k in tr_tasks)
        used = sum(gated(pools[k], t, jp)[1] for k in tr_tasks)
        if key is None or (correct, -used) > key:
            key, thr_jb = (correct, -used), t

    def majority_pick(t):
        canon = [canonical(tasks[t], text[keys_i]) for keys_i in
                 [f"{t}|sample|{s}" for s in range(len(pools[t]))]]
        reps: dict[str, int] = {}
        for c in canon:
            if c is not None:
                reps[c] = reps.get(c, 0) + 1
        if reps:
            best = max(reps, key=lambda c: reps[c])
            if reps[best] > 1:
                j = canon.index(best)
                return pools[t][j]
        return pools[t][0]

    def block(tids, score, thr):
        out = {"n": len(tids), "A": 0, "majority": 0, "C": 0, "D": 0,
               "oracle": 0, "C_tokens": 0.0, "D_tokens": 0.0, "A_tokens": 0.0}
        for t in tids:
            rows = pools[t]
            out["A"] += int(label[rows[0]])
            out["A_tokens"] += float(toks[rows[0]])
            out["majority"] += int(label[majority_pick(t)])
            pick, used = gated(rows, thr, score)
            out["C"] += int(label[pick])
            out["C_tokens"] += sum(float(toks[j]) for j in rows[:used])
            dpick = max(rows, key=lambda j: score[j])
            out["D"] += int(label[dpick])
            out["D_tokens"] += sum(float(toks[j]) for j in rows)
            out["oracle"] += int(any(label[j] for j in rows))
        return out

    out: dict = {"prefix": args.prefix, "threshold_in": thr_in,
                 "threshold_jbench": thr_jb, "by_benchmark": {},
                 "test_all": {}}
    benches = sorted({tasks[t]["benchmark"] for t in te_tasks})
    for b in benches + ["ALL"]:
        tids = [t for t in te_tasks if b == "ALL" or tasks[t]["benchmark"] == b]
        out["by_benchmark"][b] = {
            "in_domain": block(tids, p, thr_in),
            "jbench": block(tids, jp, thr_jb),
        }

    def catch(score, tids):
        errs = [i for i in tids if label[i] == 0]
        oks = [i for i in tids if label[i] == 1]
        caught = sum(1 for i in errs if score[i] < 0.5)
        kept = sum(1 for i in oks if score[i] >= 0.5)
        return {"errors": len(errs), "caught": caught,
                "correct": len(oks), "kept": kept,
                "auc": auc(list(score[tids]), list(label[tids]))}

    te_first = [i for i in first if split[i] == "test"]
    out["test_all"] = {
        "first_catch": {"in_domain": catch(p, te_first),
                        "jbench": catch(jp, te_first)},
        "all_samples_auc": {
            "in_domain": auc(list(p[te]), list(label[te])),
            "jbench": auc(list(jp[te]), list(label[te])),
        },
        "gated_thresholds": {"in_domain": thr_in, "jbench": thr_jb},
    }
    for b in benches:
        idx = [i for i in te if tasks[tid[i]]["benchmark"] == b]
        out["by_benchmark"][b]["catch"] = {
            "in_domain": catch(p, idx), "jbench": catch(jp, idx)}

    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
