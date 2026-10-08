#!/usr/bin/env python
"""Majority + head-override ("veto") evaluation on community runs.

Policy: start from the majority canonical answer (same convention as
community_eval's pick_maj), then override to the pool sample with the highest
head p_correct when it beats the majority base by a margin d tuned on the
train tasks.

Variants:
  v1  base = the majority's own pick (first instance of the mode cluster,
      fallback sample 0), exactly pick_maj's selection
  v2  base = the best-p instance within the majority answer cluster

Reports test accuracy, fixes/breaks, exact two-sided McNemar, the full d
sweep (train-tuned), the switch audit, and pool token costs.

Usage:
    veto_eval.py --prefix scion-community --out results/.../veto-scion-community.json
    veto_eval.py --prefix bonsai-community --out results/.../veto-bonsai-community.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from community_eval import canonical, load_jsonl  # noqa: E402
from probe_judge import platt, ridge_fit, ridge_score  # noqa: E402
from readout_judge import auc  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"
DEFAULT_TASKS = CASCADE / "tasks-community.json"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--states", type=Path)
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    ap.add_argument("--head", default="answer", choices=("answer", "trace"))
    ap.add_argument("--head-json", type=Path, default=None,
                    help="frozen head JSON (sidecar); default fits in-domain on train")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    states = args.states or R / f"{args.prefix}-states.npz"
    z = np.load(states, allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    tid = [str(t) for t in z["task_id"]]
    kind = [str(k) for k in z["kind"]]
    split = [str(s) for s in z["split"]]
    sample = [int(s) for s in z["sample"]]
    label = np.asarray(z["label"], dtype=np.float64)
    toks = np.asarray(z["tokens"], dtype=np.float64)
    X = (z["H"] if args.head == "answer" else z["HV"]).astype(np.float64)

    sm = [i for i, k in enumerate(kind) if k == "sample"]
    tr = [i for i in sm if split[i] == "train"]
    te = [i for i in sm if split[i] == "test"]
    if args.head_json is not None:
        # frozen deployed head (the sidecar): score X with its stored weights
        hj = json.loads(args.head_json.read_text())
        hw = np.asarray(hj["weights"]["w"], dtype=np.float64)
        hmu = np.asarray(hj["weights"]["mu"], dtype=np.float64)
        hsd = np.asarray(hj["weights"]["sd"], dtype=np.float64)
        hb = float(hj["weights"]["b"])
        ha, hc = hj["weights"]["platt"]
        p = sigmoid(ha * (((X - hmu) / hsd) @ hw + hb) + hc)
        head_src = str(args.head_json)
    else:
        model = ridge_fit(X[tr], label[tr])
        plat = platt(ridge_score(X[tr], model), label[tr])
        p = sigmoid(plat[0] * ridge_score(X, model) + plat[1])
        head_src = "in-domain refit"

    tasks_by = {t["id"]: t for t in json.loads(args.tasks.read_text())["tasks"]}
    text = {f"{r['task_id']}|sample|{r['sample']}": r["content"]
            for r in load_jsonl(R / f"{args.prefix}-samples.jsonl")}

    pools: dict[str, list[int]] = defaultdict(list)
    for i in sm:
        pools[tid[i]].append(i)
    for v in pools.values():
        v.sort(key=lambda i: sample[i])
    pools = {t: v for t, v in pools.items() if len(v) >= 3}

    def key(i):
        return f"{tid[i]}|sample|{sample[i]}"

    def canon(i):
        return canonical(tasks_by[tid[i]], text[key(i)])

    def maj_pick(t):
        rows = pools[t]
        reps: dict[str, list[int]] = defaultdict(list)
        for i in rows:
            c = canon(i)
            if c is not None:
                reps[c].append(i)
        if reps:
            m = max(reps, key=lambda c: len(reps[c]))
            if len(reps[m]) > 1:
                return reps[m][0]
        return rows[0]

    def veto_pick(t, d, variant):
        rows = pools[t]
        base = maj_pick(t)
        bc = canon(base)
        if bc is not None:
            cluster = [i for i in rows if canon(i) == bc]
        else:
            cluster = [base]
        b = base if variant == "v1" else max(cluster, key=lambda i: p[i])
        best = max(rows, key=lambda i: p[i])
        if best != b and p[best] - p[b] > d:
            return best
        return b

    tr_tasks = [t for t, rows in pools.items() if split[rows[0]] == "train"]
    te_tasks = [t for t, rows in pools.items() if split[rows[0]] == "test"]
    grid = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]

    def acc(tids, d, variant):
        return sum(int(label[veto_pick(t, d, variant)]) for t in tids)

    maj_test = sum(int(label[maj_pick(t)]) for t in te_tasks)
    oracle_test = sum(1 for t in te_tasks if any(label[i] for i in pools[t]))
    out = {
        "prefix": args.prefix, "head": args.head, "head_src": head_src,
        "states": str(states),
        "n_test": len(te_tasks), "n_train": len(tr_tasks),
        "majority_test": maj_test, "oracle_test": oracle_test,
        "within_pool_auc": auc([p[i] for i in sm], [label[i] for i in sm]),
        "pool_tokens_test": sum(float(toks[i]) for t in te_tasks for i in pools[t]),
        "pool_tokens_all": sum(float(toks[i]) for i in sm),
        "sweep": {},
    }
    for variant in ("v1", "v2"):
        tr_correct = {d: acc(tr_tasks, d, variant) for d in grid}
        best_d = max(grid, key=lambda d: (tr_correct[d], -d))
        te_correct = {d: acc(te_tasks, d, variant) for d in grid}
        plateau = [d for d in grid if tr_correct[d] == tr_correct[best_d]]
        fixes = breaks = switches = 0
        for t in te_tasks:
            m = label[maj_pick(t)]
            v = label[veto_pick(t, best_d, variant)]
            if not m and v:
                fixes += 1
            elif m and not v:
                breaks += 1
            if veto_pick(t, best_d, variant) != maj_pick(t):
                switches += 1
        out["sweep"][variant] = {
            "best_d": best_d,
            "train_correct": int(tr_correct[best_d]), "train_n": len(tr_tasks),
            "test_correct": int(te_correct[best_d]),
            "test_correct_range_plateau": [int(min(te_correct[d] for d in plateau)),
                                           int(max(te_correct[d] for d in plateau))],
            "plateau_d": plateau,
            "fixes": fixes, "breaks": breaks, "switches": switches,
            "mcnemar_p": mcnemar_exact(fixes, breaks),
            "per_d": {str(d): {"train": int(tr_correct[d]),
                               "test": int(te_correct[d])} for d in grid},
        }
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "sweep"}, indent=1))
    for v in ("v1", "v2"):
        s = out["sweep"][v]
        print(f"{v}: best_d={s['best_d']} train={s['train_correct']}/"
              f"{s['train_n']} test={s['test_correct']}/{out['n_test']} "
              f"(plateau {s['test_correct_range_plateau']}) "
              f"fixes={s['fixes']} breaks={s['breaks']} "
              f"p={s['mcnemar_p']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
