#!/usr/bin/env python
"""Community-pool slot-logit analysis.

Joins the batch readout (hidden states + answer-slot token probabilities) with
a community run's labels and evaluates:

  - sample-level AUC: raw margin (zero training) vs frozen head vs in-domain
    refit vs blend
  - task-level arms on the held-out split: single, majority, best-of-3 by each
    scorer, oracle
  - majority + head-override ("veto") with each scorer, margin tuned on train

Usage:
    analyze_logit_community.py --prefix scion-community \
        --dir .../logit-scion-community \
        --head-json .../scion-35b-a3b-decision-head.json \
        --out .../logit-scion-community.json
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
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--head-json", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    index = json.loads((args.dir / "prompts" / "index.json").read_text())
    ids = [l.strip() for l in (args.dir / "out" / "ids.txt").read_text().splitlines() if l.strip()]
    assert ids == [e["file"] for e in index], "ids.txt mismatch"
    n = len(index)
    H = np.fromfile(args.dir / "out" / "hidden.bin", dtype=np.float32).reshape(n, -1).astype(np.float64)
    y = np.array([int(e["ok"]) for e in index], dtype=np.float64)
    split = np.array([e["split"] for e in index])
    task = [e["task_id"] for e in index]
    keys = [e["key"] for e in index]

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

    # validation: harness hidden states must match the stored community states
    z = np.load(R / f"{args.prefix}-states.npz", allow_pickle=True)
    kind_z = [str(k) for k in z["kind"]]
    sel = [i for i, k in enumerate(kind_z) if k == "sample"]
    keys_z = [str(z["keys"][i]) for i in sel]
    assert keys_z == keys, "state keys do not match prompt index"
    H_stored = z["H"][sel].astype(np.float64)
    dH = float(np.abs(H - H_stored).max())
    cos = float(np.mean([H[i] @ H_stored[i] /
                         (np.linalg.norm(H[i]) * np.linalg.norm(H_stored[i]) + 1e-12)
                         for i in range(0, n, max(1, n // 50))]))

    hj = json.loads(args.head_json.read_text())
    w = np.asarray(hj["weights"]["w"], dtype=np.float64)
    mu = np.asarray(hj["weights"]["mu"], dtype=np.float64)
    sd = np.asarray(hj["weights"]["sd"], dtype=np.float64)
    b = float(hj["weights"]["b"])
    pa, pc = hj["weights"]["platt"]
    s_frozen = sigmoid(pa * (((H - mu) / sd) @ w + b) + pc)

    tr = np.where(split == "train")[0]
    te = np.where(split == "test")[0]
    model_h = ridge_fit(H[tr], y[tr])
    s_refit = ridge_score(H, model_h)

    # blend on standardized train scores
    zh = (s_refit - s_refit[tr].mean()) / (s_refit[tr].std() + 1e-12)
    zm = (margin - margin[tr].mean()) / (margin[tr].std() + 1e-12)
    best_w, best_auc = 0.5, -1.0
    for i in range(21):
        ww = i / 20.0
        a = auc(list(ww * zh[tr] + (1 - ww) * zm[tr]), list(y[tr]))
        if a is not None and a > best_auc:
            best_auc, best_w = a, ww
    s_blend = best_w * zh + (1 - best_w) * zm

    # clean blend: frozen + margin (neither score is fit on this pool, so the
    # train-tuned weight is not contaminated by in-sample optimism)
    zf = (s_frozen - s_frozen[tr].mean()) / (s_frozen[tr].std() + 1e-12)
    best_wf, best_aucf = 0.5, -1.0
    for i in range(21):
        ww = i / 20.0
        a = auc(list(ww * zf[tr] + (1 - ww) * zm[tr]), list(y[tr]))
        if a is not None and a > best_aucf:
            best_aucf, best_wf = a, ww
    s_fm = best_wf * zf + (1 - best_wf) * zm

    scorers = {"raw": margin, "frozen": s_frozen, "refit": s_refit,
               "blend_refit": s_blend, "blend_fm": s_fm}

    # pools and majority (community_eval convention)
    tasks_by = {t["id"]: t for t in json.loads(args.tasks.read_text())["tasks"]}
    text = {f"{r['task_id']}|sample|{r['sample']}": r["content"]
            for r in load_jsonl(R / f"{args.prefix}-samples.jsonl")}
    pools: dict[str, list[int]] = defaultdict(list)
    for i in range(n):
        pools[task[i]].append(i)
    for v in pools.values():
        v.sort(key=lambda i: index[i]["sample"])
    pools = {t: v for t, v in pools.items() if len(v) >= 3}

    def canon_of(i):
        return canonical(tasks_by[task[i]], text[keys[i]])

    def maj_pick(t):
        rows = pools[t]
        reps: dict[str, list[int]] = defaultdict(list)
        for i in rows:
            c = canon_of(i)
            if c is not None:
                reps[c].append(i)
        if reps:
            m = max(reps, key=lambda c: len(reps[c]))
            if len(reps[m]) > 1:
                return reps[m][0]
        return rows[0]

    tr_tasks = [t for t, rows in pools.items() if split[rows[0]] == "train"]
    te_tasks = [t for t, rows in pools.items() if split[rows[0]] == "test"]

    def arm(tids, pick):
        correct = sum(1 for t in tids if y[pick(t)] > 0.5)
        return {"n": len(tids), "correct": int(correct), "acc": correct / max(1, len(tids))}

    arms = {"single": arm(te_tasks, lambda t: pools[t][0]),
            "majority": arm(te_tasks, maj_pick)}
    for name, s in scorers.items():
        arms[f"best_of_3_{name}"] = arm(te_tasks, lambda t, s=s: max(pools[t], key=lambda i: s[i]))
    arms["oracle"] = {"n": len(te_tasks),
                      "correct": sum(1 for t in te_tasks if any(y[i] > 0.5 for i in pools[t]))}

    def veto_pick(t, d, s):
        base = maj_pick(t)
        best = max(pools[t], key=lambda i: s[i])
        return best if (best != base and s[best] - s[base] > d) else base

    grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    veto = {}
    for name, s in scorers.items():
        zs = (s - s[tr].mean()) / (s[tr].std() + 1e-12)  # train-std units
        tr_c = {d: sum(int(y[veto_pick(t, d, zs)]) for t in tr_tasks) for d in grid}
        best_d = max(grid, key=lambda d: (tr_c[d], -d))
        te_c = sum(int(y[veto_pick(t, best_d, zs)]) for t in te_tasks)
        fixes = sum(1 for t in te_tasks
                    if not y[maj_pick(t)] and y[veto_pick(t, best_d, zs)])
        breaks = sum(1 for t in te_tasks
                     if y[maj_pick(t)] and not y[veto_pick(t, best_d, zs)])
        veto[name] = {"best_d_std": best_d, "train_correct": int(tr_c[best_d]),
                      "test_correct": int(te_c), "fixes": fixes, "breaks": breaks,
                      "mcnemar_p": mcnemar_exact(fixes, breaks),
                      "per_d_test": {str(d): int(sum(int(y[veto_pick(t, d, zs)])
                                                     for t in te_tasks)) for d in grid}}

    res = {
        "prefix": args.prefix, "n": n, "dim": int(H.shape[1]),
        "validation": {"max_abs_dH": dH, "mean_cos_sample": cos},
        "blend_weight_refit": best_w,
        "blend_weight_fm": best_wf,
        "auc_samples": {name: auc(list(s[te]), list(y[te])) for name, s in scorers.items()},
        "auc_all": {name: auc(list(s), list(y)) for name, s in scorers.items()},
        "arms": arms, "veto": veto,
        "mean_pA": float(pA.mean()), "mean_pB": float(pB.mean()),
    }
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
