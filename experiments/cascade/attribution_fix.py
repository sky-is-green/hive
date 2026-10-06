#!/usr/bin/env python
"""Mitigation check: does an attribution-balanced refit remove the delta?

Baseline head: fitted on train tasks with the "own" attribution only.
Balanced head: fitted on both attributions of the same train candidates.
Both are evaluated on held-out (test-split) tasks, where every candidate is
presented under both attributions.  If the balanced head's own-minus-other
delta shrinks while AUC holds, the bias is correctable by construction.

Usage:
    attribution_fix.py --states .../attribution-states.npz \
        --audit .../attribution-audit.json
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

from probe_judge import ridge_fit, ridge_score  # noqa: E402
from readout_judge import auc  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states", type=Path, default=R / "attribution-states.npz")
    ap.add_argument("--audit", type=Path, default=R / "attribution-audit.json")
    ap.add_argument("--out", type=Path, default=R / "attribution-fix.json")
    args = ap.parse_args()

    s = np.load(args.states, allow_pickle=True)
    keys = [str(k) for k in s["keys"]]
    X = s["emb"].astype(np.float64)
    audit = json.loads(args.audit.read_text())
    rows = {f"{r['task_id']}:{r['source']}": r for r in audit["rows"]}
    splits = {t["id"]: t["split"]
              for t in json.loads((CASCADE / "tasks-bench.json").read_text())["tasks"]}

    base_idx: dict[tuple, dict] = defaultdict(dict)
    for i, k in enumerate(keys):
        task, source, attrib = k.rsplit(":", 2)
        base_idx[(task, source)][attrib] = i
    key_row = {k: rows[f"{k[0]}:{k[1]}"] for k in base_idx}

    train_bases = [k for k in base_idx if splits.get(k[0]) == "train"]
    test_bases = [k for k in base_idx if splits.get(k[0]) == "test"]

    def fit(indices: list[int]):
        y = np.array([key_row[keys[i].rsplit(":", 2)[0] + ":" + keys[i].rsplit(":", 2)[1]]["label"]
                      for i in indices], dtype=np.float64)
        return ridge_fit(X[indices], y)

    def report(model, tag):
        sc = ridge_score(X, model)
        deltas: dict[int, list] = {0: [], 1: []}
        for k in test_bases:
            lab = key_row[k]["label"]
            deltas[lab].append(float(sc[base_idx[k]["own"]] - sc[base_idx[k]["other"]]))
        out = {"delta_correct": float(np.mean(deltas[1])),
               "delta_wrong": float(np.mean(deltas[0])),
               "n_correct": len(deltas[1]), "n_wrong": len(deltas[0])}
        for cond in ("own", "other"):
            idx = [base_idx[k][cond] for k in test_bases]
            y = [key_row[k]["label"] for k in test_bases]
            out[f"auc_{cond}"] = auc(list(sc[idx]), y)
        print(f"{tag}: {json.dumps(out, indent=1)}")
        return out

    own_train = [base_idx[k]["own"] for k in train_bases]
    all_train = [base_idx[k][a] for k in train_bases for a in ("own", "other")]
    out = {
        "n_train_bases": len(train_bases), "n_test_bases": len(test_bases),
        "baseline_own_only": report(fit(own_train), "baseline (own only)"),
        "balanced": report(fit(all_train), "balanced (both attributions)"),
    }
    args.out.write_text(json.dumps(out, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
