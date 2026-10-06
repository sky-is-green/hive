#!/usr/bin/env python
"""Better selection without retries: combine cheap features to pick the pool winner.

On the existing 4-sample pools, compares rankers on the held-out test tasks:

  p              head score alone
  -len           shortest answer
  agree          canonical-answer majority agreement
  combo          ridge on [p, -len, agree] fitted on train tasks

Reports selection accuracy and the oracle gap closed; selection pays for all
N samples, so tokens are constant across rankers.

Usage:
    selection_rerank.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from checkers import _boxed, _last_number, _normalize  # noqa: E402
from probe_judge import ridge_fit, ridge_score  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"


def canonical(task, text):
    kind = task["checker"]["type"]
    if kind == "number":
        m = re.search(r"final answer[^0-9\-+]*([-+]?\d[\d,]*(?:\.\d+)?)",
                      text, re.I | re.S)
        if m:
            return m.group(1).replace(",", "")
        v = _last_number(text)
        return None if v is None else f"{v:g}"
    if kind == "boxed":
        return _boxed(text)
    if kind == "choice":
        letters = re.findall(r"\b([A-H])\b", text.upper())
        return letters[-1] if letters else None
    if kind == "contains":
        return _normalize(text)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=Path, default=R / "correction-samples.jsonl")
    ap.add_argument("--out", type=Path, default=R / "selection-rerank.json")
    args = ap.parse_args()

    tasks = {t["id"]: t for t in json.loads(
        (CASCADE / "tasks-bench.json").read_text())["tasks"]}
    pools: dict[str, list] = defaultdict(list)
    for line in args.samples.read_text().splitlines():
        rec = json.loads(line)
        pools[rec["task_id"]].append(rec)
    for rows in pools.values():
        rows.sort(key=lambda r: r["sample"])

    feats, labels, splits, tids = [], [], [], []
    for tid, rows in pools.items():
        if len(rows) < 4:
            continue
        canon = [canonical(tasks[tid], r["content"]) for r in rows]
        counts = {}
        for c in canon:
            if c is not None:
                counts[c] = counts.get(c, 0) + 1
        for r, c in zip(rows, canon):
            agree = counts.get(c, 0) / len(rows) if c is not None else 0.0
            feats.append([r["p_correct"], -r["tokens"], agree])
            labels.append(int(r["ok"]))
            tids.append(tid)
            splits.append(tasks[tid]["split"])
    X = np.asarray(feats, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    ptr = [i for i, s in enumerate(splits) if s == "train"]
    pte = [i for i, s in enumerate(splits) if s == "test"]
    model = ridge_fit(X[ptr], y[ptr])
    combo = ridge_score(X, model)

    first: dict[str, int] = {}
    for i, tid in enumerate(tids):
        first.setdefault(tid, i)
    test_tasks = sorted({t for t, s in zip(tids, splits) if s == "test"})
    base = {t: first[t] for t in test_tasks}

    def rank(name, score):
        ok = n = 0
        for t in test_tasks:
            i0 = base[t]
            rows = [(i, score[i]) for i in range(i0, i0 + 4)]
            pick = max(rows, key=lambda x: x[1])[0]
            ok += int(y[pick])
            n += 1
        return {"correct": ok, "n": n, "acc": ok / max(1, n)}

    oracle = {"correct": sum(1 for t in test_tasks if any(
        y[base[t]:base[t] + 4])), "n": len(test_tasks)}
    oracle["acc"] = oracle["correct"] / max(1, oracle["n"])

    out = {
        "p": rank("p", X[:, 0]),
        "neg_len": rank("neg_len", X[:, 1]),
        "agree": rank("agree", X[:, 2]),
        "combo": rank("combo", combo),
        "oracle": oracle,
    }
    for name, m in out.items():
        print(f"{name:8} {m['correct']}/{m['n']} acc {m['acc']:.3f}")
    gap = oracle["acc"] - out["p"]["acc"]
    print(f"oracle-p gap {gap:.3f}; combo closes "
          f"{(out['combo']['acc'] - out['p']['acc']) / max(1e-9, gap):.2f} of it")
    args.out.write_text(json.dumps(out, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
