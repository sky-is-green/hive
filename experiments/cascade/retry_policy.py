#!/usr/bin/env python
"""Retry/selection policies on the held-out task split.

Reads `correction-samples.jsonl` (4 samples/task, head scores) and the task
splits.  An accept threshold is chosen on the 70 train tasks and applied
unchanged to the 30 test tasks, so the reported numbers are held-out:

  single            first sample
  best-of-N         top head score among the first N
  retry@thr         sequential: accept the first sample with p >= thr, else the
                    best-scoring sample; tokens = consumed samples only
  two-band          accept if p >= hi; else one more sample; take the better
                    (the accept/retry/reject variant without a reject arm)
  oracle            any sample correct (upper bound of the pool)

Usage:
    retry_policy.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R = REPO / "experiments/cascade/results/jev-readout-20261006"
CASCADE = REPO / "experiments" / "cascade"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=Path, default=R / "correction-samples.jsonl")
    ap.add_argument("--out", type=Path, default=R / "retry-policy.json")
    args = ap.parse_args()

    tasks = {t["id"]: t
             for t in json.loads((CASCADE / "tasks-bench.json").read_text())["tasks"]}
    per_task: dict[str, list] = defaultdict(list)
    for line in args.samples.read_text().splitlines():
        rec = json.loads(line)
        per_task[rec["task_id"]].append(rec)
    for rows in per_task.values():
        rows.sort(key=lambda r: r["sample"])
    usable = {t: rows for t, rows in per_task.items()
              if len(rows) >= 4 and t in tasks}
    train = [t for t in usable if tasks[t]["split"] == "train"]
    test = [t for t in usable if tasks[t]["split"] == "test"]

    def tokens(rows, n):
        return sum(r["tokens"] for r in rows[:n])

    def evaluate(sel_tasks, policy, *policy_args):
        correct, tok, n = 0, 0, 0
        for t in sel_tasks:
            rows = usable[t]
            pick, used = policy(rows, *policy_args)
            n += 1
            correct += int(pick["ok"])
            tok += tokens(rows, used) if used else pick["tokens"]
        return {"n": n, "correct": correct, "tokens": tok,
                "tok_per_correct": tok / max(1, correct)}

    def p_single(rows, *_):
        return rows[0], 1

    def p_bestn(rows, n):
        return max(rows[:n], key=lambda r: r["p_correct"]), n

    def p_retry(rows, thr):
        for i, r in enumerate(rows):
            if r["p_correct"] >= thr:
                return r, i + 1
        return max(rows, key=lambda r: r["p_correct"]), len(rows)

    def p_twoband(rows, hi):
        first = rows[0]
        if first["p_correct"] >= hi:
            return first, 1
        second = max(rows[1:2], key=lambda r: r["p_correct"])
        return (first if first["p_correct"] >= second["p_correct"] else second), 2

    # threshold selection on train
    best_thr, best_key = 0.5, None
    for step in range(30, 96, 5):
        thr = step / 100
        m = evaluate(train, p_retry, thr)
        key = (m["correct"], -m["tokens"])
        if best_key is None or key > best_key:
            best_key, best_thr = key, thr
    best_hi, best_key = 0.5, None
    for step in range(30, 96, 5):
        hi = step / 100
        m = evaluate(train, p_twoband, hi)
        key = (m["correct"], -m["tokens"])
        if best_key is None or key > best_key:
            best_key, best_hi = key, hi

    out = {"n_train": len(train), "n_test": len(test),
           "chosen": {"retry_thr": best_thr, "twoband_hi": best_hi},
           "test": {
               "single": evaluate(test, p_single),
               "best_of_2": evaluate(test, p_bestn, 2),
               "best_of_3": evaluate(test, p_bestn, 3),
               "best_of_4": evaluate(test, p_bestn, 4),
               "retry": evaluate(test, p_retry, best_thr),
               "twoband": evaluate(test, p_twoband, best_hi),
           }}
    oracle = {"n": len(test),
              "correct": sum(1 for t in test if any(r["ok"] for r in usable[t]))}
    oracle["tok_per_correct"] = None
    out["test"]["oracle_best_of_4"] = oracle

    print(f"train {len(train)} tasks, test {len(test)} tasks; "
          f"chosen retry_thr={best_thr}, twoband_hi={best_hi}")
    print(f"{'policy':18} {'correct':>8} {'acc':>6} {'tokens':>8} {'tok/correct':>12}")
    for name, m in out["test"].items():
        if "correct" not in m:
            continue
        acc = m["correct"] / max(1, m["n"])
        tpc = m.get("tok_per_correct")
        tpc = f"{tpc:.0f}" if tpc else "-"
        print(f"{name:18} {m['correct']:>6}/{m['n']:<3} {acc:>6.3f} "
              f"{m.get('tokens', 0):>8} {tpc:>12}")
    args.out.write_text(json.dumps(out, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
