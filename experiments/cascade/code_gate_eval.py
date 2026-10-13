#!/usr/bin/env python3
"""Compare code-subset functional gates from community samples/hard JSONL.

The community harness computes pass metrics inside `phase_eval`, but that path
requires the embedding `states.npz` (in-domain head).  The code-subset gate
skips embeddings, so this reads the same JSONL records (the checkers already
ran at generation time and the `ok` label is stored) and reports the pass
metrics for any prefix, filtered to the code tasks:

  gen   pass@1 (first sample), pass@3 (any sample)
  hard  pass@1 (single sample per task)
  both  answer-emission rate, cap-hit rate, mean tokens

    python code_gate_eval.py --dir DIR --prefix scion-coder \
        --tasks tasks-community-code.json --out code-gate.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(recs: list[dict], cap: int, task_ids: set[str]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if r["task_id"] in task_ids:
            by_task[r["task_id"]].append(r)
    if not by_task:
        return {"n": 0}
    first = [min(v, key=lambda r: r["sample"]) for v in by_task.values()]
    any_ok = [any(r["ok"] for r in v) for v in by_task.values()]
    emitted = [bool((r.get("content") or "").strip()) for r in first]
    hit_cap = [int(r.get("tokens") or 0) >= cap for r in first]
    fin = [i for i, e in enumerate(emitted) if e]
    return {
        "n": len(first),
        "pass@1": sum(r["ok"] for r in first) / len(first),
        "pass@any": sum(any_ok) / len(any_ok),
        "answer_emitted": sum(emitted) / len(emitted),
        "hit_cap": sum(hit_cap) / len(hit_cap),
        "mean_tokens": sum((r.get("tokens") or 0) for r in first) / len(first),
        "pass@1_finished": (sum(first[i]["ok"] for i in fin) / len(fin)) if fin else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--gen-cap", type=int, default=700)
    ap.add_argument("--hard-cap", type=int, default=2048)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = Path(args.dir)
    tasks = json.loads(Path(args.tasks).read_text())["tasks"]
    ids = {t["id"] for t in tasks}
    splits = {t["id"]: t["split"] for t in tasks}

    samples = load_jsonl(d / f"{args.prefix}-samples.jsonl")
    hards = load_jsonl(d / f"{args.prefix}-hard.jsonl")

    res: dict = {"prefix": args.prefix, "tasks": len(ids), "gen": {}, "hard": {}}
    for tag, sel in (("all", ids),
                     ("test", {i for i in ids if splits.get(i) == "test"}),
                     ("train", {i for i in ids if splits.get(i) == "train"})):
        res["gen"][tag] = summarize(samples, args.gen_cap, sel)
        res["hard"][tag] = summarize(hards, args.hard_cap, sel)
        g, h = res["gen"][tag], res["hard"][tag]
        print(f"[{tag:5s}] gen n={g.get('n',0)} pass@1 {100*(g.get('pass@1') or 0):5.1f}% "
              f"pass@any {100*(g.get('pass@any') or 0):5.1f}% emitted "
              f"{100*(g.get('answer_emitted') or 0):5.1f}% cap "
              f"{100*(g.get('hit_cap') or 0):5.1f}% | hard pass@1 "
              f"{100*(h.get('pass@1') or 0):5.1f}% cap {100*(h.get('hit_cap') or 0):5.1f}%")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
