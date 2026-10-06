#!/usr/bin/env python
"""Score an existing independent judge family on the frozen J-Bench.

Reuses the cascade's own judge wrappers (`experiments/cascade/decision.py`)
so the numbers are directly comparable with the rest of the cascade record.
Outputs `judge-<name>.json` in the same schema as the readout runs.

Usage:
    score_judges.py --family tinyjev --model lostargon/Tiny-Jev-1.7B \
        --name tinyjev-1.7b --device cpu
    score_judges.py --family intern --model internlm/Intern-Decision-4B \
        --name intern-4b --device cuda:0
    score_judges.py --family jev --model autotrust/JEV-9B --name jev-9b \
        --device auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.decision import (  # noqa: E402
    InternDecisionJudge,
    Jev9BJudge,
    TinyJevJudge,
)
from experiments.cascade.readout_judge import metrics  # noqa: E402

DEFAULT_BENCH = REPO / "experiments/cascade/results/jev-readout-20261006/judge-bench.jsonl"
DEFAULT_OUT = REPO / "experiments/cascade/results/jev-readout-20261006"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--family", required=True,
                    choices=("tinyjev", "intern", "jev"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="test",
                    choices=("train", "test", "all"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = [json.loads(line) for line in args.bench.read_text().splitlines()]
    if args.split != "all":
        items = [i for i in items if i["split"] == args.split]
    if args.limit:
        items = items[:args.limit]

    cls = {"tinyjev": TinyJevJudge, "intern": InternDecisionJudge,
           "jev": Jev9BJudge}[args.family]
    print(f"loading {args.model} ({args.family}, {args.device}) ...",
          file=sys.stderr)
    judge = cls(args.model, device=args.device)

    out_items = []
    t0 = time.time()
    for n, item in enumerate(items, 1):
        task = {"prompt": item["prompt"], "bucket": item["bucket"]}
        p, ms = judge.verdict(task, item["candidate"])
        out_items.append({**item, "p_correct": p, "judge_ms": ms})
        if n % 10 == 0 or n == len(items):
            print(f"[{n}/{len(items)}] {time.time()-t0:.0f}s", flush=True)

    result = {"name": args.name, "family": args.family, "backend": args.model,
              "split": args.split, "items": out_items,
              "metrics": metrics(out_items)}
    by_bucket: dict[str, list] = {}
    for i in out_items:
        by_bucket.setdefault(i["bucket"], []).append(i)
    result["metrics_by_bucket"] = {b: metrics(v) for b, v in by_bucket.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"judge-{args.name}.json"
    path.write_text(json.dumps(result, indent=1))
    m = result["metrics"]
    print(f"=== {args.name} ({args.split}, n={m['n']}) ===")
    print(f"AUC {m['auc']:.3f} ECE {m['ece']:.3f} "
          f"@0.5 acc {m['at_0.5']['correct']/m['n']:.3f} "
          f"FA {m['at_0.5']['false_accepts']} FR {m['at_0.5']['false_rejects']}")
    if m["no_leak"]:
        nl = m["no_leak"]
        print(f"no-leak thr {nl['threshold']:.2f}: acc {nl['correct']/nl['n']:.3f}")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
