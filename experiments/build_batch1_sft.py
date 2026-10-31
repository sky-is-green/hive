#!/usr/bin/env python
"""Mix controller/planner replay with placement and critic for batch 1.

Usage::

    python experiments/build_batch1_sft.py --out experiments/data/multirole-pc-v1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "experiments" / "data"
CONTROLLER = DATA / "controller-sft-v3" / "train.jsonl"
PLANNER = DATA / "planner-sft-v1" / "train.jsonl"
PLACEMENT = DATA / "placement-sft-v1" / "train.jsonl"
CRITIC = DATA / "critic-sft-v1" / "train.jsonl"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA / "multirole-pc-v1"))
    parser.add_argument("--controller", type=int, default=1200)
    parser.add_argument("--planner", type=int, default=900)
    parser.add_argument("--placement", type=int, default=1200)
    parser.add_argument("--critic", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=611)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    sources = {
        "controller": (load(CONTROLLER), args.controller),
        "planner": (load(PLANNER), args.planner),
        "placement": (load(PLACEMENT), args.placement),
        "critic": (load(CRITIC), args.critic),
    }
    mixed: list[dict] = []
    counts: dict[str, int] = {}
    for name, (rows, requested) in sources.items():
        take = min(requested, len(rows))
        mixed.extend(rng.sample(rows, take))
        counts[name] = take
    rng.shuffle(mixed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in mixed:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {"rows": len(mixed), "counts": counts, "path": str(path)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
