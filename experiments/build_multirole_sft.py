#!/usr/bin/env python
"""Mix controller and planner rows for the two-role retention experiment.

Balanced 1:1 sample so a controller drop is interference, not a lack of
controller replay.

Usage::

    python experiments/build_multirole_sft.py \
      --out experiments/data/multirole-cp-v1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONTROLLER = REPO / "experiments" / "data" / "controller-sft-v3" / "train.jsonl"
PLANNER = REPO / "experiments" / "data" / "planner-sft-v1" / "train.jsonl"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "experiments" / "data" / "multirole-cp-v1"))
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args(argv)

    controller = load(CONTROLLER)
    planner = load(PLANNER)
    n = min(len(controller), len(planner))
    rng = random.Random(args.seed)
    mixed = rng.sample(controller, n) + rng.sample(planner, n)
    rng.shuffle(mixed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in mixed:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {
        "rows": len(mixed),
        "controller_rows": n,
        "planner_rows": n,
        "controller_available": len(controller),
        "planner_available": len(planner),
        "path": str(path),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
