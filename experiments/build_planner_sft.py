#!/usr/bin/env python
"""Build a planner-only SFT corpus for the controller+planner experiment.

Reuses the planner task families from ``build_role_sft.py``. Every row is
validated with the role-suite checker; the frozen wider planner suite is
excluded so the evaluation stays held out.

Usage::

    python experiments/build_planner_sft.py --out experiments/data/planner-sft-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.build_role_sft import (  # noqa: E402
    Builder,
    planner_drives_pair,
    planner_number,
    planner_ordered,
    planner_plan,
    planner_schedule,
    planner_waves,
)

SUITE = REPO / "experiments" / "roles" / "tasks-planner.json"
COUNTS = {
    "planner_plan": 500,
    "planner_number": 800,
    "planner_ordered": 250,
    "planner_waves": 200,
    "planner_drives_pair": 100,
    "planner_schedule": 250,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "experiments" / "data" / "planner-sft-v1"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    b = Builder(args.seed)
    rng = b.rng
    fns = {
        "planner_plan": planner_plan,
        "planner_number": planner_number,
        "planner_ordered": planner_ordered,
        "planner_waves": planner_waves,
        "planner_drives_pair": planner_drives_pair,
        "planner_schedule": planner_schedule,
    }
    for family, count in COUNTS.items():
        fn = fns[family]
        b.fill(family, count, lambda fn=fn: fn(rng), tries_per_item=30)

    suite = json.loads(SUITE.read_text())
    seen = {task["prompt"] for task in suite["tasks"]}
    rows = [row for row in b.rows if row["prompt"][1]["content"] not in seen]
    b.rng.shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    metas = [json.loads(row["meta"]) for row in rows]
    summary = {
        "rows": len(rows),
        "dropped_suite_overlap": len(b.rows) - len(rows),
        "families": {family: sum(1 for m in metas if m["family"] == family)
                     for family in sorted({m["family"] for m in metas})},
        "failed_checks": b.summary()["failed_checks"],
        "path": str(path),
    }
    (out / "stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
