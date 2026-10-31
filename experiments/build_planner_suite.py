#!/usr/bin/env python
"""Build a wider held-out planner suite.

The role suite has 17 planner tasks. This freezes a larger suite (the 17
existing tasks plus generated planner instances) for the controller+planner
multi-role experiment. Every gold answer is validated with the real checker.

Usage::

    python experiments/build_planner_suite.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.build_role_sft import (  # noqa: E402
    planner_drives_pair,
    planner_number,
    planner_ordered,
    planner_plan,
    planner_schedule,
    planner_waves,
)
from experiments.cascade.checkers import check  # noqa: E402

ROLES = REPO / "experiments" / "roles" / "tasks-roles.json"
OUT = REPO / "experiments" / "roles" / "tasks-planner.json"


def add(tasks: list[dict], task_id: str, prompt: str, answer: str, checker: dict) -> None:
    if any(t["id"] == task_id or t["prompt"] == prompt for t in tasks):
        raise ValueError("duplicate task")
    if not check({"checker": checker}, answer):
        raise SystemExit(f"gold does not pass checker: {task_id}\n{answer}\n{checker}")
    tasks.append({"id": task_id, "role": "planner", "prompt": prompt,
                  "checker": checker})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    tasks: list[dict] = []
    existing = json.loads(ROLES.read_text())["tasks"]
    for task in existing:
        if task["role"] == "planner":
            tasks.append({k: v for k, v in task.items()})

    generated = [
        (planner_plan, 8), (planner_number, 20), (planner_ordered, 8),
        (planner_waves, 8), (planner_drives_pair, 6), (planner_schedule, 10),
    ]
    next_id = 301
    for fn, count in generated:
        made = 0
        for _ in range(count * 60):
            if made >= count:
                break
            try:
                role, family, prompt, answer, checker = fn(rng)
            except Exception:  # noqa: BLE001 - infeasible instance, retry
                continue
            try:
                add(tasks, f"planner-{next_id:03d}", prompt, answer, checker)
            except ValueError:
                continue
            next_id += 1
            made += 1
        if made < count:
            print(f"  ! {fn.__name__}: made {made}/{count}")

    payload = {"name": "planner-suite", "version": 1, "count": len(tasks),
               "tasks": tasks}
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out} with {len(tasks)} tasks")
    by_type: dict[str, int] = {}
    for task in tasks:
        key = task["checker"]["type"]
        by_type[key] = by_type.get(key, 0) + 1
    print(by_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
