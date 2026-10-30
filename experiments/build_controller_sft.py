#!/usr/bin/env python
"""Build a controller-only SFT corpus for the 1.7B specialist.

Hyper-specialisation means one output contract and one family of reasoning:
capacity, first-fit assignment, dependency schedules, skills, release windows,
feasibility, priority and merge policy. No planner/router/judge rows.

Starting from the v3 generalist, this corpus is meant to fix the remaining
controller failure (a feasible assignment that is not the benchmark's first-fit
set) without diluting the model across other roles.

Every emitted row is validated with the role-suite checker. The frozen wider
controller suite (`experiments/roles/tasks-controller.json`) is excluded so the
specialist is scored on prompts it never saw.

Usage::

    python experiments/build_controller_sft.py --out experiments/data/controller-sft-v1
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.build_role_sft import (  # noqa: E402
    Builder,
    _assign_capacity,
    controller_assign,
    controller_exact,
    controller_feasible,
    controller_merge,
    controller_priority,
    controller_schedule,
    controller_window,
)
SUITE = REPO / "experiments" / "roles" / "tasks-controller.json"
COUNTS = {
    "controller_exact": 700,
    "controller_assign": 420,
    "controller_schedule": 420,
    "controller_window": 280,
    "controller_validate": 320,
    "controller_capacity": 140,
    "controller_feasible": 80,
    "controller_priority": 20,
    "controller_merge": 20,
}


def controller_validate(rng: random.Random):
    n = rng.randint(3, 4)
    nodes = [chr(ord("A") + i) for i in range(n)]
    durations = {node: rng.choice([5, 10, 15, 20]) for node in nodes}
    capacities = {"W1": rng.choice([20, 25, 30]), "W2": rng.choice([20, 25, 30])}
    assignment = _assign_capacity(durations, capacities)
    if not assignment:
        raise ValueError("no assignment")

    def valid_rows(rows: dict[str, str]) -> bool:
        if not all(t in rows for t in nodes):
            return False
        if not all(rows[t] in capacities for t in nodes):
            return False
        loads = {w: sum(durations[t] for t in nodes if rows[t] == w) for w in capacities}
        return all(loads[w] <= capacities[w] for w in capacities)

    desired = rng.random() < 0.5
    rows = {t: assignment[t] for t in nodes}
    if desired:
        if not valid_rows(rows):
            raise ValueError("first-fit not valid")
    else:
        for _ in range(30):
            rows = {t: assignment[t] for t in nodes}
            victim = rng.choice(nodes)
            other = "W2" if rows[victim] == "W1" else "W1"
            rows[victim] = other
            if not valid_rows(rows):
                break
        else:
            # Force an overload when every single move stays feasible.
            rows = {t: "W1" for t in nodes}
            if valid_rows(rows):
                raise ValueError("could not make invalid")

    proposed = ", ".join(f"{t} -> {rows[t]}" for t in nodes)
    prompt = (f"Workers: W1 has {capacities['W1']} minutes free, W2 has "
              f"{capacities['W2']} minutes free. Tasks: "
              f"{', '.join(f'{t} takes {durations[t]} minutes' for t in nodes)}. "
              f"Proposed assignment: {proposed}. Each task runs once. Is this "
              "assignment valid? Answer yes or no.")
    answer = "yes" if desired else "no"
    return ("controller", "controller-validate", prompt, answer,
            {"type": "contains", "expect": [answer]})


def controller_capacity(rng: random.Random):
    jobs = [rng.choice([4, 6, 8, 10, 12]) for _ in range(rng.choice([2, 3]))]
    cards = rng.choice([[20], [20, 16], [24]])
    reserve = 2
    usable = [c - reserve for c in cards]
    # at most two jobs per card; brute force for the tiny instances
    feasible = False
    for assignment in itertools.product(range(len(cards)), repeat=len(jobs)):
        counts = [assignment.count(i) for i in range(len(cards))]
        if any(c > 2 for c in counts):
            continue
        loads = [sum(j for j, a in zip(jobs, assignment) if a == i)
                 for i in range(len(cards))]
        if all(l <= u for l, u in zip(loads, usable)):
            feasible = True
            break
    desired = rng.random() < 0.5
    if feasible != desired:
        raise ValueError("capacity class mismatch")
    answer = "yes" if feasible else "no"
    card_text = " and ".join(f"{c} GB" for c in cards)
    job_text = ", ".join(f"{chr(ord('A') + i)} needs {j} GB" for i, j in enumerate(jobs))
    prompt = (f"{'One card has' if len(cards) == 1 else 'Two cards have'} {card_text}, "
              f"and each card keeps {reserve} GB free. Jobs: {job_text}. Up to two "
              f"jobs may share a card. Can all {len(jobs)} jobs run at once? Answer "
              "yes or no.")
    return ("controller", "controller-capacity", prompt, answer,
            {"type": "contains", "expect": [answer]})


def build(seed: int = 1337) -> tuple[Builder, list[dict], int]:
    b = Builder(seed)
    rng = b.rng
    fns = {
        "controller_exact": controller_exact,
        "controller_assign": controller_assign,
        "controller_schedule": controller_schedule,
        "controller_window": controller_window,
        "controller_validate": controller_validate,
        "controller_capacity": controller_capacity,
        "controller_feasible": controller_feasible,
        "controller_priority": controller_priority,
        "controller_merge": controller_merge,
    }
    for family, count in COUNTS.items():
        fn = fns[family]
        b.fill(family, count, lambda fn=fn: fn(rng), tries_per_item=20)

    suite = json.loads(SUITE.read_text())
    seen = {task["prompt"] for task in suite["tasks"]}
    rows = [row for row in b.rows if row["prompt"][1]["content"] not in seen]
    return b, rows, len(b.rows) - len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "experiments" / "data" / "controller-sft-v1"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    builder, rows, dropped = build(args.seed)
    builder.rng.shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = builder.summary()
    metas = [json.loads(row["meta"]) for row in rows]
    summary["rows"] = len(rows)
    summary["roles"] = dict(sorted(
        {role: sum(1 for m in metas if m["role"] == role)
         for role in {m["role"] for m in metas}}.items()))
    summary["families"] = dict(sorted(
        {family: sum(1 for m in metas if m["family"] == family)
         for family in {m["family"] for m in metas}}.items()))
    summary["dropped_suite_overlap"] = dropped
    summary["source"] = str(SUITE)
    summary["path"] = str(path)
    (out / "stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
