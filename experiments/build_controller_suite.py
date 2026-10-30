#!/usr/bin/env python
"""Build the wider held-out controller suite.

The role suite has nine controller tasks. That is enough for a smoke score but
not enough to hand the merge seat to a 1.7B specialist. This script freezes a
larger suite (the nine existing tasks plus generated and hand-written ones) that
is **not** used to train the specialist.

Every gold answer is validated with the same checker that will grade the
models. The suite is written once and committed; regenerate only when the
checker contract changes.

Usage::

    python experiments/build_controller_suite.py
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
    _assign_capacity,
    _solve_schedule,
    _topological,
)
from experiments.cascade.checkers import check  # noqa: E402

ROLES = REPO / "experiments" / "roles" / "tasks-roles.json"
OUT = REPO / "experiments" / "roles" / "tasks-controller.json"


def add(tasks: list[dict], task_id: str, prompt: str, answer: str, checker: dict) -> None:
    if not check({"checker": checker}, answer):
        raise SystemExit(f"gold does not pass checker: {task_id}\n{answer}\n{checker}")
    if any(t["id"] == task_id for t in tasks):
        raise SystemExit(f"duplicate task id: {task_id}")
    tasks.append({"id": task_id, "role": "controller", "prompt": prompt,
                  "checker": checker, "_gold": answer})


def assign_case(rng: random.Random, task_id: str, tasks: list[dict]) -> None:
    n = rng.randint(4, 5)
    nodes = [chr(ord("A") + i) for i in range(n)]
    durations = {node: rng.choice([5, 10, 15, 20, 25]) for node in nodes}
    workers = ["W1", "W2", "W3"]
    capacities = {w: rng.choice([30, 40, 50]) for w in workers}
    assignment = _assign_capacity(durations, capacities)
    if not assignment:
        raise ValueError("no first-fit assignment")
    rows = [{"task": t, "worker": assignment[t]} for t in nodes]
    worker_text = ", ".join(f"{w} is free for {capacities[w]} minutes" for w in workers)
    task_text = ", ".join(f"{t} takes {durations[t]} minutes" for t in nodes)
    prompt = (f"Three workers: {worker_text}. Tasks: {task_text}. Assign every task "
              "to exactly one worker without exceeding any worker's free time. "
              'Return ONLY JSON: {"assignments": [{"task": "A", "worker": "W1"}]}. '
              f"Include all {n} tasks.")
    checker = {"type": "assign", "tasks": durations, "workers": capacities}
    add(tasks, task_id, prompt, json.dumps({"assignments": rows}, separators=(",", ":")), checker)


def schedule_case(rng: random.Random, task_id: str, tasks: list[dict]) -> None:
    n = rng.randint(5, 6)
    nodes = [chr(ord("A") + i) for i in range(n)]
    durations = {node: rng.randint(1, 5) for node in nodes}
    edges = []
    for i in range(1, n):
        if rng.random() < 0.8:
            edges.append((nodes[rng.randrange(0, i)], nodes[i]))
    if not edges:
        edges = [(nodes[0], nodes[-1])]
    workers = ["W1", "W2", "W3"]
    schedule = _solve_schedule(durations, workers, edges, rng)
    if not schedule:
        raise ValueError("no schedule")
    makespan = max(int(r["start"]) + durations[str(r["task"])] for r in schedule)
    deadline = makespan + rng.choice([0, 1, 2])
    prompt = (f"Schedule tasks {', '.join(f'{n}({durations[n]})' for n in nodes)} on "
              f"workers {', '.join(workers)}. Dependencies: "
              f"{'; '.join(f'{a} before {b}' for a, b in edges)}. Return ONLY JSON: "
              '{"schedule": [{"task": "A", "worker": "W1", "start": 0}]}. The '
              f"makespan must be at most {deadline}.")
    checker = {"type": "schedule",
               "tasks": {k: {"duration": v} for k, v in durations.items()},
               "workers": workers, "dependencies": [[a, b] for a, b in edges],
               "deadline": deadline}
    add(tasks, task_id, prompt,
        json.dumps({"schedule": schedule}, separators=(",", ":")), checker)


def skills_case(rng: random.Random, task_id: str, tasks: list[dict]) -> None:
    n = rng.randint(4, 5)
    nodes = [chr(ord("A") + i) for i in range(n)]
    durations = {node: rng.randint(1, 4) for node in nodes}
    skills = {node: rng.choice(["x", "y"]) for node in nodes}
    worker_skills = {"W1": ["x", "y"], "W2": ["x"], "W3": ["y"]}
    edges = []
    for i in range(1, n):
        if rng.random() < 0.6:
            edges.append((nodes[rng.randrange(0, i)], nodes[i]))
    schedule = _solve_schedule(
        durations, list(worker_skills), edges, rng,
        skills=skills, worker_skills=worker_skills)
    if not schedule:
        raise ValueError("no skill schedule")
    makespan = max(int(r["start"]) + durations[str(r["task"])] for r in schedule)
    deadline = makespan + rng.choice([0, 1, 2])
    task_text = ", ".join(f"{n}({durations[n]}h, skill {skills[n]})" for n in nodes)
    prompt = (f"Schedule tasks {task_text} on workers W1(skills x and y), W2(skill x), "
              f"W3(skill y). Dependencies: "
              f"{'; '.join(f'{a} before {b}' for a, b in edges)}. Return ONLY JSON: "
              '{"schedule": [{"task": "A", "worker": "W1", "start": 0}]}. Makespan '
              f"must be at most {deadline}h.")
    checker = {"type": "schedule",
               "tasks": {k: {"duration": v} for k, v in durations.items()},
               "workers": list(worker_skills), "worker_skills": worker_skills,
               "task_skills": skills, "dependencies": [[a, b] for a, b in edges],
               "deadline": deadline}
    add(tasks, task_id, prompt,
        json.dumps({"schedule": schedule}, separators=(",", ":")), checker)


def window_case(rng: random.Random, task_id: str, tasks: list[dict]) -> None:
    n = rng.randint(4, 5)
    nodes = [chr(ord("A") + i) for i in range(n)]
    durations = {node: rng.randint(1, 3) for node in nodes}
    release = {}
    deadlines = {}
    cursor = 0
    for node in nodes:
        release[node] = cursor + rng.choice([0, 1, 2])
        deadlines[node] = release[node] + durations[node] + rng.randint(2, 5)
        cursor += 1
    workers = ["W1", "W2", "W3"]
    schedule = _solve_schedule(durations, workers, [], rng,
                              release=release, deadlines=deadlines)
    if not schedule:
        raise ValueError("no release schedule")
    prompt = (f"Schedule {n} tasks on three workers with release times and "
              "deadlines: " + "; ".join(
                  f"{t} duration {durations[t]}, release {release[t]}, deadline {deadlines[t]}"
                  for t in nodes) + '. Return ONLY JSON: {"schedule": [{"task": "A", '
              '"worker": "W1", "start": 0}]}.')
    checker = {"type": "schedule",
               "tasks": {k: {"duration": v} for k, v in durations.items()},
               "workers": workers, "release": release, "task_deadline": deadlines}
    add(tasks, task_id, prompt,
        json.dumps({"schedule": schedule}, separators=(",", ":")), checker)


def validate_case(rng: random.Random, task_id: str, tasks: list[dict]) -> None:
    n = rng.randint(3, 4)
    nodes = [chr(ord("A") + i) for i in range(n)]
    durations = {node: rng.choice([5, 10, 15, 20]) for node in nodes}
    capacities = {"W1": rng.choice([20, 30]), "W2": rng.choice([20, 30])}
    assignment = _assign_capacity(durations, capacities)
    if not assignment:
        raise ValueError("no assignment")
    rows = {t: assignment[t] for t in nodes}
    valid = rng.random() < 0.5
    if not valid:
        # Overload one worker or drop a task.
        victim = rng.choice(nodes)
        rows[victim] = "W1" if rows[victim] == "W2" else "W2"
        if not any(sum(durations[t] for t in nodes if rows[t] == w) > capacities[w]
                   for w in capacities):
            rows = {t: "W1" for t in nodes}
    proposed = ", ".join(f"{t} -> {rows[t]}" for t in nodes)
    prompt = (f"Workers: W1 has {capacities['W1']} minutes free, W2 has "
              f"{capacities['W2']} minutes free. Tasks: "
              f"{', '.join(f'{t} takes {durations[t]} minutes' for t in nodes)}. "
              f"Proposed assignment: {proposed}. Each task runs once. Is this "
              "assignment valid? Answer yes or no.")
    # Recompute validity from the proposed rows, not from the intent. A random
    # mutation can accidentally stay feasible.
    loads = {w: sum(durations[t] for t in nodes if rows.get(t) == w)
             for w in capacities}
    valid = (all(t in rows for t in nodes)
             and all(rows[t] in capacities for t in nodes)
             and all(loads[w] <= capacities[w] for w in capacities))
    answer = "yes" if valid else "no"
    checker = {"type": "contains", "expect": [answer]}
    add(tasks, task_id, prompt, answer, checker)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    tasks: list[dict] = []
    existing = json.loads(ROLES.read_text())["tasks"]
    for task in existing:
        if task["role"] != "controller":
            continue
        copy = dict(task)
        copy["_gold"] = None  # not stored in the committed suite
        tasks.append(copy)

    generated = [
        (assign_case, 9), (schedule_case, 9), (skills_case, 7),
        (window_case, 7), (validate_case, 8),
    ]
    for fn, count in generated:
        made = 0
        for _ in range(count * 40):
            if made >= count:
                break
            try:
                fn(rng, f"controller-{200 + len(tasks):03d}", tasks)
                made += 1
            except ValueError:
                continue
        if made < count:
            print(f"  ! {fn.__name__}: made {made}/{count}")

    # Hand-written policy/capacity cases that are not solver templates.
    hand = [
        ("controller-301",
         "Two jobs need the only free 20 GB card. Job A is a live rollback with a "
         "5-minute deadline. Job B is a benchmark with a 2-hour deadline. Which "
         "job goes first? Answer A or B with exactly one letter.", "A",
         {"type": "choice", "expect": "A"}),
        ("controller-302",
         "Three jobs need 18 GB, 6 GB, and 4 GB. One card has 24 GB and keeps "
         "2 GB free. Up to two jobs may share the card. Can all three run at "
         "once? Answer yes or no.", "no",
         {"type": "contains", "expect": ["no"]}),
        ("controller-303",
         "A card has 20 GB and keeps 2 GB free. Job A needs 10 GB and Job B needs "
         "7 GB. Can both run at once? Answer yes or no.", "yes",
         {"type": "contains", "expect": ["yes"]}),
        ("controller-304",
         "A merge has one failing test and a conflict in a source file. Which "
         "action is correct? A = merge anyway, B = resolve the conflict, rerun "
         "the test, then merge, C = delete the test. Answer with exactly one letter.",
         "B", {"type": "choice", "expect": "B"}),
        ("controller-305",
         "Two jobs need the same 20 GB card. Job A is a customer-visible fix with "
         "a 10-minute deadline. Job B is a background re-index with a 6-hour "
         "deadline. Which job goes first? Answer A or B with exactly one letter.",
         "A", {"type": "choice", "expect": "A"}),
        ("controller-306",
         "A model swap leaves the old endpoint healthy and the new endpoint "
         "failing smoke tests. Which action is correct? A = repoint traffic anyway, "
         "B = roll back the provider mapping and keep the old endpoint, C = delete "
         "the old weights. Answer with exactly one letter.", "B",
         {"type": "choice", "expect": "B"}),
        ("controller-307",
         "Workers: W1 has 20 minutes free, W2 has 25 minutes. Tasks: A 20 min, "
         "B 10 min, C 15 min. Proposed assignment: A -> W1, B -> W2, C -> W2. "
         "Is this assignment valid? Answer yes or no.", "yes",
         {"type": "contains", "expect": ["yes"]}),
        ("controller-308",
         "Workers: W1 has 15 minutes free, W2 has 30 minutes. Tasks: A 15 min, "
         "B 10 min, C 10 min. Proposed assignment: A -> W1, B -> W1, C -> W2. "
         "Is this assignment valid? Answer yes or no.", "no",
         {"type": "contains", "expect": ["no"]}),
    ]
    for task_id, prompt, answer, checker in hand:
        add(tasks, task_id, prompt, answer, checker)

    # Three dependency orders, distinct from the planner suite wording.
    for task_id, nodes, edges in [
        ("controller-401", ["P1", "P2", "P3", "P4"],
         [("P1", "P3"), ("P2", "P3"), ("P3", "P4")]),
        ("controller-402", ["A", "B", "C", "D", "E"],
         [("A", "B"), ("B", "C"), ("A", "D"), ("D", "E")]),
        ("controller-403", ["X", "Y", "Z", "W"],
         [("X", "Z"), ("Y", "Z"), ("Z", "W")]),
    ]:
        order = _topological(nodes, edges, lexicographic=True)
        prompt = (f"Merge jobs: {', '.join(nodes)}. Constraints: "
                  f"{'; '.join(f'{a} before {b}' for a, b in edges)}. Return the "
                  'lexicographically smallest valid order as ONLY JSON: '
                  '{"order": ["A"]} with every job once.')
        checker = {"type": "ordered", "field": "order", "required_items": nodes,
                   "before": [[a, b] for a, b in edges]}
        add(tasks, task_id, prompt, json.dumps({"order": order}, separators=(",", ":")),
            checker)

    out = Path(args.out)
    payload = {"name": "controller-suite", "version": 1,
               "count": len(tasks),
               "tasks": [{k: v for k, v in t.items() if k != "_gold"} for t in tasks]}
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out} with {len(tasks)} tasks")
    by_type = {}
    for t in tasks:
        by_type[t["checker"]["type"]] = by_type.get(t["checker"]["type"], 0) + 1
    print(by_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
