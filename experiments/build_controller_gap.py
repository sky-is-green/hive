#!/usr/bin/env python
"""Add targeted gap rows to the controller corpus.

The base controller specialist (v2) scores 52/60 on the wider controller suite.
Its remaining misses are capacity arithmetic and tight schedules:

- assignments where A+B exceed W1 (the model reports the overload but does not
  move a task);
- validation of a proposed assignment that is exactly at capacity or just over;
- six-task schedules with a tight makespan.

This script generates those families, validates each gold answer, and combines
them with the v2 corpus so the specialist keeps the rest of its behavior.

Usage::

    python experiments/build_controller_gap.py
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

from experiments.build_role_sft import _assign_capacity, _solve_schedule  # noqa: E402
from experiments.cascade.checkers import check  # noqa: E402

V2 = REPO / "experiments" / "data" / "controller-sft-v2" / "train.jsonl"
SUITE = REPO / "experiments" / "roles" / "tasks-controller.json"


def pack_solver(tasks: dict[str, int], caps: dict[str, int],
                rng: random.Random) -> dict[str, str] | None:
    order = sorted(tasks, key=lambda n: (-tasks[n], n))
    loads = {w: 0 for w in caps}
    out: dict[str, str] = {}

    def dfs(i: int) -> bool:
        if i == len(order):
            return True
        task = order[i]
        workers = list(caps)
        rng.shuffle(workers)
        for w in workers:
            if loads[w] + tasks[task] <= caps[w]:
                loads[w] += tasks[task]
                out[task] = w
                if dfs(i + 1):
                    return True
                del out[task]
                loads[w] -= tasks[task]
        return False

    return out if dfs(0) else None


def controller_pack(rng: random.Random):
    n = 5
    nodes = [chr(ord("A") + i) for i in range(n)]
    durations = {node: rng.choice([10, 15, 20, 25]) for node in nodes}
    workers = ["W1", "W2", "W3"]
    caps = {w: rng.choice([30, 35, 40]) for w in workers}
    if sum(durations.values()) > sum(caps.values()):
        raise ValueError("infeasible")
    assignment = pack_solver(durations, caps, rng)
    if not assignment:
        raise ValueError("no packing")
    # Prefer cases where the naive first fit fails: that is the failure mode.
    naive = _assign_capacity(durations, caps)
    if naive and rng.random() < 0.5:
        raise ValueError("too easy")
    rows = [{"task": t, "worker": assignment[t]} for t in nodes]
    loads = {w: sum(durations[t] for t in nodes if assignment[t] == w) for w in workers}
    worker_text = ", ".join(f"{w} is free for {caps[w]} minutes" for w in workers)
    task_text = ", ".join(f"{t} takes {durations[t]} minutes" for t in nodes)
    prompt = (f"Three workers: {worker_text}. Tasks: {task_text}. Assign every task "
              "to exactly one worker without exceeding any worker's free time. "
              'Return ONLY JSON: {"assignments": [{"task": "A", "worker": "W1"}]}. '
              f"Include all {n} tasks.")
    answer = json.dumps({"assignments": rows, "loads": loads}, separators=(",", ":"))
    checker = {"type": "assign", "tasks": durations, "workers": caps}
    return "controller", "controller-pack", prompt, answer, checker


def controller_validate_hard(rng: random.Random):
    n = rng.randint(3, 4)
    nodes = [chr(ord("A") + i) for i in range(n)]
    durations = {node: rng.choice([5, 10, 15, 20]) for node in nodes}
    caps = {"W1": rng.choice([15, 20, 25, 30]), "W2": rng.choice([15, 20, 25, 30])}
    assignment = _assign_capacity(durations, caps)
    if not assignment:
        raise ValueError("no assignment")
    desired = rng.random() < 0.5

    def valid(rows: dict[str, str]) -> bool:
        if not all(t in rows for t in nodes):
            return False
        if not all(rows[t] in caps for t in nodes):
            return False
        loads = {w: sum(durations[t] for t in nodes if rows[t] == w) for w in caps}
        return all(loads[w] <= caps[w] for w in caps)

    rows = {t: assignment[t] for t in nodes}
    if desired:
        if not valid(rows):
            raise ValueError("first-fit invalid")
    else:
        # Make W1 just over capacity when possible; otherwise overload W1.
        for _ in range(30):
            rows = {t: assignment[t] for t in nodes}
            victim = rng.choice(nodes)
            other = "W2" if rows[victim] == "W1" else "W1"
            rows[victim] = other
            if not valid(rows):
                break
        else:
            rows = {t: "W1" for t in nodes}
            if valid(rows):
                raise ValueError("could not make invalid")
    proposed = ", ".join(f"{t} -> {rows[t]}" for t in nodes)
    prompt = (f"Workers: W1 has {caps['W1']} minutes free, W2 has {caps['W2']} "
              f"minutes free. Tasks: "
              f"{', '.join(f'{t} takes {durations[t]} minutes' for t in nodes)}. "
              f"Proposed assignment: {proposed}. Each task runs once. Is this "
              "assignment valid? Answer yes or no.")
    answer = "yes" if desired else "no"
    return ("controller", "controller-validate-hard", prompt, answer,
            {"type": "contains", "expect": [answer]})


def controller_schedule_tight(rng: random.Random):
    n = 6
    nodes = [chr(ord("A") + i) for i in range(n)]
    durations = {node: rng.randint(1, 5) for node in nodes}
    edges = []
    for i in range(1, n):
        if rng.random() < 0.7:
            edges.append((nodes[rng.randrange(0, i)], nodes[i]))
    if not edges:
        edges = [(nodes[0], nodes[-1])]
    workers = ["W1", "W2", "W3"]
    schedule = _solve_schedule(durations, workers, edges, rng)
    if not schedule:
        raise ValueError("no schedule")
    makespan = max(int(r["start"]) + durations[str(r["task"])] for r in schedule)
    deadline = makespan  # tight on purpose
    prompt = (f"Schedule tasks {', '.join(f'{n}({durations[n]})' for n in nodes)} on "
              f"workers {', '.join(workers)}. Dependencies: "
              f"{'; '.join(f'{a} before {b}' for a, b in edges)}. Return ONLY JSON: "
              '{"schedule": [{"task": "A", "worker": "W1", "start": 0}]}. The '
              f"makespan must be at most {deadline}.")
    answer = json.dumps({"schedule": schedule, "makespan": makespan},
                        separators=(",", ":"))
    checker = {"type": "schedule",
               "tasks": {k: {"duration": v} for k, v in durations.items()},
               "workers": workers, "dependencies": [[a, b] for a, b in edges],
               "deadline": deadline}
    return "controller", "controller-schedule-tight", prompt, answer, checker


def add(rows: list[dict], suite_prompts: set[str], item) -> bool:
    role, family, prompt, answer, checker = item
    if prompt in suite_prompts:
        return False
    if not check({"checker": checker}, answer):
        raise SystemExit(f"gold failed checker: {family}\n{prompt}\n{answer}")
    rows.append({
        "prompt": [
            {"role": "system", "content": "You are the Hive controller. Return only the requested JSON."},
            {"role": "user", "content": prompt},
        ],
        "completion": [{"role": "assistant", "content": answer}],
        "chat_template_kwargs": {"enable_thinking": False},
        "meta": json.dumps({"role": role, "family": family, "checker": checker},
                           separators=(",", ":")),
    })
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "experiments" / "data" / "controller-sft-v3"))
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--pack", type=int, default=400)
    parser.add_argument("--validate", type=int, default=300)
    parser.add_argument("--schedule", type=int, default=300)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    suite_prompts = {t["prompt"] for t in json.loads(SUITE.read_text())["tasks"]}
    base_rows = [json.loads(line) for line in V2.read_text().splitlines() if line.strip()]
    new_rows: list[dict] = []
    failures = 0

    for fn, count in ((controller_pack, args.pack),
                      (controller_validate_hard, args.validate),
                      (controller_schedule_tight, args.schedule)):
        made = 0
        for _ in range(count * 50):
            if made >= count:
                break
            try:
                item = fn(rng)
            except ValueError:
                failures += 1
                continue
            if add(new_rows, suite_prompts, item):
                made += 1
        print(f"{fn.__name__}: {made}/{count}")

    rows = base_rows + new_rows
    rng.shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {
        "rows": len(rows),
        "base_rows": len(base_rows),
        "gap_rows": len(new_rows),
        "gap_families": {
            "controller-pack": args.pack,
            "controller-validate-hard": args.validate,
            "controller-schedule-tight": args.schedule,
        },
        "failed_generations": failures,
        "path": str(path),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
