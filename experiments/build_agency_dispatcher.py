#!/usr/bin/env python
"""Build the agency + dispatcher (Hive Operator) suites and corpus.

Agency is tool selection/order; dispatcher is job assignment and scheduling.
Both are sequencing/assignment roles and share the controller's constraint
logic, so they fit one separate two-role adapter.

Outputs:

- experiments/roles/tasks-agency.json
- experiments/roles/tasks-dispatcher.json
- experiments/data/operator-sft-v1/train.jsonl
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
    Builder,
    _assign_capacity,
    _solve_schedule,
    agency_case,
)
from experiments.cascade.checkers import check  # noqa: E402

ROLES = REPO / "experiments" / "roles" / "tasks-roles.json"
DATA = REPO / "experiments" / "data"
OUT_A = REPO / "experiments" / "roles" / "tasks-agency.json"
OUT_D = REPO / "experiments" / "roles" / "tasks-dispatcher.json"
SYSTEM = "You are the Hive dispatcher. Return only the requested answer."


def add_task(tasks: list[dict], task_id: str, role: str, prompt: str,
             answer: str, checker: dict) -> None:
    if any(t["id"] == task_id or t["prompt"] == prompt for t in tasks):
        raise ValueError("duplicate")
    if not check({"checker": checker}, answer):
        raise SystemExit(f"gold failed checker: {task_id}\n{prompt}\n{answer}\n{checker}")
    tasks.append({"id": task_id, "role": role, "prompt": prompt, "checker": checker})


def dispatcher_assign(rng: random.Random):
    n = rng.randint(4, 6)
    jobs = [chr(ord("A") + i) for i in range(n)]
    durations = {j: rng.choice([5, 10, 15, 20]) for j in jobs}
    workers = ["W1", "W2", "W3"]
    caps = {w: rng.choice([30, 40, 50, 60]) for w in workers}
    if sum(durations.values()) > sum(caps.values()):
        raise ValueError("infeasible")
    assignment = _assign_capacity(durations, caps)
    if not assignment:
        raise ValueError("no assignment")
    rows = [{"task": j, "worker": assignment[j]} for j in jobs]
    loads = {w: sum(durations[j] for j in jobs if assignment[j] == w) for w in workers}
    worker_text = ", ".join(f"{w} has {caps[w]} minutes" for w in workers)
    job_text = ", ".join(f"{j} takes {durations[j]} minutes" for j in jobs)
    prompt = (f"Dispatch jobs {job_text} to workers {worker_text}. Assign every "
              "job exactly once without exceeding a worker's time. Return ONLY "
              'JSON: {"assignments": [{"task": "A", "worker": "W1"}]}.')
    answer = json.dumps({"assignments": rows, "loads": loads}, separators=(",", ":"))
    return ("dispatcher", "dispatcher-assign", prompt, answer,
            {"type": "assign", "tasks": durations, "workers": caps})


def dispatcher_schedule(rng: random.Random):
    n = rng.randint(5, 6)
    jobs = [chr(ord("A") + i) for i in range(n)]
    durations = {j: rng.randint(1, 5) for j in jobs}
    workers = ["W1", "W2", "W3"]
    edges = []
    for i in range(1, n):
        if rng.random() < 0.7:
            edges.append((jobs[rng.randrange(0, i)], jobs[i]))
    if not edges:
        edges = [(jobs[0], jobs[-1])]
    schedule = _solve_schedule(durations, workers, edges, rng)
    if not schedule:
        raise ValueError("no schedule")
    makespan = max(int(r["start"]) + durations[str(r["task"])] for r in schedule)
    deadline = makespan + 1
    prompt = (f"Schedule jobs {', '.join(f'{j}({durations[j]})' for j in jobs)} on "
              f"workers {', '.join(workers)}. Dependencies: "
              f"{'; '.join(f'{a} before {b}' for a, b in edges)}. Return ONLY JSON: "
              '{"schedule": [{"task": "A", "worker": "W1", "start": 0}]}. Makespan '
              f"must be at most {deadline}.")
    answer = json.dumps({"schedule": schedule}, separators=(",", ":"))
    return ("dispatcher", "dispatcher-schedule", prompt, answer,
            {"type": "schedule",
             "tasks": {k: {"duration": v} for k, v in durations.items()},
             "workers": workers, "dependencies": [[a, b] for a, b in edges],
             "deadline": deadline})


def dispatcher_priority(rng: random.Random):
    a = "a live rollback with a 5-minute deadline"
    b = rng.choice(["a benchmark with a 2-hour deadline",
                    "a background re-index with a 6-hour deadline"])
    pair = [a, b]
    rng.shuffle(pair)
    answer = "A" if pair[0] == a else "B"
    prompt = (f"Two jobs need the only free worker. Job A is {pair[0]}. Job B is "
              f"{pair[1]}. Which job goes first? Answer A or B with exactly one letter.")
    return ("dispatcher", "dispatcher-priority", prompt, answer,
            {"type": "choice", "expect": answer})


def dispatcher_validate(rng: random.Random):
    n = rng.randint(3, 4)
    jobs = [chr(ord("A") + i) for i in range(n)]
    durations = {j: rng.choice([5, 10, 15, 20]) for j in jobs}
    caps = {"W1": rng.choice([15, 20, 25, 30]), "W2": rng.choice([15, 20, 25, 30])}
    assignment = _assign_capacity(durations, caps)
    if not assignment:
        raise ValueError("no assignment")
    desired = rng.random() < 0.5
    rows = dict(assignment)
    if not desired:
        for _ in range(30):
            rows = dict(assignment)
            victim = rng.choice(jobs)
            rows[victim] = "W2" if rows[victim] == "W1" else "W1"
            loads = {w: sum(durations[j] for j in jobs if rows[j] == w) for w in caps}
            if any(loads[w] > caps[w] for w in caps):
                break
        else:
            rows = {j: "W1" for j in jobs}
            loads = {w: sum(durations[j] for j in jobs if rows[j] == w) for w in caps}
            if not any(loads[w] > caps[w] for w in caps):
                raise ValueError("could not corrupt")
    loads = {w: sum(durations[j] for j in jobs if rows[j] == w) for w in caps}
    valid = all(loads[w] <= caps[w] for w in caps) == desired
    proposed = ", ".join(f"{j} -> {rows[j]}" for j in jobs)
    prompt = (f"Workers: W1 has {caps['W1']} minutes, W2 has {caps['W2']} minutes. "
              f"Jobs: {', '.join(f'{j} takes {durations[j]} minutes' for j in jobs)}. "
              f"Proposed dispatch: {proposed}. Is this dispatch valid? Answer yes "
              "or no.")
    answer = "yes" if desired else "no"
    return ("dispatcher", "dispatcher-validate", prompt, answer,
            {"type": "contains", "expect": [answer]})


def add_row(rows: list[dict], item, suite_prompts: set[str]) -> None:
    role, family, prompt, answer, checker = item
    if prompt in suite_prompts:
        return
    if not check({"checker": checker}, answer):
        raise SystemExit(f"gold failed: {family}")
    rows.append({
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": answer}],
        "chat_template_kwargs": {"enable_thinking": False},
        "meta": json.dumps({"role": role, "family": family, "checker": checker},
                           separators=(",", ":")),
    })


def build_agency_suite(seed: int, count: int = 40) -> list[dict]:
    rng = random.Random(seed)
    tasks: list[dict] = []
    existing = json.loads(ROLES.read_text())["tasks"]
    for task in existing:
        if task["role"] == "agency":
            tasks.append({k: v for k, v in task.items()})
    made = 0
    for _ in range(count * 80):
        if made >= count:
            break
        try:
            role, family, prompt, answer, checker = agency_case(rng)
        except Exception:  # noqa: BLE001
            continue
        try:
            add_task(tasks, f"agency-{len(tasks)+1:03d}", role, prompt, answer, checker)
        except ValueError:
            continue
        made += 1
    return tasks


def build_dispatcher_suite(seed: int, count: int = 40) -> list[dict]:
    rng = random.Random(seed)
    tasks: list[dict] = []
    fns = [dispatcher_assign, dispatcher_schedule, dispatcher_priority, dispatcher_validate]
    made = 0
    for _ in range(count * 80):
        if made >= count:
            break
        try:
            role, family, prompt, answer, checker = rng.choice(fns)(rng)
        except Exception:  # noqa: BLE001
            continue
        try:
            add_task(tasks, f"dispatcher-{len(tasks)+1:03d}", role, prompt, answer, checker)
        except ValueError:
            continue
        made += 1
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA / "operator-sft-v1"))
    parser.add_argument("--suite-seed", type=int, default=20261104)
    parser.add_argument("--corpus-seed", type=int, default=7272)
    args = parser.parse_args(argv)

    agency = build_agency_suite(args.suite_seed, 40)
    dispatcher = build_dispatcher_suite(args.suite_seed + 1, 40)
    OUT_A.write_text(json.dumps({"name": "agency-suite", "version": 1,
                                 "count": len(agency), "tasks": agency}, indent=1) + "\n")
    OUT_D.write_text(json.dumps({"name": "dispatcher-suite", "version": 1,
                                 "count": len(dispatcher), "tasks": dispatcher}, indent=1) + "\n")
    print(f"wrote {OUT_A} ({len(agency)}) and {OUT_D} ({len(dispatcher)})")

    suite_prompts = {t["prompt"] for t in agency + dispatcher}
    b = Builder(args.corpus_seed)
    rng = b.rng
    b.fill("agency", 1200, lambda: agency_case(rng), tries_per_item=30)
    rows = [r for r in b.rows if r["prompt"][1]["content"] not in suite_prompts]
    dispatcher_rows: list[dict] = []
    for fn, count in [(dispatcher_assign, 500), (dispatcher_schedule, 400),
                      (dispatcher_priority, 100), (dispatcher_validate, 200)]:
        made = 0
        for _ in range(count * 50):
            if made >= count:
                break
            try:
                item = fn(rng)
            except Exception:  # noqa: BLE001
                continue
            before = len(dispatcher_rows)
            add_row(dispatcher_rows, item, suite_prompts)
            if len(dispatcher_rows) > before:
                made += 1
    mixed = rows + dispatcher_rows
    rng.shuffle(mixed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in mixed:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {"rows": len(mixed), "agency": len(rows),
             "dispatcher": len(dispatcher_rows), "path": str(path)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
