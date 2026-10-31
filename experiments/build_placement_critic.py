#!/usr/bin/env python
"""Build the placement + critic suites and training corpora.

Placement is controller assignment with VRAM instead of minutes. Critic is the
validation half of controller: check an assignment, schedule, or order and
answer valid/invalid. Both share the constraint logic already learned by the
controller+planner checkpoint.

Outputs:

- experiments/roles/tasks-placement.json
- experiments/roles/tasks-critic.json
- experiments/data/placement-sft-v1/train.jsonl
- experiments/data/critic-sft-v1/train.jsonl

Every gold answer is validated with the real checker. The suites are generated
first and exact prompt matches are removed from the corpora so evaluation stays
held out.
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
    _assign_capacity,
    _solve_schedule,
    _topological,
)
from experiments.cascade.checkers import check  # noqa: E402

SYSTEM = {
    "placement": "You are the Hive placement planner. Return only the requested answer.",
    "critic": "You are the Hive critic. Check the proposal and answer yes or no.",
}


def add_task(tasks: list[dict], task_id: str, role: str, prompt: str,
             answer: str, checker: dict) -> None:
    if any(t["id"] == task_id or t["prompt"] == prompt for t in tasks):
        raise ValueError("duplicate")
    if not check({"checker": checker}, answer):
        raise SystemExit(f"gold failed checker: {task_id}\n{prompt}\n{answer}\n{checker}")
    tasks.append({"id": task_id, "role": role, "prompt": prompt, "checker": checker})


def add_row(rows: list[dict], role: str, family: str, prompt: str,
            answer: str, checker: dict) -> None:
    if not check({"checker": checker}, answer):
        raise SystemExit(f"gold failed checker: {family}\n{prompt}\n{answer}\n{checker}")
    rows.append({
        "prompt": [
            {"role": "system", "content": SYSTEM[role]},
            {"role": "user", "content": prompt},
        ],
        "completion": [{"role": "assistant", "content": answer}],
        "chat_template_kwargs": {"enable_thinking": False},
        "meta": json.dumps({"role": role, "family": family, "checker": checker},
                           separators=(",", ":")),
    })


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


def placement_blocks(rng: random.Random):
    weights = round(rng.uniform(6, 16), 2)
    reserve = rng.choice([1, 2])
    kv = rng.choice([0.5, 0.75, 1.0, 1.5, 2.0])
    total = rng.choice([16, 20, 24])
    fit = int((total - reserve - weights) // kv)
    if fit < 0:
        raise ValueError("negative")
    prompt = (f"A model has {weights} GB of weights, a {reserve} GB runtime "
              f"reserve, and {kv} GB of KV cache per 8k context block. A card has "
              f"{total} GB total and keeps {reserve} GB free for the system. How "
              "many whole 8k context blocks fit on the card? Answer with the "
              "number only.")
    usable = total - reserve
    free = usable - weights
    answer = (f"{total} - {reserve} = {usable} GB usable; {usable} - {weights} "
              f"= {free:.2f} GB for KV; {free:.2f} / {kv} = "
              f"{free / kv:.2f}, so {fit} whole blocks fit.")
    return ("placement", "placement-blocks", prompt, answer,
            {"type": "number", "expect": fit})


def placement_fit(rng: random.Random):
    weights = round(rng.uniform(6, 18), 2)
    kv = rng.choice([0.5, 1.0, 1.5, 2.0])
    reserve = rng.choice([1, 2])
    total = rng.choice([16, 20, 24])
    need = weights + kv
    usable = total - reserve
    fits = need <= usable
    prompt = (f"A model needs {weights} GB of weights and {kv} GB of KV cache. "
              f"A {total} GB card keeps {reserve} GB free for the system. Does "
              "it fit? Answer yes or no.")
    answer = "yes" if fits else "no"
    return ("placement", "placement-fit", prompt, answer,
            {"type": "contains", "expect": [answer]})


def placement_smallest(rng: random.Random):
    weights = round(rng.uniform(8, 20), 2)
    reserve = rng.choice([1, 2, 4])
    cards = sorted({rng.choice([16, 20, 24, 32, 40]) for _ in range(4)})
    need = weights + reserve
    fits = [c for c in cards if c >= need]
    if not fits:
        raise ValueError("no card")
    answer = str(min(fits))
    prompt = (f"A model has {weights} GB of weights and needs a {reserve} GB "
              f"runtime reserve. The candidate cards are {', '.join(map(str, cards))} "
              "GB. Which is the smallest card that fits? Answer with the number "
              "only.")
    return ("placement", "placement-smallest", prompt, answer,
            {"type": "number", "expect": int(answer)})


def placement_assign(rng: random.Random):
    n = 3
    models = [f"M{i}" for i in range(1, n + 1)]
    sizes = {model: rng.choice([4, 6, 8, 10, 12, 14]) for model in models}
    cards = {"C1": rng.choice([16, 20, 24, 28]), "C2": rng.choice([16, 20, 24, 28])}
    if sum(sizes.values()) > sum(cards.values()):
        raise ValueError("infeasible")
    assignment = _assign_capacity(sizes, cards)
    if not assignment:
        raise ValueError("no assignment")
    # Enforce at most two models per card in the gold.
    counts = {card: sum(1 for m in models if assignment[m] == card) for card in cards}
    if any(count > 2 for count in counts.values()):
        raise ValueError("too many per card")
    rows = [{"task": m, "worker": assignment[m]} for m in models]
    loads = {card: sum(sizes[m] for m in models if assignment[m] == card)
             for card in cards}
    prompt = (f"Models M1, M2, M3 need {sizes['M1']} GB, {sizes['M2']} GB, and "
              f"{sizes['M3']} GB. Cards C1={cards['C1']} GB and C2={cards['C2']} GB. "
              "Assign every model to exactly one card, at most two models per "
              'card. Return ONLY JSON: {"assignments": [{"task": "M1", "worker": "C1"}]}.')
    answer = json.dumps({"assignments": rows, "loads": loads}, separators=(",", ":"))
    return ("placement", "placement-assign", prompt, answer,
            {"type": "assign", "tasks": sizes, "workers": cards})


def placement_two_cards(rng: random.Random):
    sizes = [rng.choice([6, 8, 10, 12, 14, 16]) for _ in range(3)]
    cards = [rng.choice([20, 24, 28, 32]), rng.choice([16, 20, 24, 28])]
    feasible = False
    for assignment in itertools.product(range(2), repeat=3):
        if assignment.count(0) > 2 or assignment.count(1) > 2:
            continue
        loads = [sum(s for s, a in zip(sizes, assignment) if a == i) for i in range(2)]
        if all(load <= cap for load, cap in zip(loads, cards)):
            feasible = True
            break
    answer = "yes" if feasible else "no"
    prompt = (f"Models A, B, C need {sizes[0]} GB, {sizes[1]} GB, and {sizes[2]} "
              f"GB. Cards C1={cards[0]} GB and C2={cards[1]} GB. At most two "
              f"models may share a card. Can all three run at once? Answer yes or no.")
    return ("placement", "placement-two-cards", prompt, answer,
            {"type": "contains", "expect": [answer]})


# --------------------------------------------------------------------------
# critic
# --------------------------------------------------------------------------


def _valid_assignment(rows: dict[str, str], durations: dict[str, int],
                      caps: dict[str, int]) -> bool:
    if set(rows) != set(durations):
        return False
    if any(worker not in caps for worker in rows.values()):
        return False
    loads = {w: sum(durations[t] for t in durations if rows[t] == w) for w in caps}
    return all(loads[w] <= caps[w] for w in caps)


def critic_assign(rng: random.Random):
    n = rng.randint(3, 4)
    tasks = [chr(ord("A") + i) for i in range(n)]
    durations = {t: rng.choice([5, 10, 15, 20]) for t in tasks}
    caps = {"W1": rng.choice([15, 20, 25, 30]), "W2": rng.choice([15, 20, 25, 30])}
    assignment = _assign_capacity(durations, caps)
    if not assignment:
        raise ValueError("no assignment")
    desired = rng.random() < 0.5
    rows = dict(assignment)
    if desired:
        if not _valid_assignment(rows, durations, caps):
            raise ValueError("first-fit invalid")
    else:
        for _ in range(30):
            rows = dict(assignment)
            victim = rng.choice(tasks)
            rows[victim] = "W2" if rows[victim] == "W1" else "W1"
            if not _valid_assignment(rows, durations, caps):
                break
        else:
            rows = {t: "W1" for t in tasks}
            if _valid_assignment(rows, durations, caps):
                raise ValueError("could not corrupt")
    proposed = ", ".join(f"{t} -> {rows[t]}" for t in tasks)
    prompt = (f"Workers: W1 has {caps['W1']} minutes free, W2 has {caps['W2']} "
              f"minutes free. Tasks: "
              f"{', '.join(f'{t} takes {durations[t]} minutes' for t in tasks)}. "
              f"Proposed assignment: {proposed}. Each task runs once. Is this "
              "assignment valid? Answer yes or no.")
    answer = "yes" if desired else "no"
    return ("critic", "critic-assign", prompt, answer,
            {"type": "contains", "expect": [answer]})


def _schedule_spec(durations, workers, edges, deadline):
    return {"type": "schedule",
            "tasks": {k: {"duration": v} for k, v in durations.items()},
            "workers": workers, "dependencies": [[a, b] for a, b in edges],
            "deadline": deadline}


def critic_schedule(rng: random.Random):
    n = rng.randint(4, 5)
    tasks = [chr(ord("A") + i) for i in range(n)]
    durations = {t: rng.randint(1, 4) for t in tasks}
    workers = ["W1", "W2"]
    edges = []
    for i in range(1, n):
        if rng.random() < 0.7:
            edges.append((tasks[rng.randrange(0, i)], tasks[i]))
    if not edges:
        edges = [(tasks[0], tasks[-1])]
    schedule = _solve_schedule(durations, workers, edges, rng)
    if not schedule:
        raise ValueError("no schedule")
    makespan = max(int(r["start"]) + durations[str(r["task"])] for r in schedule)
    deadline = makespan + 1
    spec = _schedule_spec(durations, workers, edges, deadline)
    if not check({"checker": spec}, json.dumps({"schedule": schedule}, separators=(",", ":"))):
        raise ValueError("gold schedule invalid")
    desired = rng.random() < 0.5
    rows = [dict(r) for r in schedule]
    if not desired:
        for _ in range(40):
            rows = [dict(r) for r in schedule]
            target = rng.choice(rows)
            mode = rng.choice(["start0", "overlap", "late", "worker"])
            if mode == "start0":
                target["start"] = 0
            elif mode == "late":
                target["start"] = deadline + 1
            elif mode == "worker":
                target["worker"] = "W3"
            else:
                other = rng.choice(rows)
                target["worker"] = other["worker"]
                target["start"] = int(other["start"])
            candidate = json.dumps({"schedule": rows}, separators=(",", ":"))
            if not check({"checker": spec}, candidate):
                break
        else:
            raise ValueError("could not corrupt schedule")
    proposed = json.dumps({"schedule": rows}, separators=(",", ":"))
    prompt = (f"Tasks: {', '.join(f'{t}={durations[t]}' for t in tasks)}. "
              f"Workers: {', '.join(workers)}. Dependencies: "
              f"{'; '.join(f'{a} before {b}' for a, b in edges)}. Deadline: "
              f"{deadline}. Proposed schedule: {proposed}. Is this schedule "
              "valid? Answer yes or no.")
    answer = "yes" if desired else "no"
    return ("critic", "critic-schedule", prompt, answer,
            {"type": "contains", "expect": [answer]})


def critic_order(rng: random.Random):
    n = rng.randint(4, 6)
    nodes = [chr(ord("A") + i) for i in range(n)]
    edges = []
    for i in range(1, n):
        edges.append((nodes[rng.randrange(0, i)], nodes[i]))
    order = _topological(nodes, edges)
    spec = {"type": "ordered", "field": "order", "required_items": nodes,
            "before": [[a, b] for a, b in edges]}
    desired = rng.random() < 0.5
    proposed = list(order)
    if not desired:
        for _ in range(30):
            proposed = list(order)
            i, j = rng.sample(range(n), 2)
            proposed[i], proposed[j] = proposed[j], proposed[i]
            if not check({"checker": spec}, json.dumps({"order": proposed}, separators=(",", ":"))):
                break
        else:
            raise ValueError("could not corrupt order")
    prompt = (f"Merge jobs: {', '.join(nodes)}. Constraints: "
              f"{'; '.join(f'{a} before {b}' for a, b in edges)}. Proposed "
              f"order: {proposed}. Is this order valid? Answer yes or no.")
    answer = "yes" if desired else "no"
    return ("critic", "critic-order", prompt, answer,
            {"type": "contains", "expect": [answer]})


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

PLACEMENT_FNS = [placement_blocks, placement_fit, placement_smallest,
                 placement_assign, placement_two_cards]
CRITIC_FNS = [critic_assign, critic_schedule, critic_order]


def build_suite(seed: int):
    rng = random.Random(seed)
    placement: list[dict] = []
    critic: list[dict] = []
    for fn, count in [(placement_blocks, 10), (placement_fit, 10),
                      (placement_smallest, 8), (placement_assign, 10),
                      (placement_two_cards, 8)]:
        made = 0
        for _ in range(count * 60):
            if made >= count:
                break
            try:
                _, _, prompt, answer, checker = fn(rng)
                add_task(placement, f"placement-{len(placement) + 1:03d}",
                         "placement", prompt, answer, checker)
            except ValueError:
                continue
            made += 1
    for fn, count in [(critic_assign, 14), (critic_schedule, 14), (critic_order, 10)]:
        made = 0
        for _ in range(count * 60):
            if made >= count:
                break
            try:
                _, _, prompt, answer, checker = fn(rng)
                add_task(critic, f"critic-{len(critic) + 1:03d}",
                         "critic", prompt, answer, checker)
            except ValueError:
                continue
            made += 1
    return placement, critic


def build_corpus(seed: int, suite_prompts: set[str]):
    rng = random.Random(seed)
    placement: list[dict] = []
    critic: list[dict] = []
    for fn, count in [(placement_blocks, 300), (placement_fit, 300),
                      (placement_smallest, 150), (placement_assign, 300),
                      (placement_two_cards, 150)]:
        made = 0
        for _ in range(count * 40):
            if made >= count:
                break
            try:
                role, family, prompt, answer, checker = fn(rng)
            except ValueError:
                continue
            if prompt in suite_prompts:
                continue
            add_row(placement, role, family, prompt, answer, checker)
            made += 1
    for fn, count in [(critic_assign, 400), (critic_schedule, 400), (critic_order, 200)]:
        made = 0
        for _ in range(count * 60):
            if made >= count:
                break
            try:
                role, family, prompt, answer, checker = fn(rng)
            except ValueError:
                continue
            if prompt in suite_prompts:
                continue
            add_row(critic, role, family, prompt, answer, checker)
            made += 1
    return placement, critic


def write_suite(path: Path, name: str, tasks: list[dict]) -> None:
    payload = {"name": name, "version": 1, "count": len(tasks), "tasks": tasks}
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {path} ({len(tasks)} tasks)")


def write_corpus(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-seed", type=int, default=20261101)
    parser.add_argument("--corpus-seed", type=int, default=5151)
    args = parser.parse_args(argv)

    placement_suite, critic_suite = build_suite(args.suite_seed)
    suite_prompts = {t["prompt"] for t in placement_suite + critic_suite}
    placement_rows, critic_rows = build_corpus(args.corpus_seed, suite_prompts)

    write_suite(REPO / "experiments" / "roles" / "tasks-placement.json",
                "placement-suite", placement_suite)
    write_suite(REPO / "experiments" / "roles" / "tasks-critic.json",
                "critic-suite", critic_suite)
    write_corpus(REPO / "experiments" / "data" / "placement-sft-v1" / "train.jsonl",
                 placement_rows)
    write_corpus(REPO / "experiments" / "data" / "critic-sft-v1" / "train.jsonl",
                 critic_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
