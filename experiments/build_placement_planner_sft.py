#!/usr/bin/env python
"""Build a placement-planner recovery corpus and suite.

Tests the curriculum hypothesis: after training a role, teach it to plan for
that role. Placement plans and placement assignments with a `check` trace are
mixed with controller/planner replay, then trained on the failed 3-role model.

Outputs:

- experiments/roles/tasks-placement-plan.json  (12 held-out planning tasks)
- experiments/data/multirole-cpp-plan-v1/train.jsonl
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
DATA = REPO / "experiments" / "data"
CONTROLLER = DATA / "controller-sft-v3" / "train.jsonl"
PLANNER = DATA / "planner-sft-v1" / "train.jsonl"
PLACEMENT = DATA / "placement-sft-v1" / "train.jsonl"
PLACEMENT_PLAN_SUITE = REPO / "experiments" / "roles" / "tasks-placement-plan.json"

SYSTEM = "You are the Hive placement planner. Return only the requested answer."


def solve(models: dict[str, int], cards: dict[str, int],
          rng: random.Random) -> dict[str, str] | None:
    order = sorted(models, key=lambda m: (-models[m], m))
    loads = {c: 0 for c in cards}
    counts = {c: 0 for c in cards}
    out: dict[str, str] = {}

    def dfs(i: int) -> bool:
        if i == len(order):
            return True
        model = order[i]
        choices = list(cards)
        rng.shuffle(choices)
        for card in choices:
            if counts[card] >= 2:
                continue
            if loads[card] + models[model] <= cards[card]:
                loads[card] += models[model]
                counts[card] += 1
                out[model] = card
                if dfs(i + 1):
                    return True
                del out[model]
                counts[card] -= 1
                loads[card] -= models[model]
        return False

    return out if dfs(0) else None


def placement_scenario(rng: random.Random):
    models = {f"M{i}": rng.choice([4, 6, 8, 10, 12, 14]) for i in range(1, 4)}
    cards = {"C1": rng.choice([16, 20, 24, 28]), "C2": rng.choice([16, 20, 24, 28])}
    reserve = 2
    return models, cards, reserve


def placement_plan(rng: random.Random):
    models, cards, reserve = placement_scenario(rng)
    usable = {c: cap - reserve for c, cap in cards.items()}
    prompt = (f"Plan a safe placement of models M1, M2, M3 needing "
              f"{models['M1']} GB, {models['M2']} GB, and {models['M3']} GB on "
              f"cards C1={cards['C1']} GB and C2={cards['C2']} GB. Each card keeps "
              "2 GB free and holds at most two models. Return ONLY JSON: "
              '{"steps": ["..."], "risks": ["..."]}. Include at least 3 steps and 2 risks.')
    steps = [
        f"Compute usable capacity: C1={usable['C1']} GB, C2={usable['C2']} GB.",
        "Place the largest model first, then add a second model only if the card's load stays within usable capacity.",
        "Check every model is assigned exactly once and every card load is at or below its usable capacity.",
    ]
    risks = [
        "Packing two models without recomputing the second load can exceed the reserve.",
        "If no card has two free slots, a model must be evicted or moved before the placement is safe.",
    ]
    answer = json.dumps({"steps": steps, "risks": risks}, separators=(",", ":"))
    return ("placement", "placement-plan", prompt, answer,
            {"type": "json", "required": ["steps", "risks"],
             "min_items": {"steps": 3, "risks": 2}})


def placement_assign_check(rng: random.Random):
    models, cards, reserve = placement_scenario(rng)
    usable = {c: cap - reserve for c, cap in cards.items()}
    if sum(models.values()) > sum(usable.values()):
        raise ValueError("infeasible")
    assignment = solve(models, usable, rng)
    if not assignment:
        raise ValueError("no assignment")
    rows = [{"task": m, "worker": assignment[m]} for m in models]
    loads = {c: sum(models[m] for m in models if assignment[m] == c) for c in usable}
    check = [
        f"C1 usable {usable['C1']} GB, C2 usable {usable['C2']} GB",
        f"C1 load {loads['C1']} GB, C2 load {loads['C2']} GB",
        "every card load is within usable capacity and every model is placed once",
    ]
    prompt = (f"Models M1, M2, M3 need {models['M1']} GB, {models['M2']} GB, and "
              f"{models['M3']} GB. Cards C1={cards['C1']} GB and C2={cards['C2']} GB. "
              "Each card keeps 2 GB free and holds at most two models. Return ONLY "
              'JSON with a short "check" list, then "assignments" and "loads".')
    answer = json.dumps({"check": check, "assignments": rows, "loads": loads},
                        separators=(",", ":"))
    return ("placement", "placement-assign-check", prompt, answer,
            {"type": "assign", "tasks": models, "workers": usable})


def add_row(rows: list[dict], item, suite_prompts: set[str]) -> bool:
    role, family, prompt, answer, checker = item
    if prompt in suite_prompts:
        return False
    rows.append({
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "completion": [{"role": "assistant", "content": answer}],
        "chat_template_kwargs": {"enable_thinking": False},
        "meta": json.dumps({"role": role, "family": family, "checker": checker},
                           separators=(",", ":")),
    })
    return True


def build_plan_suite(seed: int, count: int = 12) -> list[dict]:
    rng = random.Random(seed)
    tasks: list[dict] = []
    for _ in range(count * 80):
        if len(tasks) >= count:
            break
        try:
            _, _, prompt, answer, checker = placement_plan(rng)
        except ValueError:
            continue
        if any(t["prompt"] == prompt for t in tasks):
            continue
        from experiments.cascade.checkers import check
        if not check({"checker": checker}, answer):
            raise SystemExit("plan gold failed")
        tasks.append({"id": f"placement-plan-{len(tasks)+1:03d}",
                      "role": "placement", "prompt": prompt, "checker": checker})
    return tasks


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA / "multirole-cpp-plan-v1"))
    parser.add_argument("--controller", type=int, default=1500)
    parser.add_argument("--planner", type=int, default=1000)
    parser.add_argument("--placement", type=int, default=600)
    parser.add_argument("--plan", type=int, default=600)
    parser.add_argument("--check", type=int, default=400)
    parser.add_argument("--seed", type=int, default=919)
    args = parser.parse_args(argv)

    suite = build_plan_suite(20261102)
    suite_prompts = {t["prompt"] for t in suite}
    suite_path = PLACEMENT_PLAN_SUITE
    suite_path.write_text(json.dumps(
        {"name": "placement-plan-suite", "version": 1, "count": len(suite),
         "tasks": suite}, indent=1) + "\n")
    print(f"wrote {suite_path} ({len(suite)} tasks)")

    rng = random.Random(args.seed)
    new_rows: list[dict] = []
    for fn, count in [(placement_plan, args.plan), (placement_assign_check, args.check)]:
        made = 0
        for _ in range(count * 40):
            if made >= count:
                break
            try:
                item = fn(rng)
            except ValueError:
                continue
            if add_row(new_rows, item, suite_prompts):
                made += 1
        print(f"{fn.__name__}: {made}")

    sources = {
        "controller": (load(CONTROLLER), args.controller),
        "planner": (load(PLANNER), args.planner),
        "placement": (load(PLACEMENT), args.placement),
    }
    mixed = list(new_rows)
    counts = {"controller": 0, "planner": 0, "placement_replay": 0,
              "placement_plan": args.plan, "placement_check": args.check}
    for name, (rows, requested) in sources.items():
        take = min(requested, len(rows))
        mixed.extend(rng.sample(rows, take))
        counts[name if name != "placement" else "placement_replay"] = take
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
