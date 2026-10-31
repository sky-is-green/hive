#!/usr/bin/env python
"""Build the combined mechanics + RAG (Hive Clerk) suites and corpus.

Mechanics and RAG are both document-in / structured-answer-out roles. They are
combined into one separate adapter so the live Conductor is untouched.

Outputs:

- experiments/roles/tasks-mechanics.json
- experiments/roles/tasks-rag.json
- experiments/data/clerk-sft-v1/train.jsonl
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

from experiments.build_role_sft import Builder, mechanics_case, rag_case  # noqa: E402
from experiments.cascade.checkers import check  # noqa: E402

ROLES = REPO / "experiments" / "roles" / "tasks-roles.json"
DATA = REPO / "experiments" / "data"
OUT_M = REPO / "experiments" / "roles" / "tasks-mechanics.json"
OUT_R = REPO / "experiments" / "roles" / "tasks-rag.json"


def add_task(tasks: list[dict], task_id: str, role: str, prompt: str,
             answer: str, checker: dict) -> None:
    if any(t["id"] == task_id or t["prompt"] == prompt for t in tasks):
        raise ValueError("duplicate")
    if not check({"checker": checker}, answer):
        raise SystemExit(f"gold failed checker: {task_id}\n{prompt}\n{answer}\n{checker}")
    tasks.append({"id": task_id, "role": role, "prompt": prompt, "checker": checker})


def build_suite(seed: int, role: str, fn, count: int) -> list[dict]:
    rng = random.Random(seed)
    tasks: list[dict] = []
    existing = json.loads(ROLES.read_text())["tasks"]
    for task in existing:
        if task["role"] == role:
            tasks.append({k: v for k, v in task.items()})
    made = 0
    for _ in range(count * 80):
        if made >= count:
            break
        try:
            r, family, prompt, answer, checker = fn(rng)
        except Exception:  # noqa: BLE001 - infeasible random instance
            continue
        try:
            add_task(tasks, f"{role}-{200 + len(tasks):03d}", role, prompt, answer, checker)
        except ValueError:
            continue
        made += 1
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA / "clerk-sft-v1"))
    parser.add_argument("--suite-seed", type=int, default=20261103)
    parser.add_argument("--corpus-seed", type=int, default=8181)
    args = parser.parse_args(argv)

    mechanics = build_suite(args.suite_seed, "mechanics", mechanics_case, 36)
    rag = build_suite(args.suite_seed + 1, "rag", rag_case, 36)
    OUT_M.write_text(json.dumps({"name": "mechanics-suite", "version": 1,
                                 "count": len(mechanics), "tasks": mechanics}, indent=1) + "\n")
    OUT_R.write_text(json.dumps({"name": "rag-suite", "version": 1,
                                 "count": len(rag), "tasks": rag}, indent=1) + "\n")
    print(f"wrote {OUT_M} ({len(mechanics)}) and {OUT_R} ({len(rag)})")

    b = Builder(args.corpus_seed)
    rng = b.rng
    b.fill("mechanics", 1200, lambda: mechanics_case(rng), tries_per_item=30)
    b.fill("rag", 1200, lambda: rag_case(rng), tries_per_item=30)
    seen = {t["prompt"] for t in mechanics + rag}
    rows = [row for row in b.rows if row["prompt"][1]["content"] not in seen]
    b.rng.shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = b.summary()
    summary["rows"] = len(rows)
    summary["dropped_suite_overlap"] = len(b.rows) - len(rows)
    summary["path"] = str(path)
    (out / "stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
