#!/usr/bin/env python
"""Merge harvest runs into one corpus of record, re-verifying every label.

Round 1 (``decision-harvest-20261017``) is the base corpus. Round-2 runs are
organic-only top-ups of the train tasks (more samples per task, higher token
caps on math); they append to the base, never replace it. Every candidate in
the merged output is re-labeled with the task's own checker
(``checkers.check``) and any mismatch is fatal.

Test-side extension runs (e.g. the math tasks that round 1 left without
coverage) are written to a **separate** file so the frozen organic test slice
stays untouched; evaluate them with the refit heads as an extension.

Usage, from the repo root:

    .venv/bin/python experiments/cascade/harvest_merge.py \
        --base experiments/cascade/results/decision-harvest-20261017/harvest-bench.jsonl \
        --run experiments/cascade/results/decision-harvest-r2-20261017/runs/nonmath \
        --run experiments/cascade/results/decision-harvest-r2-20261017/runs/math \
        --test-extension experiments/cascade/results/decision-harvest-r2-20261017/runs/testmath \
        --out experiments/cascade/results/decision-harvest-r2-20261017
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cascade.checkers import check  # noqa: E402

TASKS_FILE = REPO_ROOT / "experiments" / "cascade" / "tasks-bench.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def verify(items: list[dict], task_by_id: dict[str, dict], name: str) -> int:
    mismatched = []
    for item in items:
        task = task_by_id[item["task_id"]]
        got = int(bool(check(task, item["candidate"])))
        if got != item["label"]:
            mismatched.append((item["item_id"], item["label"], got))
    if mismatched:
        print(f"[merge] FATAL: {len(mismatched)} label mismatches in {name}:",
              mismatched[:10])
        raise SystemExit(1)
    print(f"[merge] {name}: {len(items)} labels re-verified, 0 mismatches")
    return len(items)


def split_stats(items: list[dict]) -> dict:
    per_split = collections.defaultdict(lambda: {"correct": 0, "wrong": 0})
    per_origin = collections.Counter()
    per_family = collections.defaultdict(lambda: {"correct": 0, "wrong": 0})
    for item in items:
        family = item["task_id"].split("-")[0]
        key = "correct" if item["label"] else "wrong"
        per_split[item["split"]][key] += 1
        per_family[family][key] += 1
        per_origin[item["origin"]] += 1
    return {
        "per_split": {k: dict(v) for k, v in sorted(per_split.items())},
        "per_family": {k: dict(v) for k, v in sorted(per_family.items())},
        "per_origin": dict(sorted(per_origin.items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, required=True,
                    help="round-1 harvest-bench.jsonl (kept as the base)")
    ap.add_argument("--run", type=Path, action="append", default=[],
                    help="round-2 run dir with harvest-bench.jsonl (repeatable)")
    ap.add_argument("--test-extension", type=Path, action="append", default=[],
                    help="test-side extension run dir (written separately)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tasks = json.loads(TASKS_FILE.read_text())["tasks"]
    task_by_id = {t["id"]: t for t in tasks}

    base = load_jsonl(args.base)
    verify(base, task_by_id, "base")

    merged = list(base)
    run_meta = []
    for run_dir in args.run:
        items = load_jsonl(run_dir / "harvest-bench.jsonl")
        bad = [i["item_id"] for i in items if i["split"] != "train"]
        if bad:
            print(f"[merge] FATAL: {run_dir} contains non-train items: {bad[:5]}")
            return 1
        verify(items, task_by_id, str(run_dir.name))
        merged.extend(items)
        run_manifest = run_dir / "harvest-manifest.json"
        run_meta.append({
            "dir": rel(run_dir),
            "items": len(items),
            "config": (json.loads(run_manifest.read_text())["config"]
                       if run_manifest.exists() else None),
        })

    ids = [i["item_id"] for i in merged]
    if len(ids) != len(set(ids)):
        dup = [k for k, v in collections.Counter(ids).items() if v > 1]
        print(f"[merge] FATAL: duplicate item ids: {dup[:10]}")
        return 1
    cands = collections.Counter((i["task_id"], i["candidate"]) for i in merged)
    dup_c = [k for k, v in cands.items() if v > 1]
    if dup_c:
        print(f"[merge] FATAL: duplicate candidates: {[(t, c[:40]) for t, c in dup_c[:5]]}")
        return 1

    merged.sort(key=lambda i: i["item_id"])
    payload = "\n".join(json.dumps(i, sort_keys=True) for i in merged) + "\n"
    stats = split_stats(merged)
    stats["wrongs_total"] = sum(1 for i in merged if i["label"] == 0)
    stats["items_total"] = len(merged)

    args.out.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.out / "harvest-bench-r2.jsonl"
    out_jsonl.write_text(payload)
    manifest = {
        "created": __import__("time").strftime("%Y-%m-%d", __import__("time").gmtime()),
        "purpose": ("round-1 harvest plus round-2 organic train top-up; every "
                    "label re-verified with the task checker"),
        "base": rel(args.base),
        "runs": run_meta,
        "items": len(merged),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        **stats,
    }
    (args.out / "harvest-manifest-r2.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[merge] wrote {out_jsonl} ({len(merged)} items, "
          f"{stats['wrongs_total']} wrong) and harvest-manifest-r2.json")

    if args.test_extension:
        ext: list[dict] = []
        ext_meta = []
        for run_dir in args.test_extension:
            items = load_jsonl(run_dir / "harvest-bench.jsonl")
            bad = [i["item_id"] for i in items if i["split"] != "test"]
            if bad:
                print(f"[merge] FATAL: {run_dir} contains non-test items: {bad[:5]}")
                return 1
            verify(items, task_by_id, str(run_dir.name))
            ext.extend(items)
            run_manifest = run_dir / "harvest-manifest.json"
            ext_meta.append({
                "dir": rel(run_dir),
                "items": len(items),
                "config": (json.loads(run_manifest.read_text())["config"]
                           if run_manifest.exists() else None),
            })
        ext.sort(key=lambda i: i["item_id"])
        ext_payload = "\n".join(json.dumps(i, sort_keys=True) for i in ext) + "\n"
        ext_jsonl = args.out / "harvest-testmath.jsonl"
        ext_jsonl.write_text(ext_payload)
        ext_manifest = {
            "created": manifest["created"],
            "purpose": ("test-side organic extension for the math tasks round 1 "
                        "left without coverage; NOT part of the training corpus"),
            "runs": ext_meta,
            "items": len(ext),
            "sha256": hashlib.sha256(ext_payload.encode()).hexdigest(),
            **split_stats(ext),
        }
        (args.out / "harvest-testmath-manifest.json").write_text(
            json.dumps(ext_manifest, indent=1) + "\n")
        print(f"[merge] wrote {ext_jsonl} ({len(ext)} extension items) and "
              f"harvest-testmath-manifest.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
