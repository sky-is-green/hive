#!/usr/bin/env python
"""Gate the role -> planner -> role curriculum recovery test.

Compares a candidate against the failed 3-role model (`multirole-cpp-v1`) and
the known Scion baselines. The candidate must recover controller and planner
while keeping placement and improving placement planning.

    python experiments/curriculum_gate.py \
      --candidate-dir ../train-runs/multirole-cpp-plan-v1 \
      --before-dir ../train-runs/multirole-cpp-v1 \
      --out ../train-runs/multirole-cpp-plan-v1/curriculum-gate.json

Exit 0 means the curriculum hypothesis is supported; it does not promote the
live endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def accuracy(data: dict | None) -> float | None:
    if not data:
        return None
    value = (data.get("overall") or {}).get("accuracy")
    if value is not None:
        return float(value)
    rows = data.get("rows") or []
    if not rows:
        return None
    return sum(bool(r.get("ok")) for r in rows) / len(rows)


def count_ok(data: dict | None) -> tuple[int, int] | None:
    if not data:
        return None
    rows = data.get("rows") or []
    if not rows:
        return None
    return sum(bool(r.get("ok")) for r in rows), len(rows)


def original(data: dict | None, ids: set[str]) -> tuple[int, int] | None:
    if not data:
        return None
    rows = {r.get("task_id"): r for r in data.get("rows") or []}
    present = [i for i in ids if i in rows]
    if not present:
        return None
    return sum(bool(rows[i].get("ok")) for i in present), len(present)


def role_ids(tasks_path: Path, role: str) -> set[str]:
    tasks = json.loads(tasks_path.read_text()).get("tasks") or []
    return {t["id"] for t in tasks if t.get("role") == role}


def check(checks: list[dict], reasons: list[str], name: str, ok: bool,
          detail: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    if not ok:
        reasons.append(f"{name}: {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--before-dir", default=str(REPO.parent / "train-runs" / "multirole-cpp-v1"))
    parser.add_argument("--tasks", default=str(REPO / "experiments" / "roles" / "tasks-roles.json"))
    parser.add_argument("--out", default="")
    parser.add_argument("--require-gguf", action="store_true")
    args = parser.parse_args(argv)

    cand = Path(args.candidate_dir).resolve()
    before = Path(args.before_dir).resolve()
    tasks_path = Path(args.tasks).resolve()
    controller_ids = role_ids(tasks_path, "controller")
    planner_ids = role_ids(tasks_path, "planner")

    checks: list[dict] = []
    reasons: list[str] = []
    report: dict[str, dict] = {}

    specs = [
        ("controller", "eval-controller.json", 0.95, 0.9, controller_ids),
        ("planner", "eval-planner.json", 0.87, 0.87, planner_ids),
        ("placement", "eval-placement.json", 0.90, 0.90, set()),
        ("placement-plan", "eval-placement-plan.json", 0.90, 0.90, set()),
    ]
    for role, filename, min_acc, min_acc_before, ids in specs:
        c = load(cand / filename)
        b = load(before / filename)
        c_acc = accuracy(c)
        b_acc = accuracy(b)
        c_cnt = count_ok(c)
        c_orig = original(c, ids) if ids else None
        report[role] = {"before": b_acc, "after": c_acc,
                        "before_count": count_ok(b), "after_count": c_cnt,
                        "original": c_orig}
        if c_acc is None:
            check(checks, reasons, f"{role}:present", False,
                  f"missing {cand / filename}")
            continue
        check(checks, reasons, f"{role}:min", c_acc + 1e-9 >= min_acc,
              f"{c_acc:.4f} >= {min_acc:.3f}")
        if b_acc is not None and role == "placement-plan" and b_acc + 1e-9 < min_acc:
            # Require improvement only while the skill is below the bar. An
            # anchor run on an already-perfect planner should not fail here.
            check(checks, reasons, "placement-plan:improved", c_acc > b_acc + 1e-9,
                  f"{c_acc:.4f} > before {b_acc:.4f}")
        if role == "controller":
            check(checks, reasons, "controller:original", c_orig is not None and c_orig[0] >= 9,
                  f"{c_orig[0]}/{c_orig[1]} >= 9" if c_orig else "missing original rows")
        if role == "planner":
            check(checks, reasons, "planner:original", c_orig is not None and c_orig[0] >= 16,
                  f"{c_orig[0]}/{c_orig[1]} >= 16" if c_orig else "missing original rows")

        if args.require_gguf:
            g = load(cand / filename.replace(".json", "-gguf.json"))
            g_acc = accuracy(g)
            report[role]["after_gguf"] = g_acc
            if g_acc is None:
                check(checks, reasons, f"{role}:gguf", False, "missing GGUF eval")
            else:
                check(checks, reasons, f"{role}:gguf_min", g_acc + 1e-9 >= min_acc,
                      f"{g_acc:.4f} >= {min_acc:.3f}")
                check(checks, reasons, f"{role}:gguf_gap", c_acc - g_acc <= 0.05,
                      f"HF {c_acc:.4f} vs GGUF {g_acc:.4f}")

    scion_controller = 50 / 60
    scion_planner_orig = (15, 17)
    report["scion"] = {"controller": scion_controller,
                       "planner_original": scion_planner_orig}
    passed = not reasons
    payload = {"candidate_dir": str(cand), "before_dir": str(before),
               "passed": passed, "checks": checks, "reasons": reasons,
               "roles": report}
    print(f"curriculum gate {'PASS' if passed else 'FAIL'}")
    for row in checks:
        print(f"  {'ok  ' if row['ok'] else 'FAIL'} {row['name']:28s} {row['detail']}")
    print("\nbefore -> after")
    for role, data in report.items():
        if role == "scion":
            continue
        print(f"  {role:16s} {data.get('before')} -> {data.get('after')} "
              f"(original {data.get('original')})")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print("wrote", out)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
