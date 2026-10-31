#!/usr/bin/env python
"""Gate a queued multi-role checkpoint against the protected roles.

Every batch is evaluated on the new roles **and** every protected role. The
gate fails if an earlier role loses more than the allowed number of tasks, if a
new role misses its minimum, or if the GGUF/HF gap is too large.

    python experiments/role_gate.py \
      --batch placement-critic \
      --candidate-dir ../train-runs/multirole-pc-v1 \
      --out ../train-runs/multirole-pc-v1/gate-report.json

Exit code 0 means the batch may advance to a canary; it does **not** promote
the live endpoint. Promotion is a separate, approved step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUEUE = REPO / "experiments" / "role_queue.json"


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def eval_files(directory: Path, role: str) -> tuple[Path, Path]:
    return directory / f"eval-{role}.json", directory / f"eval-{role}-gguf.json"


def overall(eval_data: dict | None) -> float | None:
    if not eval_data:
        return None
    value = (eval_data.get("overall") or {}).get("accuracy")
    if value is not None:
        return float(value)
    rows = eval_data.get("rows") or []
    if not rows:
        return None
    return sum(bool(row.get("ok")) for row in rows) / len(rows)


def original_score(eval_data: dict | None, task_ids: set[str]) -> tuple[int, int] | None:
    if not eval_data:
        return None
    rows = {row.get("task_id"): row for row in eval_data.get("rows") or []}
    present = [task_id for task_id in task_ids if task_id in rows]
    if not present:
        return None
    return sum(bool(rows[task_id].get("ok")) for task_id in present), len(present)


def role_task_ids(tasks_path: Path, role: str) -> set[str]:
    tasks = json.loads(tasks_path.read_text()).get("tasks") or []
    return {task["id"] for task in tasks if task.get("role") == role}


def add_check(checks: list[dict], reasons: list[str], name: str, ok: bool,
              detail: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    if not ok:
        reasons.append(f"{name}: {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default=str(QUEUE))
    parser.add_argument("--batch", default="")
    parser.add_argument("--roles", default="",
                        help="override the batch roles, comma-separated")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--baseline-dir", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    queue = json.loads(Path(args.queue).read_text())
    batch = None
    if args.batch:
        batch = next((b for b in queue.get("batches", []) if b["id"] == args.batch), None)
        if batch is None:
            raise SystemExit(f"unknown batch {args.batch!r}")
    roles = [r.strip() for r in args.roles.split(",") if r.strip()] if args.roles else []
    if not roles and batch:
        roles = list(batch.get("roles") or [])
    if not roles:
        raise SystemExit("no roles to gate")

    gate = queue.get("gate") or {}
    protected = ["controller", "planner"]
    all_roles = list(dict.fromkeys(protected + roles))
    candidate_dir = Path(args.candidate_dir).resolve()
    baseline_dir = Path(args.baseline_dir or queue.get("baseline_dir") or "").resolve()
    tasks_path = (REPO / queue["tasks"]["original"]).resolve()

    checks: list[dict] = []
    reasons: list[str] = []
    report_roles: dict[str, dict] = {}

    for role in all_roles:
        hf_path, gguf_path = eval_files(candidate_dir, role)
        hf = load_json(hf_path)
        gguf = load_json(gguf_path)
        hf_acc = overall(hf)
        gguf_acc = overall(gguf)
        role_report = {
            "hf_eval": str(hf_path) if hf else None,
            "gguf_eval": str(gguf_path) if gguf else None,
            "hf_accuracy": hf_acc,
            "gguf_accuracy": gguf_acc,
        }

        if hf_acc is None:
            add_check(checks, reasons, f"{role}:hf_present", False,
                      f"missing or unreadable {hf_path}")
        if gate.get("require_gguf", True) and gguf_acc is None:
            add_check(checks, reasons, f"{role}:gguf_present", False,
                      f"missing or unreadable {gguf_path}")

        if hf_acc is not None and gguf_acc is not None:
            gap = hf_acc - gguf_acc
            role_report["gguf_gap"] = round(gap, 4)
            add_check(checks, reasons, f"{role}:gguf_gap", gap <= gate.get("gguf_max_gap", 0.05),
                      f"HF {hf_acc:.3f} vs GGUF {gguf_acc:.3f} (max gap "
                      f"{gate.get('gguf_max_gap', 0.05):.3f})")

        if role in protected:
            base_hf_path, base_gguf_path = eval_files(baseline_dir, role)
            base_hf = load_json(base_hf_path)
            base_gguf = load_json(base_gguf_path)
            base_hf_acc = overall(base_hf)
            base_gguf_acc = overall(base_gguf)
            task_ids = role_task_ids(tasks_path, role)
            cand_hf_orig = original_score(hf, task_ids)
            cand_gguf_orig = original_score(gguf, task_ids)
            base_hf_orig = original_score(base_hf, task_ids)
            base_gguf_orig = original_score(base_gguf, task_ids)
            role_report["original_tasks"] = len(task_ids)
            role_report["hf_original"] = cand_hf_orig
            role_report["gguf_original"] = cand_gguf_orig
            role_report["baseline_hf_original"] = base_hf_orig
            role_report["baseline_gguf_original"] = base_gguf_orig

            if role == "controller":
                orig_min = int(gate.get("controller", {}).get("orig_min", 9))
            else:
                orig_min = None
            max_loss = float(gate.get(role, {}).get("wider_max_loss", 2))

            if base_hf_acc is None or base_gguf_acc is None:
                add_check(checks, reasons, f"{role}:baseline_present", False,
                          f"missing baseline eval in {baseline_dir}")
            else:
                if hf_acc is not None:
                    # Use absolute task loss instead of a fraction for wider suites.
                    cand_n = len((hf or {}).get("rows") or [])
                    base_n = len((base_hf or {}).get("rows") or [])
                    floor_acc = (round(base_hf_acc * base_n) - max_loss) / max(1, cand_n)
                    add_check(checks, reasons, f"{role}:hf_wider", hf_acc + 1e-9 >= floor_acc,
                              f"HF {hf_acc:.3f} >= {floor_acc:.3f} "
                              f"(baseline {base_hf_acc:.3f}, max loss {max_loss:g})")
                if gguf_acc is not None:
                    base_n = len((base_gguf or {}).get("rows") or [])
                    cand_n = len((gguf or {}).get("rows") or [])
                    floor_acc = (round(base_gguf_acc * base_n) - max_loss) / max(1, cand_n)
                    add_check(checks, reasons, f"{role}:gguf_wider", gguf_acc + 1e-9 >= floor_acc,
                              f"GGUF {gguf_acc:.3f} >= {floor_acc:.3f} "
                              f"(baseline {base_gguf_acc:.3f}, max loss {max_loss:g})")

                if role == "controller" and orig_min is not None:
                    if cand_hf_orig:
                        add_check(checks, reasons, "controller:hf_original",
                                  cand_hf_orig[0] >= orig_min,
                                  f"{cand_hf_orig[0]}/{cand_hf_orig[1]} >= {orig_min}")
                    if cand_gguf_orig:
                        add_check(checks, reasons, "controller:gguf_original",
                                  cand_gguf_orig[0] >= orig_min,
                                  f"{cand_gguf_orig[0]}/{cand_gguf_orig[1]} >= {orig_min}")
                if role == "planner":
                    hf_min = int(gate.get("planner", {}).get("orig_min_hf", 16))
                    gguf_min = int(gate.get("planner", {}).get("orig_min_gguf", 15))
                    if cand_hf_orig:
                        add_check(checks, reasons, "planner:hf_original",
                                  cand_hf_orig[0] >= hf_min,
                                  f"{cand_hf_orig[0]}/{cand_hf_orig[1]} >= {hf_min}")
                    if cand_gguf_orig:
                        add_check(checks, reasons, "planner:gguf_original",
                                  cand_gguf_orig[0] >= gguf_min,
                                  f"{cand_gguf_orig[0]}/{cand_gguf_orig[1]} >= {gguf_min}")
        else:
            minimum = float((gate.get("new_role_min") or {}).get(role, 0.9))
            if hf_acc is not None:
                add_check(checks, reasons, f"{role}:hf_min", hf_acc + 1e-9 >= minimum,
                          f"HF {hf_acc:.3f} >= {minimum:.3f}")
            if gguf_acc is not None:
                add_check(checks, reasons, f"{role}:gguf_min", gguf_acc + 1e-9 >= minimum,
                          f"GGUF {gguf_acc:.3f} >= {minimum:.3f}")

        report_roles[role] = role_report

    passed = not reasons
    report = {
        "batch": args.batch or "custom",
        "roles": all_roles,
        "candidate_dir": str(candidate_dir),
        "baseline_dir": str(baseline_dir),
        "passed": passed,
        "checks": checks,
        "reasons": reasons,
        "roles_report": report_roles,
    }
    print(f"gate {'PASS' if passed else 'FAIL'}: {args.batch or 'custom'} "
          f"({', '.join(all_roles)})")
    for check in checks:
        print(f"  {'ok  ' if check['ok'] else 'FAIL'} {check['name']:28s} {check['detail']}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print("wrote", out)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
