#!/usr/bin/env python3
"""Score the repair pass twice: by the registered checker, and by answer content.

Why twice: the bench's `label` and this run's `gold_ok_after` both use the task's registered
checker.  For `boxed` tasks that checker only reads ``\\boxed{...}``, so a repaired answer that
states the same, correct value as "Final answer: 3600" scores as wrong.  That is the interface
being strict, not the model being wrong, and the two must not be mixed -- the first number tells
us whether the harvested artifact would still be consumed as correct, the second whether the
model's *answer* is correct.  The tolerant comparator is still gold-based and stays eval-only.

    .venv/bin/python experiments/cascade/analyze_repair.py \
        --run experiments/cascade/results/repair-pass-20261018/repair.jsonl \
        --out experiments/cascade/results/repair-pass-20261018/SCORING.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from checkers import _boxed, _math_equal  # noqa: E402

FINAL_NUM = re.compile(r"final answer[^0-9\-+]*([-+]?\d[\d,]*(?:\.\d+)?)", re.I | re.S)
LETTER = re.compile(r"\b([A-D])\b")
BOXED = re.compile(r"\\boxed\{(.+?)\}", re.S)


def extract(text: str) -> str | None:
    m = BOXED.findall(text or "")
    if m:
        return m[-1].strip()
    m = FINAL_NUM.findall(text or "")
    if m:
        return m[-1].replace(",", "")
    m = LETTER.findall((text or "").upper())
    return m[-1] if m else None


def tolerant_ok(spec: dict, text: str) -> bool | None:
    """Compare the extracted final answer with the spec's expected value, ignoring framing."""
    if not text:
        return False
    ck = spec["checker"]
    kind = ck["type"]
    if kind == "code":
        return None  # the strict run already executed the tests; framing is not a variable here
    got = extract(text)
    if got is None:
        return False
    if kind == "number":
        try:
            return abs(float(got) - float(ck["expect"])) <= float(ck.get("tol", 1e-6))
        except ValueError:
            return False
    if kind == "boxed":
        try:
            return _math_equal(got, str(ck["expect"]))
        except Exception:  # noqa: BLE001
            return False
    if kind == "choice":
        return str(got).upper() == str(ck["expect"]).upper()
    if kind == "contains":
        return any(str(e).lower() in (text or "").lower() for e in ck["expect"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = REPO / "experiments" / "cascade"
    ap.add_argument("--run", type=Path,
                    default=root / "results" / "repair-pass-20261018" / "repair.jsonl")
    ap.add_argument("--bench", type=Path,
                    default=root / "results" / "dual-generation-20261018" / "organic-66.jsonl")
    ap.add_argument("--tasks", type=Path, nargs="+", default=[root / "tasks-bench.json"])
    ap.add_argument("--out", type=Path,
                    default=root / "results" / "repair-pass-20261018" / "SCORING.md")
    args = ap.parse_args()

    specs: dict[str, dict] = {}
    for f in args.tasks:
        d = json.loads(f.read_text())
        for it in (d.get("items") or d.get("tasks") or []):
            if "checker" in it:
                specs.setdefault(str(it.get("id") or it.get("task_id")), it)

    rows = [json.loads(l) for l in args.run.read_text().splitlines() if l.strip()]
    for r in rows:
        spec = specs.get(r["task_id"])
        r["kind"] = (spec or {}).get("checker", {}).get("type")
        if r["kind"] == "code":
            # For code, running the tests *is* the content comparison: there is no framing to
            # forgive, the program either satisfies the task or it does not.
            r["tolerant"] = r.get("gold_ok_after")
        else:
            r["tolerant"] = tolerant_ok(spec, r.get("text") or "") if spec else None

    lines = [
        "# Repair pass: two scorings (2026-10-18)",
        "",
        "`strict` is the task's registered checker (what `label` and `gold_ok_after` use): for "
        "`boxed` tasks it requires `\\boxed{...}`.  `content` extracts the final answer (box, else "
        "'Final answer: n', else letter) and compares it with the expected value -- the same "
        "extraction the harvest's `final` column used.  Both are gold-based and eval-only.",
        "",
        "| arm | n | strict repaired | content repaired | strict harm | content harm |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm, label in (("A", "evidence repair"), ("B", "generic recheck"), ("C", "harm control")):
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        s_rep = sum(1 for r in sub if r.get("gold_ok_after") is True)
        c_rep = sum(1 for r in sub if r["tolerant"] is True)
        s_harm = sum(1 for r in sub if r.get("gold_ok_after") is False and r["orig_label"] == 1)
        c_harm = sum(1 for r in sub if r["tolerant"] is False and r["orig_label"] == 1)
        lines.append(f"| {arm} {label} | {len(sub)} | {s_rep}/{len(sub)} | {c_rep}/{len(sub)} | "
                     f"{s_harm}/{len(sub)} | {c_harm}/{len(sub)} |")

    artifacts = [r for r in rows if r.get("gold_ok_after") is not r["tolerant"]
                 and r["kind"] != "code"]
    lines += ["", "## Where the two scorings disagree", ""]
    if artifacts:
        lines += ["| item | arm | strict | content | expected | new final |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in artifacts:
            exp = specs.get(r["task_id"], {}).get("checker", {}).get("expect")
            lines.append(f"| `{r['item_id']}` | {r['arm']} | "
                         f"{'pass' if r.get('gold_ok_after') else 'fail'} | "
                         f"{'pass' if r['tolerant'] else 'fail'} | {exp} | "
                         f"{extract(r.get('text') or '') or '-'} |")
    else:
        lines.append("none")
    lines += ["", "## Per item", "",
              "| item | arm | kind | finding | strict | content | flags after | tok | s |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(rows, key=lambda r: (r["item_id"], r["arm"])):
        lines.append(f"| `{r['item_id']}` | {r['arm']} | {r.get('kind') or '-'} | "
                     f"{r.get('detail') or '-'} | "
                     f"{'pass' if r.get('gold_ok_after') else 'fail'} | "
                     f"{'pass' if r['tolerant'] else ('n/a' if r['tolerant'] is None else 'fail')} | "
                     f"{'yes' if r.get('flags_after') else 'no'} | "
                     f"{r.get('completion_tokens') or '-'} | {r.get('ms', 0) / 1000:.1f} |")
    args.out.write_text("\n".join(lines) + "\n")

    print("\n".join(lines[:14]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
