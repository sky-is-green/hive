#!/usr/bin/env python
"""The escalation gate, measured over every candidate judge.

The architecture under test: **the 35B generates, a judge says whether the
answer is right, and the 176B answers only when the judge sends it up.**

That makes the judge the most valuable thing in the cascade, because every
verdict it gets right saves a 176B generation (20.68 s measured) and every
verdict it gets wrong wastes one.  So the metrics that matter are not
"accuracy" but the gate's economics:

  - the errors it catches (a 176B generation avoided)
  - the correct answers it breaks (a 176B generation wasted)
  - what a verdict costs (against the 20.68 s it can save)

The candidates are judged on the *same* 28 organic tasks and the *same*
35B answers, so the arms are paired.  The 176B's own decision head is an
arm too (its p_correct is read from the saved measurement, so the arm is
exactly what was measured - the head is deterministic).

Usage::

    ../.venv-rocm/bin/python -m experiments.judge_gate --arm jev
    ../.venv-rocm/bin/python -m experiments.judge_gate --arm tinyjev --arm intern
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import check  # noqa: E402

#: the campaign arm: the 35B's own answers to the 28 organic tasks, with
#: the labels the tasks' own checkers gave them.
ARM = REPO / "experiments/cascade/results/cascade-arm-escalation-20261020/report.json"
TASKS = REPO / "experiments/cascade/tasks-organic-28.json"
#: the 176B's own head, measured on the same answers
#: (experiments/heads_on_the_cascade.py)
HEAD176 = REPO / "experiments/cascade/results/heads-on-the-cascade-20261021.json"
#: what a 176B generation costs, measured (experiments/escalation_arms.py)
GENERATION_S = 20.68


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for a in pos for b in neg if a > b) + \
        0.5 * sum(1 for a in pos for b in neg if a == b)
    return round(wins / (len(pos) * len(neg)), 4)


def load_items() -> list[dict[str, Any]]:
    recs = json.loads(ARM.read_text())["records"]
    tasks = {t["id"]: t for t in json.loads(TASKS.read_text())["tasks"]}
    out = []
    for r in recs:
        task = tasks.get(r["id"])
        cand = (r.get("scion_answer") or "").strip()
        if task is None or not cand:
            continue
        out.append({"id": r["id"], "task": task, "candidate": cand,
                    "ok": bool(r.get("scion_ok")), "bucket": r.get("bucket") or ""})
    return out


def judge_for(arm: str, model: str, device: str):
    """The judge candidate for an arm."""
    if arm == "tinyjev":
        from harness.cascade.validator import TinyJevValidator
        return TinyJevValidator(model, device=device)
    if arm == "intern":
        from experiments.cascade.decision import InternDecisionJudge
        return InternDecisionJudge(model, device=device)
    if arm == "jev":
        from experiments.cascade.decision import Jev9BJudge
        return Jev9BJudge(model, device=device)
    if arm == "clef":
        from harness.cascade.validator import ClefValidator
        return ClefValidator(model, device=device)
    if arm == "ternary":
        from harness.cascade.validator import ClefTernaryValidator
        return ClefTernaryValidator(model, device=device)
    raise SystemExit(f"unknown arm {arm!r}")


def score(judge, items: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    ps, ms = [], []
    for it in items:
        p, t = judge.verdict(it["task"], it["candidate"])
        ps.append(float(p))
        ms.append(float(t))
    return ps, ms


def report(name: str, ps: list[float], ms: list[float], items: list[dict[str, Any]],
           extra: Optional[dict] = None) -> dict:
    pos = [p for p, it in zip(ps, items) if it["ok"]]
    neg = [p for p, it in zip(ps, items) if not it["ok"]]
    out: dict[str, Any] = {
        "the_judge": name,
        "the_n": len(items),
        "the_auc": auc(pos, neg),
        "the_mean_p_on_the_correct": round(statistics.mean(pos), 4) if pos else None,
        "the_mean_p_on_the_wrong": round(statistics.mean(neg), 4) if neg else None,
        "the_mean_ms": round(statistics.mean(ms), 1) if ms else None,
        "the_p_correct": {it["id"]: round(p, 6) for it, p in zip(items, ps)},
        "the_sweep": [],
    }
    if extra:
        out.update(extra)
    for step in range(5, 100, 5):
        thr = step / 100.0
        caught = [it for p, it in zip(ps, items) if not it["ok"] and p < thr]
        broken = [it for p, it in zip(ps, items) if it["ok"] and p < thr]
        kept = [it for p, it in zip(ps, items) if it["ok"] and p >= thr]
        out["the_sweep"].append({
            "the_threshold": thr,
            "the_errors_caught": len(caught),
            "the_errors": len(neg),
            "the_correct_broken": len(broken),
            "the_correct_kept": len(kept),
            "the_correct": len(pos),
        })
    return out


def the_best_operating_point(sweep: list[dict]) -> dict:
    """The threshold that catches the most errors without breaking a correct answer.

    The gate's economics: a broken correct answer costs a wasted 176B
    generation; a caught error saves one.  So the operating point that
    maximises (caught - broken) is the one to ship, and the "no leak" one
    (zero broken) is the safe one.
    """
    no_leak = next((s for s in sweep if s["the_correct_broken"] == 0
                    and s["the_errors_caught"] > 0), None)
    best = max(sweep, key=lambda s: (s["the_errors_caught"] - s["the_correct_broken"],
                                     s["the_errors_caught"]))
    return {"the_best_net": best, "the_no_leak": no_leak}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[],
                    help="the judge candidates: tinyjev, intern, jev, clef, ternary")
    ap.add_argument("--model", default="",
                    help="the model id (the default is the arm's own)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    items = load_items()
    print(f"the labelled set: {len(items)} answers "
          f"({sum(1 for it in items if it['ok'])} correct, "
          f"{sum(1 for it in items if not it['ok'])} wrong)", flush=True)

    defaults = {"tinyjev": "lostargon/Tiny-Jev-1.7B",
                "intern": "internlm/Intern-Decision-0.8B",
                "jev": "autotrust/JEV-9B",
                "clef": "Cloudflare/clef-flash",
                "ternary": ""}
    arms = args.arm or ["tinyjev", "intern", "jev"]
    results = []

    # the 176B's own head, from the saved measurement
    if Path(HEAD176).is_file():
        saved = json.loads(Path(HEAD176).read_text())
        by_id = {r["id"]: r for r in saved["rows"]}
        head_items = [it for it in items if it["id"] in by_id]
        ps = [by_id[it["id"]]["the J-Bench head"] for it in head_items]
        ms = [by_id[it["id"]]["the_readout_s"] * 1000.0 for it in head_items]
        results.append(report("the 176B's own head", ps, ms, head_items,
                              {"the_source": "the saved measurement (a prefill)"}))

    for arm in arms:
        model = args.model or defaults.get(arm, "")
        if not model:
            continue
        print(f"\n=== the {arm} arm: {model}", flush=True)
        try:
            judge = judge_for(arm, model, args.device)
        except Exception as exc:  # noqa: BLE001
            print(f"  it could not be loaded: {exc}", file=sys.stderr)
            continue
        t0 = time.time()
        ps, ms = score(judge, items)
        print(f"  scored {len(ps)} answers in {time.time() - t0:.1f}s", flush=True)
        results.append(report(f"the {arm} judge ({model.split('/')[-1]})", ps, ms, items))

    for r in results:
        r.update(the_best_operating_point(r["the_sweep"]))

    print("\n" + json.dumps({"the_arms": results}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"the_arms": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
