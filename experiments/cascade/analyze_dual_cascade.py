#!/usr/bin/env python3
"""Read a dual-generation run and score the cascade, not the models.

`dual_generation.py` answers "how do the two models compare".  This answers the question the
cascade actually poses: with the gold-free gates as the only accept signal, how much of the work
can each tier take, what does an escalation cost, and which order of the two tiers is cheaper.

Gold is used for *reporting only* (this is an eval script).  Nothing here runs in the serving path.

The labels in this bench are "the 176B answered this correctly", not "this is solvable": a
label-0 item is one the big model failed, and a small model passing it is a win, not a false
accept.  So the two label classes are reported apart and never averaged.

    .venv/bin/python experiments/cascade/analyze_dual_cascade.py \
        results/dual-generation-20261018/full-66
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_generation import final_answer
from consistency_check import check_answer


def load(out: Path) -> tuple[list[dict], str, str]:
    rows = [json.loads(l) for l in (out / "dual-generation.jsonl").read_text().splitlines() if l.strip()]
    summ = json.loads((out / "dual-generation-summary.json").read_text())
    a, b = summ["models"]
    # The gate is recomputed from the stored text rather than read back from the run: the verifier
    # improves, and an analysis pinned to the flags a past run happened to record would keep scoring
    # a gate the project no longer has.  (This run's recorded flags predate the current verifier.)
    for r in rows:
        for name in (a, b):
            r[name]["consistency_flag"] = check_answer(r[name].get("text", ""))["flagged"]
    return rows, a, b


def accepted(rec: dict) -> bool:
    """The deployable accept at a tier: the gold-free verifier did not flag it.

    Deliberately nothing else.  `checker_pass` in these rows is the task's gold comparator
    (`numeric_gold`), which is eval-only: a gate that consulted it would be reading the answer key.
    """
    return not rec.get("consistency_flag")


def truncated(rec: dict) -> bool:
    return rec.get("finish_reason") == "length"


def tier_report(rows: list[dict], name: str, label: int) -> dict:
    sub = [r for r in rows if r["label"] == label]
    n = len(sub)
    acc = [r for r in sub if accepted(r[name])]
    gold = [r for r in sub if r[name].get("checker_pass")]
    fails = [r for r in sub if not accepted(r[name])]
    return {
        "items": n,
        "gold_correct": len(gold),
        "accepted": len(acc),
        "accept_rate": len(acc) / n if n else None,
        # what the gate let through that was wrong, and what it rejected that was right
        "accepted_but_wrong": sum(1 for r in acc if not r[name].get("checker_pass")),
        "rejected_but_correct": sum(1 for r in sub
                                    if not accepted(r[name]) and r[name].get("checker_pass")),
        "failed": len(fails),
        "failed_truncated": sum(1 for r in fails if truncated(r[name])),
        "failed_flagged": sum(1 for r in fails if r[name].get("consistency_flag")),
        "failed_silent": sum(1 for r in fails
                             if not truncated(r[name]) and not r[name].get("consistency_flag")),
        "no_final": sum(1 for r in sub if r[name].get("final") is None),
    }


def cascade(rows: list[dict], first: str, second: str, label: int) -> dict:
    """Two-tier flow at one label class: tier 1 takes what it can accept, the rest escalate."""
    sub = [r for r in rows if r["label"] == label]
    t1_ok = [r for r in sub if accepted(r[first])]
    esc = [r for r in sub if not accepted(r[first])]
    t2_ok = [r for r in esc if accepted(r[second])]
    dead = [r for r in esc if not accepted(r[second])]
    return {
        "items": len(sub),
        "tier1_accepts": len(t1_ok),
        "escalated": len(esc),
        "escalation_rate": len(esc) / len(sub) if sub else None,
        "tier2_resolves": len(t2_ok),
        "unresolved": len(dead),
        "unresolved_truncated": sum(1 for r in dead if truncated(r[first]) or truncated(r[second])),
        "unresolved_ids": [r["item_id"] for r in dead][:12],
    }


def cost(rows: list[dict], flow: str, first: str, second: str, label: int, swap_s: float) -> dict:
    """Wall time of the flow vs always-using either tier, in seconds, from the measured per-item ms.

    An escalation is charged `swap_s` once, for the model swap the deployment pays when tier 2 is
    not resident: with the 176B holding both cards, only one tier is ever loaded at a time.
    """
    sub = [r for r in rows if r["label"] == label]
    always_1 = sum(r[first]["ms"] for r in sub) / 1000
    always_2 = sum(r[second]["ms"] for r in sub) / 1000
    if flow == "cascade":
        total = 0.0
        for r in sub:
            total += r[first]["ms"] / 1000
            if not accepted(r[first]):
                total += r[second]["ms"] / 1000 + swap_s
        return {"always_first_s": always_1, "always_second_s": always_2,
                "cascade_s": total, "swaps": sum(1 for r in sub if not accepted(r[first]))}
    return {"always_first_s": always_1, "always_second_s": always_2}


def bench_costs(bench: Path) -> dict[str, dict]:
    """Per-item cost and answer of the model the labels came from — the 176B, from the harvest.

    The bench row keeps the big model's own candidate and its `gen` block, so the comparison
    "cascade vs just ask the big one" is measured on the same items, not modelled.
    """
    out: dict[str, dict] = {}
    for line in bench.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        g = r.get("gen") or {}
        out[r["item_id"]] = {"ms": g.get("ms"), "tokens": g.get("completion_tokens"),
                             "final": final_answer(r.get("candidate") or ""),
                             "label": r.get("label")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--bench", type=Path, default=None,
                    help="the bench the run read, for the 176B's own per-item cost (harvest rows)")
    ap.add_argument("--swap-s", type=float, default=32.0,
                    help="seconds to swap a tier in (the 176B serve comes healthy in ~32 s)")
    args = ap.parse_args()

    rows, a, b = load(args.out)
    rep = {"out": str(args.out), "models": [a, b], "items": len(rows), "tiers": {}, "cascade": {}}

    for name in (a, b):
        rep["tiers"][name] = {
            "gold_accuracy": sum(1 for r in rows if r[name].get("checker_pass")) / len(rows),
            "gold_accuracy_where_176b_solved":
                sum(1 for r in rows if r["label"] == 1 and r[name].get("checker_pass")) /
                sum(1 for r in rows if r["label"] == 1),
            "solved_by_176b": tier_report(rows, name, 1),
            "failed_by_176b": tier_report(rows, name, 0),
            "tokens_median": sorted((r[name].get("completion_tokens") or 0) for r in rows)[len(rows) // 2],
            "ms_median": sorted(r[name]["ms"] for r in rows)[len(rows) // 2],
        }

    for first, second, tag in ((a, b, f"{a}_then_{b}"), (b, a, f"{b}_then_{a}")):
        rep["cascade"][tag] = {
            "on_solved_by_176b": cascade(rows, first, second, 1),
            "on_failed_by_176b": cascade(rows, first, second, 0),
            "cost_on_solved": cost(rows, "cascade", first, second, 1, args.swap_s),
        }

    agree = [r for r in rows if r["agree"] is not None]
    rep["agreement"] = {
        "both_answered": len(agree),
        "agreed": sum(1 for r in agree if r["agree"]),
        "disagreed": sum(1 for r in agree if r["agree"] is False),
        "on_disagreements_176b_solved": sum(1 for r in agree
                                            if r["agree"] is False and r["label"] == 1),
    }

    if args.bench:
        big = bench_costs(args.bench)
        have = [r for r in rows if r["item_id"] in big and big[r["item_id"]]["ms"]]
        solved = [r for r in have if big[r["item_id"]]["label"] == 1]
        mss = sorted(big[r["item_id"]]["ms"] for r in have)
        toks = sorted(big[r["item_id"]]["tokens"] or 0 for r in have)
        rep["big_model_176b"] = {
            "items": len(have),
            "ms_median": mss[len(mss) // 2],
            "ms_mean": sum(mss) / len(mss),
            "tokens_median": toks[len(toks) // 2],
            "total_s_on_solved": sum(big[r["item_id"]]["ms"] for r in solved) / 1000,
            "same_final_as_big": {
                name: {
                    "same": sum(1 for r in have
                                if r[name].get("final") is not None
                                and r[name]["final"] == big[r["item_id"]]["final"]),
                    "both_answered": sum(1 for r in have
                                         if r[name].get("final") is not None
                                         and big[r["item_id"]]["final"] is not None),
                } for name in (a, b)
            },
        }
        for block in rep["cascade"].values():
            c = block.get("cost_on_solved") or {}
            if c:
                c["always_176b_s"] = rep["big_model_176b"]["total_s_on_solved"]
                c["saved_vs_176b_s"] = c["always_176b_s"] - c["cascade_s"]

    print(json.dumps(rep, indent=1))
    (args.out / "cascade-analysis.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
