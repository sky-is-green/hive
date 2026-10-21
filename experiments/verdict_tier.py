#!/usr/bin/env python3
"""The 176B's own decision head, judging the 35B's answers.

The campaign escalated to the 176B as a *generator*: the small tier answers,
the verifier flags it, and the 176B answers again.  That costs a full
generation - 27 s warm, 230 s co-sited.

The 176B also has a *readout* head (AUC 0.994 on the frozen judge bench,
`SkyIsNotGreen/decision-heads`): the last prompt token's hidden state, read
through the engine's `emb=`, predicts whether a candidate answer is right.
That is a **prefill**, not a generation: 2.4 s against 27 s.

This measures what that buys, on the campaign's own 28 organic tasks and
the 35B's own answers to them:

  - how well the head separates the 35B's right answers from its wrong ones
  - what a verdict-only escalation would have cost
  - how many of the 35B's errors the head would have caught

Usage: verdict_tier.py [--limit N] [--out JSON]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARM = REPO / "experiments/cascade/results/cascade-arm-escalation-20261020/report.json"
TASKS = REPO / "experiments/cascade/tasks-organic-28.json"
VERDICT_URL = "http://127.0.0.1:8081/v1/verdict"


def verdict(task: str, candidate: str) -> dict:
    body = json.dumps({"task": task, "candidate": candidate}).encode()
    req = urllib.request.Request(VERDICT_URL, data=body,
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    d["wall_s"] = round(time.time() - t0, 2)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    recs = json.loads(ARM.read_text())["records"]
    tasks = {t["id"]: t for t in json.loads(TASKS.read_text())["tasks"]}
    if args.limit:
        recs = recs[: args.limit]

    rows = []
    for r in recs:
        task = tasks.get(r["id"])
        if task is None:
            continue
        cand = r.get("scion_answer") or ""
        if not cand.strip():
            continue
        v = verdict(task["prompt"], cand)
        rows.append({
            "id": r["id"],
            "the_35B_ok": bool(r.get("scion_ok")),
            "p_correct": v["p_correct"],
            "accept": v["accept"],
            "verdict_s": v.get("ms", 0) / 1000.0,
            "wall_s": v["wall_s"],
            "the_35B_s": (r.get("scion_ms") or 0) / 1000.0,
        })
        print("  %-14s the 35B %-3s  p=%.4f  %5.2fs  (the 35B took %5.1fs)"
              % (r["id"][:14], "ok" if r.get("scion_ok") else "WRONG",
                 v["p_correct"], rows[-1]["verdict_s"], rows[-1]["the_35B_s"]),
              flush=True)

    # the separation
    pos = [x["p_correct"] for x in rows if x["the_35B_ok"]]
    neg = [x["p_correct"] for x in rows if not x["the_35B_ok"]]
    wins = sum(1 for a in pos for b in neg if a > b) + \
        0.5 * sum(1 for a in pos for b in neg if a == b)
    auc = wins / (len(pos) * len(neg)) if pos and neg else float("nan")

    # the cost of a verdict-only escalation against a generation escalation
    vtimes = [x["verdict_s"] for x in rows]
    gtimes = [x["the_35B_s"] for x in rows]
    out = {
        "n": len(rows),
        "the_35B_correct": len(pos),
        "the_35B_wrong": len(neg),
        "the_head_auc_on_the_35B's_answers": round(auc, 4),
        "the_mean_p_on_the_correct": round(statistics.mean(pos), 4) if pos else None,
        "the_mean_p_on_the_wrong": round(statistics.mean(neg), 4) if neg else None,
        "the_mean_verdict_s": round(statistics.mean(vtimes), 2),
        "the_median_verdict_s": round(statistics.median(vtimes), 2),
        "the_mean_generation_s": round(statistics.mean(gtimes), 2),
        "the_verdict_speedup": round(statistics.mean(gtimes) / statistics.mean(vtimes), 1),
        "the_errors_caught_at_0.5": sum(1 for x in rows
                                        if not x["the_35B_ok"] and x["p_correct"] < 0.5),
        "the_correct_kept_at_0.5": sum(1 for x in rows
                                       if x["the_35B_ok"] and x["p_correct"] >= 0.5),
    }
    print("\n" + json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
