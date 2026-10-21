#!/usr/bin/env python3
"""The escalation tier, measured two ways: the 176B answers, or the 176B judges.

The campaign's escalation tier asks the 176B to **generate** an answer
when the small tier's answer is flagged.  The 176B also has a **decision
head** (AUC 0.994 on the frozen judge bench, `SkyIsNotGreen/decision-
heads`): the last prompt token's hidden state, read through the engine's
`emb=`, predicts whether a candidate answer is right.  A verdict is a
prefill - 2.4 s - where a generation is 27 s warm and 230 s co-sited.

This runs both arms over the same tasks with the same small-tier answers
and the same verifier, and reports the served-correct and the cost of
each:

  arm "generate": the 176B answers again, and its answer is served
  arm "verdict":  the 176B judges the small tier's answer; a low
                   p_correct escalates to a generation, a high one
                   keeps the answer as it is

Usage: escalation_arms.py [--tasks TASKS] [--limit N] [--out JSON]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARM = REPO / "experiments/cascade/results/cascade-arm-escalation-20261020/report.json"
TASKS = REPO / "experiments/cascade/tasks-organic-28.json"
BASE = "http://127.0.0.1:8081"


def post(path: str, payload: dict, timeout: float = 900, retries: int = 3) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            # 503: the serve's single slot is still busy with the previous
            # request (the engine answers one at a time).  Wait and retry.
            if exc.code != 503 or attempt == retries - 1:
                raise
            last = exc
            time.sleep(5 + 5 * attempt)
    raise last if last else RuntimeError("the request failed")


def verdict(task: str, candidate: str) -> dict:
    return post("/v1/verdict", {"task": task, "candidate": candidate})


def generate(task: str, max_tokens: int = 1024) -> dict:
    # a thinking budget: without one the model can spend every token thinking
    # and the reply has no answer (the serve answers 503 to the next request
    # while the engine is still busy with this one)
    return post("/v1/chat/completions",
                {"model": "scion-flashnext-176b-a6b-q6k",
                 "messages": [{"role": "user", "content": task}],
                 "max_tokens": max_tokens, "temperature": 0.7,
                 "reasoning_budget_tokens": 512})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--arm", choices=("judge", "generate", "both"), default="both",
                    help="the two arms are alternatives, so they are measured "
                         "separately: interleaving them makes each inherit the "
                         "other's expert cache (a generation's decode evicts the "
                         "judge's experts)")
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
        prompt = task["prompt"]
        cand = (r.get("scion_answer") or "").strip()
        if not cand:
            continue
        row = {"id": r["id"], "the_35B_ok": bool(r.get("scion_ok"))}

        # the judge arm: the 176B reads the small tier's answer
        if args.arm in ("judge", "both"):
            try:
                v = verdict(prompt, cand)
                row["p_correct"] = v["p_correct"]
                row["the_verdict_s"] = v.get("ms", 0) / 1000.0
                row["the_verdict_keeps"] = v["p_correct"] >= args.threshold
            except Exception as exc:  # noqa: BLE001 - a 503 is a measurement, not a crash
                print("  %-14s the verdict failed: %s" % (r["id"][:14], exc), flush=True)
                row["p_correct"] = None
                row["the_verdict_s"] = 0.0
                row["the_verdict_keeps"] = False

        # the generate arm: the 176B answers again
        if args.arm in ("generate", "both"):
            t0 = time.time()
            try:
                g = generate(prompt)
                row["the_generation_s"] = round(time.time() - t0, 2)
                row["the_176B_answer"] = (g.get("choices") or [{}])[0].get("message", {}).get("content", "")
            except Exception as exc:  # noqa: BLE001
                row["the_generation_s"] = round(time.time() - t0, 2)
                row["the_176B_answer"] = ""
                row["the_generation_error"] = str(exc)[:120]

        rows.append(row)
        p = row.get("p_correct")
        print("  %-14s the 35B %-5s p=%s %-7s | the generation %s"
              % (r["id"][:14], "ok" if row["the_35B_ok"] else "WRONG",
                 ("%.4f" % p) if p is not None else "  --  ",
                 "keeps" if row.get("the_verdict_keeps") else "escalates",
                 ("%6.1fs" % row["the_generation_s"]) if "the_generation_s" in row else "   -"),
              flush=True)

    print("\n  the judge arm needs the task's own checker to score the "
          "176B's answers; the summary below reports the arm's COST and "
          "the judge's decisions, and the accuracy of each arm is scored "
          "by the campaign's own checkers in the report.\n")

    judged = [x for x in rows if x.get("p_correct") is not None]
    vs = [x["the_verdict_s"] for x in judged]
    gs = [x["the_generation_s"] for x in rows if "the_generation_s" in x]
    # the judge arm's cost: a verdict for everyone, and a generation only
    # for the ones the judge sent up
    escalations = [x for x in judged if not x["the_verdict_keeps"]]
    judge_cost = sum(vs) + sum(x.get("the_generation_s", 0.0) for x in escalations)
    out = {
        "n": len(rows),
        "the_arm": args.arm,
        "the_judged": len(judged),
        "the_threshold": args.threshold,
        "the_mean_verdict_s": round(statistics.mean(vs), 2) if vs else None,
        "the_median_verdict_s": round(statistics.median(vs), 2) if vs else None,
        "the_mean_generation_s": round(statistics.mean(gs), 2) if gs else None,
        "the_median_generation_s": round(statistics.median(gs), 2) if gs else None,
        "the_judge_arm_escalations": len(escalations),
        "the_judge_arm_escalation_rate": round(len(escalations) / len(judged), 3) if judged else None,
        "the_generate_arm_cost_s": round(sum(gs), 1) if gs else None,
        "the_judge_arm_cost_s": round(judge_cost, 1) if judged else None,
        "the_judge_arm_saving": round(sum(gs) / judge_cost, 2) if gs and judge_cost else None,
        "the_errors_the_judge_sends_up": sum(1 for x in judged
                                             if not x["the_35B_ok"] and not x["the_verdict_keeps"]),
        "the_correct_the_judge_sends_up": sum(1 for x in judged
                                              if x["the_35B_ok"] and not x["the_verdict_keeps"]),
        "the_judge_arm_keeps_the_correct": sum(1 for x in judged
                                               if x["the_35B_ok"] and x["the_verdict_keeps"]),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
