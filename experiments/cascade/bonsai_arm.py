"""Bonsai 2 27B as the pre-API escalation arm.

The campaign's escalation tier used the 176B, and its generate arm was
net zero: 2 of the 7 errors fixed, 2 of the 21 correct broken.  The 176B
is the wrong shape for that role anyway - it is the *teacher* of
judgements (its head is the judge), not a better generator than the 35B.
The pre-API escalation wants a model that is better *and* cheap enough to
keep warm: a 27B dense.

This runs the same arm on the same 28 organic tasks, against the same
35B answers, with the same checkers:

* the 35B alone (the campaign's report),
* Bonsai 2 27B alone (every task sent up),
* the oracle escalation (only the 35B's errors sent up),
* and the realized policy once a judge gates it (the judge's numbers are
  the campaign's - 2 of 7 errors found, 0 of 21 correct sent up).

    python -m experiments.cascade.bonsai_arm --base http://127.0.0.1:8897
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARM = REPO / "experiments/cascade/results/cascade-arm-escalation-20261020/report.json"
TASKS = REPO / "experiments/cascade/tasks-organic-28.json"
GATE = REPO / "experiments/cascade/results/judge-gate-20261021/judge-gate.json"
OUT = REPO / "experiments/cascade/results/bonsai-escalation-20261021"


def post(base: str, path: str, payload: dict, timeout: float = 1200,
         retries: int = 3) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in (503, 429) or attempt == retries - 1:
                raise
            last = exc
            time.sleep(5 + 5 * attempt)
    raise last if last else RuntimeError("the request failed")


def answer(base: str, model: str, prompt: str, max_tokens: int,
           temperature: float) -> tuple[str, float]:
    t0 = time.time()
    response = post(base, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    text = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return text, time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8897")
    parser.add_argument("--model", default="bonsai")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(OUT / "bonsai-arm.json"))
    parser.add_argument("--reuse", default="",
                        help="a previous bonsai-arm.json to re-score (no "
                             "generation): the judge gate's numbers can change "
                             "without re-running the model")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    from experiments.cascade.checkers import check

    tasks = {t["id"]: t for t in json.loads(TASKS.read_text())["tasks"]}
    baseline = json.loads(ARM.read_text())["records"]
    if args.limit:
        baseline = baseline[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.reuse:
        rows = json.loads(Path(args.reuse).read_text())["rows"]
    for record in baseline:
        if args.reuse:
            break
        task = tasks.get(record["id"])
        if task is None:
            continue
        try:
            text, seconds = answer(args.base, args.model, task["prompt"],
                                   args.max_tokens, args.temperature)
            ok = bool(check(task, text))
            error = None
        except Exception as exc:  # noqa: BLE001 - a timeout is a measurement
            text, seconds, ok, error = "", 0.0, False, str(exc)[:160]
        row = {
            "id": record["id"],
            "the_35B_ok": bool(record.get("scion_ok")),
            "the_bonsai_ok": ok,
            "the_bonsai_answer": text,
            "the_bonsai_s": round(seconds, 2),
        }
        if error:
            row["the_error"] = error
        rows.append(row)
        print("  %-14s the 35B %-5s | Bonsai %-5s %6.1fs | %s"
              % (record["id"][:14], "ok" if row["the_35B_ok"] else "WRONG",
                 "ok" if ok else "WRONG", seconds, (text or "")[-60:].replace("\n", " ")),
              flush=True)
        (out_path.parent / "bonsai-arm.jsonl").open("a").write(
            json.dumps(row) + "\n")

    n = len(rows)
    small_right = sum(1 for r in rows if r["the_35B_ok"])
    big_right = sum(1 for r in rows if r["the_bonsai_ok"])
    fixed = sum(1 for r in rows if not r["the_35B_ok"] and r["the_bonsai_ok"])
    broken = sum(1 for r in rows if r["the_35B_ok"] and not r["the_bonsai_ok"])
    times = [r["the_bonsai_s"] for r in rows]
    # The realized policy: the 176B's head, at its best-net threshold, flags
    # a candidate; only the flagged ones go to Bonsai.  The threshold and
    # the per-item p come from the campaign's judge-gate run, so this is the
    # judge the system actually has - not an oracle.
    head = next(a for a in json.loads(GATE.read_text())["the_arms"]
                if "176B" in a["the_judge"])
    gate_threshold = head["the_best_net"]["the_threshold"]
    flagged = {i for i, p in head["the_p_correct"].items() if p < gate_threshold}
    gated = [r for r in rows if r["id"] in flagged]
    gated_right = sum(1 for r in gated if r["the_bonsai_ok"])
    correct_by_gate = (sum(1 for r in rows
                           if r["the_35B_ok"] and r["id"] not in flagged)
                       + gated_right)
    summary = {
        "n": n,
        "the_35B_accuracy": round(small_right / n, 3),
        "the_bonsai_accuracy": round(big_right / n, 3),
        "the_35B_wrong": n - small_right,
        "the_bonsai_wrong": n - big_right,
        "the_fixed": fixed,
        "the_broken": broken,
        "the_net": fixed - broken,
        "the_oracle_escalation_accuracy": round((small_right + fixed) / n, 3),
        "the_measured_judge_gate_accuracy": round(correct_by_gate / n, 3),
        "the_gate_threshold": gate_threshold,
        "the_gate_escalations": len(gated),
        "the_gate_errors_caught": sum(1 for r in gated if not r["the_35B_ok"]),
        "the_gate_correct_sent_up": sum(1 for r in gated if r["the_35B_ok"]),
        "the_measured_judge_escalations": len(gated),
        "the_mean_s": round(statistics.mean(times), 2) if times else None,
        "the_total_s": round(sum(times), 1),
    }
    payload = {"summary": summary, "rows": rows}
    out_path.write_text(json.dumps(payload, indent=1))
    print()
    print(json.dumps(summary, indent=1))
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
