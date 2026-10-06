#!/usr/bin/env python
"""Hinted retry: show the model its flagged answer, ask for a correction.

For each task, the first sample of the correction run is treated as the
previous attempt.  The model is asked to re-derive the task from scratch with
that answer in front of it ("it may be incorrect").  The control is blind
resampling: the same model, same temperature, same task, without the hint
(the remaining correction samples).

Evaluates:
  conditional success  among tasks where the previous answer was wrong:
                       blind resample vs hinted retry
  arms (test split)    single, blind retry@thr, hinted retry@thr, hinted best
  thresholds           fit on the 70 train tasks, applied to the 30 test tasks

Usage:
    hinted_retry.py --server http://127.0.0.1:8897
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from checkers import check  # noqa: E402
from readout_judge import order_body  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"
SYSTEM = "You are a decision function. Reply with the answer label only."
ANSWER_SYSTEM = ("You are a precise assistant. Answer the user's task. For code "
                 "questions return only the requested Python function in a "
                 "single code block.")
HINT = ("{prompt}\n\nA previous attempt at this task produced the answer below, "
        "which may be incorrect:\n\n---\n{previous}\n---\n\n"
        "Re-derive the task from scratch, check every step, and give a corrected "
        "answer. If the previous answer was actually correct, reproduce it. "
        "End with your final answer.")


def post(url: str, path: str, body: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    global args
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--head", type=Path, default=R / "scion-35b-a3b-decision-head.json")
    ap.add_argument("--samples", type=Path, default=R / "correction-samples.jsonl")
    ap.add_argument("--out", type=Path, default=R / "hinted-samples.jsonl")
    ap.add_argument("--eval-out", type=Path, default=R / "hinted-eval.json")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    head = json.loads(args.head.read_text())
    hd = (np.asarray(head["weights"]["w"]), np.asarray(head["weights"]["mu"]),
          np.asarray(head["weights"]["sd"]), head["weights"]["b"],
          tuple(head["weights"]["platt"]))

    def head_score(task_prompt: str, candidate: str) -> float:
        W, MU, SD, B, (PA, PC) = hd
        prompt = post(args.server, "/apply-template", {
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": order_body(
                             {"prompt": task_prompt, "candidate": candidate}, False)}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        emb = post(args.server, "/embedding",
                   {"content": prompt, "embd_normalize": -1})
        h = np.asarray(emb[0]["embedding"][0], dtype=np.float64)
        s = float(((h - MU) / SD) @ W + B)
        return float(1.0 / (1.0 + np.exp(-(PA * s + PC))))

    tasks = {t["id"]: t for t in json.loads(
        (CASCADE / "tasks-bench.json").read_text())["tasks"]}
    pool: dict[str, list] = defaultdict(list)
    for line in args.samples.read_text().splitlines():
        rec = json.loads(line)
        pool[rec["task_id"]].append(rec)
    for rows in pool.values():
        rows.sort(key=lambda r: r["sample"])
    task_ids = [t for t in tasks if t in pool and len(pool[t]) >= 4]
    if args.limit:
        task_ids = task_ids[:args.limit]

    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            done.add(json.loads(line)["task_id"])
    sf = open(args.out, "a")
    t0 = time.time()
    for n, tid in enumerate(task_ids, 1):
        if tid in done:
            continue
        task = tasks[tid]
        prev = pool[tid][0]
        k_ok = 0
        for k in range(args.retries):
            body = {"model": "scion",
                    "messages": [{"role": "system", "content": ANSWER_SYSTEM},
                                 {"role": "user", "content": HINT.format(
                                     prompt=task["prompt"],
                                     previous=prev["content"][:4000])}],
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature}
            started = time.time()
            data = post(args.server, "/v1/chat/completions", body, timeout=1800)
            msg = (data.get("choices") or [{}])[0].get("message", {})
            content = msg.get("content", "") or ""
            rec = {"task_id": tid, "bucket": task["bucket"], "split": task["split"],
                   "retry": k, "content": content,
                   "reasoning": msg.get("reasoning_content", "") or "",
                   "tokens": int((data.get("usage") or {}).get("completion_tokens") or 0),
                   "ms": round((time.time() - started) * 1000, 1),
                   "p_correct": 0.0, "ok": bool(check(task, content))}
            rec["p_correct"] = head_score(task["prompt"], content)
            k_ok += int(rec["ok"])
            sf.write(json.dumps(rec) + "\n")
            sf.flush()
        print(f"[{n}/{len(task_ids)}] {tid} ok {k_ok}/{args.retries} "
              f"{time.time()-t0:.0f}s", flush=True)
    sf.close()

    hinted: dict[str, list] = defaultdict(list)
    for line in args.out.read_text().splitlines():
        rec = json.loads(line)
        hinted[rec["task_id"]].append(rec)
    for rows in hinted.values():
        rows.sort(key=lambda r: r["retry"])

    train = [t for t in task_ids if tasks[t]["split"] == "train"]
    test = [t for t in task_ids if tasks[t]["split"] == "test"]

    def pick_best(rows):
        return max(rows, key=lambda r: r["p_correct"])

    def arm_seq(rows, thr):
        best = pick_best(rows)
        used = 0
        for r in rows:
            used += 1
            if r["p_correct"] >= thr:
                return r, used
        return best, used

    blind_rows = {t: pool[t] for t in task_ids}
    hint_rows = {t: pool[t][:1] + hinted[t] for t in task_ids if hinted.get(t)}

    def eval_arm(rows_by_task, sel, thr):
        correct = tok = n = 0
        for t in sel:
            rows = rows_by_task.get(t)
            if not rows:
                continue
            pick, used = arm_seq(rows, thr)
            n += 1
            correct += int(pick["ok"])
            tok += sum(r["tokens"] for r in rows[:used])
        return {"n": n, "correct": correct, "tokens": tok}

    best_thr, best_key = 0.5, None
    for step in range(30, 96, 5):
        thr = step / 100
        m = eval_arm(blind_rows, train, thr)
        key = (m["correct"], -m["tokens"])
        if best_key is None or key > best_key:
            best_key, best_thr = key, thr

    # conditional: previous wrong -> does a retry fix it?
    cond = {"n_wrong_prev": 0, "blind_any_fix": 0, "hint_any_fix": 0,
            "blind_first_fix": 0, "hint_first_fix": 0}
    for t in test:
        if pool[t][0]["ok"]:
            continue
        cond["n_wrong_prev"] += 1
        cond["blind_any_fix"] += int(any(r["ok"] for r in pool[t][1:]))
        cond["hint_any_fix"] += int(any(r["ok"] for r in hinted.get(t, [])))
        cond["blind_first_fix"] += int(bool(pool[t][1]["ok"])) if len(pool[t]) > 1 else 0
        cond["hint_first_fix"] += int(bool(hinted[t][0]["ok"])) if hinted.get(t) else 0

    out = {"n_tasks": len(task_ids), "best_thr": best_thr, "conditional": cond,
           "test": {
               "single": eval_arm({t: pool[t][:1] for t in task_ids}, test, 1.1),
               "blind_retry": eval_arm(blind_rows, test, best_thr),
               "hinted_retry": eval_arm(hint_rows, test, best_thr),
               "hinted_best": {**eval_arm({t: [pick_best(hint_rows[t])] for t in task_ids},
                                          test, 1.1)},
           }}
    # hinted_best tokens = all inspected
    hb = 0
    for t in test:
        hb += sum(r["tokens"] for r in hint_rows[t])
    out["test"]["hinted_best"] = {
        "n": len(test),
        "correct": sum(1 for t in test if pick_best(hint_rows[t])["ok"]),
        "tokens": hb}

    print(f"tasks {len(task_ids)}; chosen thr {best_thr}; "
          f"conditional on prev-wrong (test): {cond}")
    for name, m in out["test"].items():
        print(f"{name:14} {m['correct']}/{m['n']} tokens {m['tokens']}")
    args.eval_out.write_text(json.dumps(out, indent=1))
    print("wrote", args.eval_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
