#!/usr/bin/env python
"""L1/L2 correction study on the bench: selection, retry, and baselines.

Generates N samples per task (Scion server, /v1/chat/completions), scores each
with the exported decision head (judge prompt -> /embedding -> ridge+Platt),
and evaluates arms:

  B0      first sample (single attempt, temp 0.7)
  B1      self-consistency majority over N, where a canonical answer exists
  T1      best-of-N by head p_correct
  T2@t    sequential head-gated retry: accept the first sample with p>=t
          (else the best), paying only for the samples it inspects
  (B0-greedy reference is read from scion-answers.json, no extra generation)

Samples are checkpointed to JSONL after every task, and each sample carries
the full text plus token count so the same run doubles as the L3 monitor
dataset (per-position hidden dumps via test-mtp-probe).

Usage:
    correction_eval.py --server http://127.0.0.1:8897 \
        --head results/jev-readout-20261006/scion-35b-a3b-decision-head.json \
        --out results/jev-readout-20261006/correction-eval.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from checkers import _boxed, _last_number, _normalize, check  # noqa: E402
from readout_judge import order_body  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"
DEFAULT_TASKS = CASCADE / "tasks-bench.json"
SYSTEM = "You are a decision function. Reply with the answer label only."
ANSWER_SYSTEM = ("You are a precise assistant. Answer the user's task. For code "
                 "questions return only the requested Python function in a "
                 "single code block.")


def post(url: str, path: str, body: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def canonical(task: dict, text: str) -> str | None:
    kind = task["checker"]["type"]
    if kind == "number":
        marker = re.search(r"final answer[^0-9\-+]*([-+]?\d[\d,]*(?:\.\d+)?)",
                           text, re.I | re.S)
        if marker:
            return marker.group(1).replace(",", "")
        v = _last_number(text)
        return None if v is None else f"{v:g}"
    if kind == "boxed":
        return _boxed(text)
    if kind == "choice":
        letters = re.findall(r"\b([A-H])\b", text.upper())
        return letters[-1] if letters else None
    if kind == "contains":
        return _normalize(text)
    return None  # code and others: no canonical form


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--head", type=Path,
                    default=R / "scion-35b-a3b-decision-head.json")
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    ap.add_argument("--out", type=Path, default=R / "correction-eval.json")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="reuse correction-samples.jsonl if present")
    args = ap.parse_args()

    head = json.loads(args.head.read_text())
    W = np.asarray(head["weights"]["w"])
    MU = np.asarray(head["weights"]["mu"])
    SD = np.asarray(head["weights"]["sd"])
    B = head["weights"]["b"]
    PA, PC = head["weights"]["platt"]

    tasks = json.loads(args.tasks.read_text())["tasks"]
    if args.limit:
        tasks = tasks[:args.limit]
    samples_path = args.out.with_name("correction-samples.jsonl")

    done: dict[str, list] = {}
    if args.resume and samples_path.exists():
        for line in samples_path.read_text().splitlines():
            rec = json.loads(line)
            done.setdefault(rec["task_id"], []).append(rec)
        print(f"resume: {len(done)} tasks already captured", flush=True)

    sf = open(samples_path, "a")
    t0 = time.time()
    for n, task in enumerate(tasks, 1):
        if task["id"] in done and len(done[task["id"]]) >= args.samples:
            continue
        rows = done.get(task["id"], [])
        for k in range(len(rows), args.samples):
            body = {"model": "scion",
                    "messages": [{"role": "system", "content": ANSWER_SYSTEM},
                                 {"role": "user", "content": task["prompt"]}],
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature}
            started = time.time()
            data = post(args.server, "/v1/chat/completions", body)
            msg = (data.get("choices") or [{}])[0].get("message", {})
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            usage = data.get("usage") or {}
            # head score on the judge prompt
            prompt = post(args.server, "/apply-template", {
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user",
                              "content": order_body({"prompt": task["prompt"],
                                                     "candidate": content}, False)}],
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
            emb = post(args.server, "/embedding",
                       {"content": prompt, "embd_normalize": -1})
            h = np.asarray(emb[0]["embedding"][0], dtype=np.float64)
            score = float(((h - MU) / SD) @ W + B)
            p = float(1.0 / (1.0 + np.exp(-(PA * score + PC))))
            rec = {"task_id": task["id"], "bucket": task["bucket"],
                   "split": task["split"], "sample": k,
                   "content": content, "reasoning": reasoning,
                   "tokens": int(usage.get("completion_tokens") or 0),
                   "ms": round((time.time() - started) * 1000, 1),
                   "head_score": score, "p_correct": round(p, 6),
                   "ok": bool(check(task, content))}
            rows.append(rec)
            sf.write(json.dumps(rec) + "\n")
            sf.flush()
        done[task["id"]] = rows
        print(f"[{n}/{len(tasks)}] {task['id']} "
              f"ok={sum(r['ok'] for r in rows)}/{len(rows)} {time.time()-t0:.0f}s",
              flush=True)
    sf.close()

    # ------------------------------------------------------------------ arms
    arms = {name: {"correct": 0, "tokens": 0, "n": 0}
            for name in ("B0", "B1", "T1")}
    per_task = []
    for task in tasks:
        rows = done.get(task["id"])
        if not rows or len(rows) < args.samples:
            continue
        kinds = task["checker"]["type"]
        b0 = rows[0]
        # B1: majority canonical (code/others fall back to sample 0)
        canon = [canonical(task, r["content"]) for r in rows]
        reps = {}
        for i, c in enumerate(canon):
            if c is not None:
                reps.setdefault(c, []).append(i)
        if reps:
            maj = max(reps, key=lambda c: len(reps[c]))
            if len(reps[maj]) > 1:
                b1 = rows[reps[maj][0]]
            else:
                b1 = b0
        else:
            b1 = b0
        t1 = max(rows, key=lambda r: r["p_correct"])
        for name, r in (("B0", b0), ("B1", b1), ("T1", t1)):
            arms[name]["n"] += 1
            arms[name]["correct"] += int(r["ok"])
            arms[name]["tokens"] += int(r["tokens"])
        t2 = {}
        for thr in (0.5, 0.87):
            used, choice = 0, None
            best = max(rows, key=lambda r: r["p_correct"])
            for r in rows:
                used += 1
                if r["p_correct"] >= thr:
                    choice = r
                    break
            choice = choice or best
            t2[f"T2@{thr}"] = {"ok": bool(choice["ok"]),
                               "tokens": sum(int(x["tokens"]) for x in rows[:used]),
                               "used": used}
        per_task.append({"task_id": task["id"], "bucket": task["bucket"],
                         "split": task["split"],
                         "B0": {"ok": b0["ok"], "tokens": b0["tokens"]},
                         "B1": {"ok": b1["ok"], "tokens": b1["tokens"]},
                         "T1": {"ok": t1["ok"], "tokens": t1["tokens"]},
                         "T2": t2})

    t2_agg = {}
    for key in ("T2@0.5", "T2@0.87"):
        sel = [t for t in per_task if key in t["T2"]]
        t2_agg[key] = {
            "n": len(sel),
            "correct": sum(1 for t in sel if t["T2"][key]["ok"]),
            "tokens": sum(t["T2"][key]["tokens"] for t in sel),
            "mean_used": round(sum(t["T2"][key]["used"] for t in sel)
                               / max(1, len(sel)), 2)}
    result = {"samples_per_task": args.samples,
              "temperature": args.temperature,
              "arms": arms, "retry": t2_agg, "per_task": per_task}
    args.out.write_text(json.dumps(result, indent=1))

    print(f"\n=== correction arms (n={len(per_task)}) ===")
    print(f"{'arm':10} {'correct':>8} {'acc':>6} {'tokens':>9} {'tok/correct':>12}")
    for name, a in arms.items():
        acc = a["correct"] / max(1, a["n"])
        tpc = a["tokens"] / max(1, a["correct"])
        print(f"{name:10} {a['correct']:>6}/{a['n']:<3} {acc:>6.3f} "
              f"{a['tokens']:>9} {tpc:>12.1f}")
    for key, a in t2_agg.items():
        acc = a["correct"] / max(1, a["n"])
        tpc = a["tokens"] / max(1, a["correct"])
        print(f"{key:10} {a['correct']:>6}/{a['n']:<3} {acc:>6.3f} "
              f"{a['tokens']:>9} {tpc:>12.1f} (mean used {a['mean_used']})")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
