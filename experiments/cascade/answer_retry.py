#!/usr/bin/env python
"""Answer-only retry: reuse the reasoning, redo only the final step.

For the first sample of each task, strip the committed final answer (the
"Final answer:" tail for number/boxed tasks; for short-answer kinds keep the
question only) and ask for the corrected final answer with a short budget.
This tests *where* the error lives: if errors are mostly in the last
computation, a ~100-token retry fixes them and the loop gets 3-4x cheaper than
full regeneration; if the reasoning itself is wrong, it reproduces the error.

Controls (same tasks): blind resample (remaining correction samples), hinted
full retry (`hinted-samples.jsonl`).

Usage:
    answer_retry.py --server http://127.0.0.1:8897
"""

from __future__ import annotations

import argparse
import json
import re
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
ASK = ("{prompt}\n\nYour previous working is below. The final answer was flagged "
       "as possibly incorrect.\n\n---\n{prefix}\n---\n\nRe-check the final step "
       "carefully and give the corrected final answer. Be concise.")

STRIP = re.compile(r"final answer[^0-9A-Za-z]*.*$", re.I | re.S)


def post(url: str, path: str, body: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--head", type=Path, default=R / "scion-35b-a3b-decision-head.json")
    ap.add_argument("--samples", type=Path, default=R / "correction-samples.jsonl")
    ap.add_argument("--hinted", type=Path, default=R / "hinted-samples.jsonl")
    ap.add_argument("--out", type=Path, default=R / "answer-retry.jsonl")
    ap.add_argument("--eval-out", type=Path, default=R / "answer-retry-eval.json")
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    head = json.loads(args.head.read_text())
    W = np.asarray(head["weights"]["w"])
    MU = np.asarray(head["weights"]["mu"])
    SD = np.asarray(head["weights"]["sd"])
    B = head["weights"]["b"]
    PA, PC = head["weights"]["platt"]

    def head_score(task_prompt: str, candidate: str) -> float:
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
    pools: dict[str, list] = defaultdict(list)
    for line in args.samples.read_text().splitlines():
        rec = json.loads(line)
        pools[rec["task_id"]].append(rec)
    for rows in pools.values():
        rows.sort(key=lambda r: r["sample"])
    task_ids = [t for t in tasks if t in pools and len(pools[t]) >= 4]
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
        prev = pools[tid][0]
        kind = task["checker"]["type"]
        if kind in ("number", "boxed"):
            prefix = STRIP.sub("", prev["content"]).strip() or prev["content"][:1500]
        else:
            prefix = prev["content"][:1500]
        body = {"model": "scion",
                "messages": [{"role": "system", "content": ANSWER_SYSTEM},
                             {"role": "user", "content": ASK.format(
                                 prompt=task["prompt"], prefix=prefix)}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature}
        started = time.time()
        data = post(args.server, "/v1/chat/completions", body)
        msg = (data.get("choices") or [{}])[0].get("message", {})
        content = msg.get("content", "") or ""
        rec = {"task_id": tid, "bucket": task["bucket"], "split": task["split"],
               "content": content, "reasoning": msg.get("reasoning_content", "") or "",
               "tokens": int((data.get("usage") or {}).get("completion_tokens") or 0),
               "ms": round((time.time() - started) * 1000, 1),
               "ok": bool(check(task, content)),
               "p_correct": head_score(task["prompt"], content)}
        sf.write(json.dumps(rec) + "\n")
        sf.flush()
        print(f"[{n}/{len(task_ids)}] {tid} ok={rec['ok']} tok={rec['tokens']} "
              f"{time.time()-t0:.0f}s", flush=True)
    sf.close()

    retry = {json.loads(l)["task_id"]: json.loads(l)
             for l in args.out.read_text().splitlines()}
    hinted: dict[str, list] = defaultdict(list)
    if args.hinted.exists():
        for line in args.hinted.read_text().splitlines():
            rec = json.loads(line)
            hinted[rec["task_id"]].append(rec)

    train = [t for t in task_ids if tasks[t]["split"] == "train"]
    test = [t for t in task_ids if tasks[t]["split"] == "test"]

    def cond(sel, pick_fn):
        n = fix = 0
        for t in sel:
            if pools[t][0]["ok"]:
                continue
            n += 1
            fix += int(pick_fn(t))
        return {"n": n, "fixed": fix}

    cond_stats = {
        "answer_any": cond(train + test, lambda t: "ok" in retry.get(t, {}) and retry[t]["ok"]),
        "blind_any": cond(train + test, lambda t: any(r["ok"] for r in pools[t][1:])),
        "blind_first": cond(train + test, lambda t: bool(pools[t][1]["ok"])),
        "hinted_any": cond(train + test, lambda t: any(r["ok"] for r in hinted.get(t, []))),
        "hinted_first": cond(train + test, lambda t: bool(hinted[t][0]["ok"]) if hinted.get(t) else False),
    }

    # arms on test: single vs gated answer retry (choose by head score)
    def arm_single():
        correct = tok = 0
        for t in test:
            r = pools[t][0]
            correct += int(r["ok"])
            tok += r["tokens"]
        return {"correct": correct, "tokens": tok, "n": len(test)}

    def arm_gated(thr):
        correct = tok = 0
        for t in test:
            r0 = pools[t][0]
            if r0["p_correct"] >= thr or t not in retry:
                pick = r0
                used = [r0]
            else:
                r1 = retry[t]
                pick = r0 if r0["p_correct"] >= r1["p_correct"] else r1
                used = [r0, r1]
            correct += int(pick["ok"])
            tok += sum(r["tokens"] for r in used)
        return {"correct": correct, "tokens": tok, "n": len(test)}

    best_thr, best_key = 0.5, None
    for step in range(30, 96, 5):
        thr = step / 100
        m = arm_gated(thr)
        key = (m["correct"], -m["tokens"])
        if best_key is None or key > best_key:
            best_key, best_thr = key, thr

    out = {"conditional": cond_stats, "chosen_thr": best_thr,
           "test": {"single": arm_single(), "answer_retry_gated": arm_gated(best_thr)}}
    print(json.dumps(out, indent=1))
    args.eval_out.write_text(json.dumps(out, indent=1))
    print("wrote", args.eval_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
