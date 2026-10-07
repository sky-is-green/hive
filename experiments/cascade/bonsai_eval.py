#!/usr/bin/env python
"""Bonsai dense test: sampling loop + decision heads on Ternary-Bonsai-2-27B.

The Scion protocol, ported to the trained-low-bit dense counterpart:
generate 4 samples/task at the Bonsai card's recommended thinking sampling
(temp 1.0, top_p 0.95, top_k 20, min_p 0.05; reasoning traced), checker-label
the visible answers, embed the judge prompt's last-token state through
``/embedding`` (``--embeddings --pooling last --embd-normalize -1``) in two
candidate formats (answer-only and reasoning-tail + answer, as in the Scion
re-read test), fit the head on the train tasks in the serving format, then
evaluate on the held-out tasks:

  arms        B0 first sample, B1 majority, T1 best-of-4 by head, T2 gated
  frontier    best-of-1..4 by head, oracle, gated sequential (thr on train)
  retry       blind (later samples) vs hinted (shown the first answer)
  catch       in-domain and J-Bench heads on Bonsai's own answers

Phases (resumable, checkpointed JSONL):
  gen         4 samples/task
  hinted      2 hinted retries for tasks whose first sample is wrong
  embed       judge-prompt hidden states -> <prefix>-states.npz
  eval        head fit/export + all arms/frontier/retry/catch metrics

Usage:
    bonsai_eval.py --server http://127.0.0.1:8897 --phase gen
    bonsai_eval.py --server http://127.0.0.1:8897 --phase hinted
    bonsai_eval.py --server http://127.0.0.1:8897 --phase embed
    bonsai_eval.py --phase eval
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

from checkers import _boxed, _last_number, _normalize, check  # noqa: E402
from probe_judge import platt, ridge_fit, ridge_score  # noqa: E402
from readout_judge import auc, best_ba, order_body, threshold_stats  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"
DEFAULT_TASKS = CASCADE / "tasks-bench.json"
SYSTEM = "You are a decision function. Reply with the answer label only."
ANSWER_SYSTEM = ("You are a precise assistant. Answer the user's task. For code "
                 "questions return only the requested Python function in a "
                 "single code block.")
HINT = ("{prompt}\n\nA previous attempt at this task produced the answer below, "
        "which may be incorrect:\n\n---\n{previous}\n---\n\n"
        "Re-derive the task from scratch, check every step, and give a corrected "
        "answer. If the previous answer was actually correct, reproduce it. "
        "End with your final answer.")
MODEL = "prism-ml/Ternary-Bonsai-2-27B-gguf"
RUNTIME = ("moe-corr fork (llama.cpp 0.2.0-dev b10721), llama-server "
           "--embeddings --pooling last --embd-normalize -1")
PROMPT = ("system: You are a decision function. Reply with the answer "
          "label only.\nuser: Question: {task}\nCandidate answer: "
          "{candidate}\nIs the candidate answer correct?\nA. Yes\n"
          "B. No\nassistant (generation prompt, thinking disabled)")


def post(url: str, path: str, body: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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
    return None


def trace_candidate(reasoning: str, content: str, tail_chars: int) -> str:
    reasoning = (reasoning or "").strip()
    if reasoning:
        return reasoning[-tail_chars:] + "\n\n" + content
    return content


def gen_one(server: str, task: dict, *, system: str, user: str, sample: int,
            args) -> dict:
    kwargs: dict = {} if args.thinking else {"enable_thinking": False}
    if args.reasoning_effort:
        kwargs["reasoning_effort"] = args.reasoning_effort
    body = {"model": "bonsai",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "chat_template_kwargs": kwargs}
    started = time.time()
    data = post(server, "/v1/chat/completions", body)
    msg = (data.get("choices") or [{}])[0].get("message", {})
    content = msg.get("content", "") or ""
    usage = data.get("usage") or {}
    return {"task_id": task["id"], "bucket": task["bucket"],
            "split": task["split"], "sample": sample,
            "content": content, "reasoning": msg.get("reasoning_content", "") or "",
            "tokens": int(usage.get("completion_tokens") or 0),
            "ms": round((time.time() - started) * 1000, 1),
            "ok": bool(check(task, content))}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def phase_gen(args, tasks: list[dict], prefix: str) -> None:
    path = args.out_dir / f"{prefix}-samples.jsonl"
    done: dict[str, list] = defaultdict(list)
    for rec in load_jsonl(path):
        done[rec["task_id"]].append(rec)
    sf = open(path, "a")
    t0 = time.time()
    for n, task in enumerate(tasks, 1):
        rows = done.get(task["id"], [])
        if len(rows) >= args.samples:
            continue
        for k in range(len(rows), args.samples):
            rec = gen_one(args.server, task, system=ANSWER_SYSTEM,
                          user=task["prompt"], sample=k, args=args)
            sf.write(json.dumps(rec) + "\n")
            sf.flush()
            rows.append(rec)
        print(f"[{n}/{len(tasks)}] {task['id']} "
              f"ok={sum(r['ok'] for r in rows)}/{len(rows)} "
              f"{time.time()-t0:.0f}s", flush=True)
    sf.close()
    print(f"wrote {path}")


def phase_hinted(args, tasks: list[dict], prefix: str) -> None:
    samples_path = args.out_dir / f"{prefix}-samples.jsonl"
    pool: dict[str, list] = defaultdict(list)
    for rec in load_jsonl(samples_path):
        pool[rec["task_id"]].append(rec)
    for rows in pool.values():
        rows.sort(key=lambda r: r["sample"])
    todo = [t for t in tasks if pool.get(t["id"]) and not pool[t["id"]][0]["ok"]]
    path = args.out_dir / f"{prefix}-hinted.jsonl"
    done: dict[str, int] = defaultdict(int)
    for rec in load_jsonl(path):
        done[rec["task_id"]] += 1
    sf = open(path, "a")
    t0 = time.time()
    for n, task in enumerate(todo, 1):
        if done.get(task["id"], 0) >= args.hinted:
            continue
        prev = pool[task["id"]][0]["content"][:4000]
        for k in range(done.get(task["id"], 0), args.hinted):
            rec = gen_one(args.server, task, system=ANSWER_SYSTEM,
                          user=HINT.format(prompt=task["prompt"], previous=prev),
                          sample=k, args=args)
            sf.write(json.dumps(rec) + "\n")
            sf.flush()
        print(f"[{n}/{len(todo)}] {task['id']} {time.time()-t0:.0f}s", flush=True)
    sf.close()
    print(f"wrote {path} ({len(todo)} wrong-first tasks)")


def embed_prompt(server: str, task_prompt: str, candidate: str) -> np.ndarray:
    prompt = post(server, "/apply-template", {
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": order_body(
                         {"prompt": task_prompt, "candidate": candidate}, False)}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    emb = post(server, "/embedding", {"content": prompt, "embd_normalize": -1})
    return np.asarray(emb[0]["embedding"][0], dtype=np.float32)


def phase_embed(args, tasks: list[dict], prefix: str) -> None:
    tasks_by = {t["id"]: t for t in tasks}
    items: list[dict] = []
    for kind, name in (("sample", f"{prefix}-samples.jsonl"),
                       ("hint", f"{prefix}-hinted.jsonl")):
        for rec in load_jsonl(args.out_dir / name):
            items.append({"key": f"{rec['task_id']}|{kind}|{rec['sample']}",
                          "task_id": rec["task_id"], "kind": kind,
                          "sample": rec["sample"], "split": rec["split"],
                          "label": int(bool(rec["ok"])),
                          "tokens": rec["tokens"],
                          "answer": rec["content"],
                          "trace": trace_candidate(rec.get("reasoning", ""),
                                                   rec["content"],
                                                   args.tail_chars)})
    ha, hv = [], []
    t0 = time.time()
    for n, item in enumerate(items, 1):
        task = tasks_by[item["task_id"]]
        ha.append(embed_prompt(args.server, task["prompt"], item["answer"]))
        hv.append(embed_prompt(args.server, task["prompt"], item["trace"]))
        if n % 25 == 0 or n == len(items):
            print(f"[{n}/{len(items)}] {time.time()-t0:.0f}s", flush=True)
    out = args.out_dir / f"{prefix}-states.npz"
    np.savez_compressed(
        out, keys=np.array([i["key"] for i in items]),
        task_id=np.array([i["task_id"] for i in items]),
        kind=np.array([i["kind"] for i in items]),
        sample=np.array([i["sample"] for i in items]),
        split=np.array([i["split"] for i in items]),
        label=np.array([i["label"] for i in items]),
        tokens=np.array([i["tokens"] for i in items]),
        H=np.stack(ha).astype(np.float32),
        HV=np.stack(hv).astype(np.float32))
    print(f"wrote {out} ({len(items)} x {ha[0].shape[0]}, answer+trace)")


# ------------------------------------------------------------------ eval
def score_head(X, model, plat):
    s = ridge_score(X, model)
    a, c = plat
    return s, sigmoid(a * s + c)


def head_metrics(p, y) -> dict:
    ba, thr = best_ba(list(p), list(y))
    return {"n": len(y), "auc": auc(list(p), list(y)),
            "best_ba": {"bal_acc": ba, "threshold": thr},
            "at_0.5": threshold_stats(list(p), list(y), 0.5)}


def catch_stats(p, y, thr) -> dict:
    wrong = [i for i, l in enumerate(y) if l == 0]
    right = [i for i, l in enumerate(y) if l == 1]
    caught = sum(1 for i in wrong if p[i] < thr)
    low = sum(1 for i in right if p[i] < thr)
    return {"threshold": round(float(thr), 4),
            "errors": len(wrong), "caught": caught,
            "catch_rate": caught / max(1, len(wrong)),
            "correct_kept": len(right) - low, "correct": len(right),
            "keep_rate": (len(right) - low) / max(1, len(right))}


def phase_eval(args, tasks: list[dict], prefix: str) -> None:
    z = np.load(args.out_dir / f"{prefix}-states.npz", allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    tid = [str(t) for t in z["task_id"]]
    kind = [str(k) for k in z["kind"]]
    split = [str(s) for s in z["split"]]
    sample = [int(s) for s in z["sample"]]
    label = np.asarray(z["label"], dtype=np.float64)
    toks = np.asarray(z["tokens"], dtype=np.float64)
    XA = z["H"].astype(np.float64)
    XV = z["HV"].astype(np.float64)
    tasks_by = {t["id"]: t for t in tasks}

    sm = [i for i, k in enumerate(kind) if k == "sample"]
    tr = [i for i in sm if split[i] == "train"]
    te = [i for i in sm if split[i] == "test"]

    # fit the in-domain head on train samples, both candidate formats
    fits = {}
    for name, X in (("answer", XA), ("trace", XV)):
        model = ridge_fit(X[tr], label[tr])
        plat = platt(ridge_score(X[tr], model), label[tr])
        _, p = score_head(X, model, plat)
        tr_m = head_metrics(p[tr], label[tr])
        te_m = head_metrics(p[te], label[te])
        fits[name] = {"model": model, "plat": plat, "p": p,
                      "train": tr_m, "test": te_m}
        auc_txt = f"{te_m['auc']:.3f}" if te_m["auc"] is not None else "n/a"
        print(f"in-domain head ({name}): test AUC {auc_txt} "
              f"BA {te_m['best_ba']['bal_acc']:.3f}", flush=True)
    primary = fits["trace"] if args.primary == "trace" else fits["answer"]
    p_all = primary["p"]
    te_metrics = primary["test"]
    thr_ba = te_metrics["best_ba"]["threshold"]

    # pools: task -> sample item ids, ordered
    pools: dict[str, list[int]] = defaultdict(list)
    for i in sm:
        pools[tid[i]].append(i)
    for v in pools.values():
        v.sort(key=lambda i: sample[i])
    pools = {t: v for t, v in pools.items() if len(v) >= args.samples}

    text: dict[str, str] = {}
    for rec in load_jsonl(args.out_dir / f"{prefix}-samples.jsonl"):
        text[f"{rec['task_id']}|sample|{rec['sample']}"] = rec["content"]

    # ------------------------------------------------------------- arms
    arms = {name: {"correct": 0, "tokens": 0.0, "n": 0}
            for name in ("B0", "B1", "T1")}
    per_task = []
    for t, rows in pools.items():
        b0 = rows[0]
        canon = [canonical(tasks_by[t], text[keys[i]]) for i in rows]
        reps: dict[str, list[int]] = {}
        for j, c in enumerate(canon):
            if c is not None:
                reps.setdefault(c, []).append(j)
        b1 = b0
        if reps:
            maj = max(reps, key=lambda c: len(reps[c]))
            if len(reps[maj]) > 1:
                b1 = rows[reps[maj][0]]
        t1 = max(rows, key=lambda i: p_all[i])
        for name, i in (("B0", b0), ("B1", b1), ("T1", t1)):
            arms[name]["n"] += 1
            arms[name]["correct"] += int(label[i])
            arms[name]["tokens"] += float(toks[i])
        per_task.append({"task_id": t, "split": split[b0],
                         "B0": int(label[b0]), "B1": int(label[b1]),
                         "T1": int(label[t1]), "tokens": float(toks[b0])})
    for name, a in arms.items():
        a["acc"] = a["correct"] / max(1, a["n"])
        a["tok_per_correct"] = a["tokens"] / max(1, a["correct"])

    # gated sequential (T2): first sample with p >= thr, tuned on train
    def gated(rows, thr):
        used = 0
        for i in rows:
            used += 1
            if p_all[i] >= thr:
                return i, used
        return max(rows, key=lambda i: p_all[i]), used

    tr_tasks = [t for t, rows in pools.items() if split[rows[0]] == "train"]
    te_tasks = [t for t, rows in pools.items() if split[rows[0]] == "test"]
    best_thr, best_key = 0.5, None
    for step in range(30, 100, 5):
        thr = step / 100
        correct = sum(int(label[gated(pools[t], thr)[0]]) for t in tr_tasks)
        used = sum(gated(pools[t], thr)[1] for t in tr_tasks)
        key = (correct, -used)
        if best_key is None or key > best_key:
            best_key, best_thr = key, thr
    t2 = {}
    for thr in (0.5, best_thr):
        correct = used = 0
        tok = 0.0
        for t in te_tasks:
            pick, u = gated(pools[t], thr)
            correct += int(label[pick])
            used += u
            tok += sum(float(toks[i]) for i in pools[t][:u])
        t2[f"T2@{thr}"] = {"n": len(te_tasks), "correct": correct, "tokens": tok,
                           "mean_used": round(used / max(1, len(te_tasks)), 2),
                           "tok_per_correct": tok / max(1, correct)}

    # ------------------------------------------------------- frontier
    def best_of(k: int):
        correct = 0
        tok = 0.0
        for t in te_tasks:
            pick = max(pools[t][:k], key=lambda i: p_all[i])
            correct += int(label[pick])
            tok += sum(float(toks[i]) for i in pools[t][:k])
        return {"correct": correct, "n": len(te_tasks),
                "acc": correct / max(1, len(te_tasks)), "tokens": tok,
                "tok_per_correct": tok / max(1, correct)}

    oracle = {"correct": sum(1 for t in te_tasks
                             if any(label[i] for i in pools[t])),
              "n": len(te_tasks)}
    oracle["acc"] = oracle["correct"] / max(1, oracle["n"])
    frontier = {f"best_of_{k}": best_of(k) for k in (1, 2, 3, 4)}
    frontier["oracle_best_of_4"] = oracle
    gt_correct = used = 0
    gt_tok = 0.0
    for t in te_tasks:
        pick, u = gated(pools[t], best_thr)
        gt_correct += int(label[pick])
        used += u
        gt_tok += sum(float(toks[i]) for i in pools[t][:u])
    frontier["gated_sequential"] = {
        "policy": f"accept first sample with p >= {best_thr} (tuned on train), "
                  "else best by score",
        "test": {"correct": gt_correct, "n": len(te_tasks),
                 "acc": gt_correct / max(1, len(te_tasks)),
                 "mean_samples": round(used / max(1, len(te_tasks)), 2),
                 "tokens": gt_tok,
                 "tok_per_correct": gt_tok / max(1, gt_correct)}}

    # --------------------------------------------------- retry (blind/hinted)
    hint_pool: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(kind):
        if k == "hint":
            hint_pool[tid[i]].append(i)
    for v in hint_pool.values():
        v.sort(key=lambda i: sample[i])
    cond_test = {"n_wrong_prev": 0, "blind_any_fix": 0, "blind_first_fix": 0,
                 "hinted_any_fix": 0, "hinted_first_fix": 0}
    cond = dict(cond_test)
    for t, rows in pools.items():
        if label[rows[0]] == 1:
            continue
        tail = rows[1:]
        hrows = hint_pool.get(t, [])
        for c in (cond, cond_test if split[rows[0]] == "test" else None):
            if c is None:
                continue
            c["n_wrong_prev"] += 1
            c["blind_any_fix"] += int(any(label[i] for i in tail))
            c["blind_first_fix"] += int(bool(label[tail[0]]))
            c["hinted_any_fix"] += int(any(label[i] for i in hrows))
            c["hinted_first_fix"] += int(bool(hrows and label[hrows[0]]))

    # -------------------------------------------- natural-error catch
    def catch_block(model_, plat_, X) -> dict:
        _, p = score_head(X, model_, plat_)
        psm = p[sm]
        first = [i for i in sm if sample[i] == 0]
        _, bthr = best_ba(list(p[tr]), list(label[tr]))
        return {"train_threshold": bthr,
                "all_samples": {f"thr_{t}": catch_stats(psm, label[sm], t)
                                for t in (0.5, bthr)},
                "first_samples": {f"thr_{t}": catch_stats(p[first],
                                                          label[first], t)
                                  for t in (0.5, bthr)},
                "metrics_all": head_metrics(psm, label[sm])}

    catch = {"indomain_selection": catch_block(primary["model"],
                                               primary["plat"], XV
                                               if args.primary == "trace"
                                               else XA)}
    for name in ("answer", "trace"):
        if name != args.primary:
            catch[f"indomain_{name}"] = catch_block(fits[name]["model"],
                                                    fits[name]["plat"],
                                                    XA if name == "answer" else XV)

    jb_states = args.out_dir / "probe-states-bonsai27b.npz"
    if jb_states.exists():
        zj = np.load(jb_states, allow_pickle=True)
        Xj = zj["L1"].astype(np.float64)
        yj = np.asarray(zj["labels"], dtype=np.float64)
        sj = [str(s) for s in zj["splits"]]
        jtr = [i for i, s in enumerate(sj) if s == "train"]
        jte = [i for i, s in enumerate(sj) if s == "test"]
        jmodel = ridge_fit(Xj[jtr], yj[jtr])
        jplat = platt(ridge_score(Xj[jtr], jmodel), yj[jtr])
        jm = head_metrics(sigmoid(jplat[0] * ridge_score(Xj[jte], jmodel)
                                  + jplat[1]), yj[jte])
        catch["jbench_answer"] = catch_block(jmodel, jplat, XA)
        catch["jbench_test"] = jm

    # ------------------------------------------------------- export
    tr_label = label[tr]
    head = {
        "name": f"bonsai-2-27b-{prefix}-{args.primary}-decision-head",
        "model": MODEL,
        "runtime": RUNTIME,
        "readout": {"kind": "ridge-to-labels", "position": "last prompt token",
                    "layer": "final", "dim": int(primary["model"][0].shape[0]),
                    "candidate": args.primary,
                    "calibration": "Platt on train scores"},
        "prompt": PROMPT,
        "training": {"bench": "cascade-bench-v1 4-sample pools (train tasks)",
                     "n_train": len(tr),
                     "n_pos": int(tr_label.sum()),
                     "n_neg": int(len(tr) - tr_label.sum()),
                     "candidate_format": args.primary},
        "weights": {"w": [round(float(v), 8) for v in primary["model"][0]],
                    "mu": [round(float(v), 6) for v in primary["model"][1][0]],
                    "sd": [round(float(v), 6) for v in primary["model"][1][1]],
                    "b": float(primary["model"][1][2]),
                    "platt": [float(primary["plat"][0]),
                              float(primary["plat"][1])]},
        "test": {"n": len(te), "auc": te_metrics["auc"],
                 "best_ba": te_metrics["best_ba"],
                 "threshold": thr_ba,
                 "at_0.5": te_metrics["at_0.5"]},
        "apply": ("p_correct = sigmoid(a*(ridge((h-mu)/sd)+b)+c) on the last "
                  "prompt-token hidden state from /embedding"),
    }
    head_path = args.out_dir / f"{prefix}-indomain-head.json"
    head_path.write_text(json.dumps(head))
    print(f"wrote {head_path} ({head_path.stat().st_size/1024:.0f} KB)")

    result = {
        "prefix": prefix, "model": MODEL,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "top_k": args.top_k, "min_p": args.min_p,
                     "presence_penalty": args.presence_penalty,
                     "repetition_penalty": args.repetition_penalty,
                     "max_tokens": args.max_tokens,
                     "thinking": args.thinking,
                     "reasoning_effort": args.reasoning_effort or "template default"},
        "primary_candidate": args.primary,
        "n_tasks": len(pools), "n_samples": len(sm),
        "indomain_heads": {name: {"train": f["train"], "test": f["test"]}
                           for name, f in fits.items()},
        "primary": {"candidate": args.primary,
                    "train": primary["train"], "test": te_metrics,
                    "threshold": thr_ba},
        "arms": arms, "gated": t2, "frontier": frontier,
        "conditional": {"all": cond, "test": cond_test},
        "catch": catch,
        "per_task": per_task,
    }
    out_path = args.out_dir / f"{prefix}-eval.json"
    out_path.write_text(json.dumps(result, indent=1))
    print(f"wrote {out_path}")


def phase_items(args, tasks: list[dict], prefix: str) -> None:
    """First sample per task as a prompted self-judge item set (own items)."""
    tasks_by = {t["id"]: t for t in tasks}
    pool: dict[str, list] = defaultdict(list)
    for rec in load_jsonl(args.out_dir / f"{prefix}-samples.jsonl"):
        pool[rec["task_id"]].append(rec)
    items = []
    for tid, rows in pool.items():
        rows.sort(key=lambda r: r["sample"])
        r0 = rows[0]
        items.append({"item_id": f"{tid}:own", "task_id": tid,
                      "bucket": r0["bucket"], "split": r0["split"],
                      "label": int(bool(r0["ok"])), "origin": "bonsai-own",
                      "prompt": tasks_by[tid]["prompt"],
                      "candidate": r0["content"]})
    out = args.out_dir / f"{prefix}-own-items.jsonl"
    out.write_text("\n".join(json.dumps(i, sort_keys=True) for i in items) + "\n")
    print(f"wrote {out} ({len(items)} items)")
    ans = args.out_dir / f"{prefix}-answers.json"
    ans.write_text(json.dumps({"base": "bonsai", "records": [
        {"id": i["task_id"], "bucket": i["bucket"], "split": i["split"],
         "prompt": i["prompt"], "scion_answer": i["candidate"],
         "scion_ok": bool(i["label"])} for i in items]}, indent=1))
    print(f"wrote {ans}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default="http://127.0.0.1:8897")
    ap.add_argument("--phase", default="all",
                    choices=("gen", "hinted", "embed", "eval", "items", "all"))
    ap.add_argument("--prefix", default="bonsai")
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    ap.add_argument("--out-dir", type=Path, default=R)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--hinted", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--reasoning-effort", default="",
                    help="template reasoning_effort (xhigh/medium/low); empty "
                         "= template default")
    ap.add_argument("--tail-chars", type=int, default=1000)
    ap.add_argument("--primary", default="trace",
                    choices=("answer", "trace"))
    args = ap.parse_args()

    tasks = json.loads(args.tasks.read_text())["tasks"]
    if args.limit:
        tasks = tasks[:args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ("gen", "all"):
        phase_gen(args, tasks, args.prefix)
    if args.phase in ("hinted", "all"):
        phase_hinted(args, tasks, args.prefix)
    if args.phase in ("embed", "all"):
        phase_embed(args, tasks, args.prefix)
    if args.phase in ("eval", "all"):
        phase_eval(args, tasks, args.prefix)
    if args.phase in ("items", "all"):
        phase_items(args, tasks, args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
