#!/usr/bin/env python
"""Community-benchmark arms with a decision head (GSM8K/MATH/HumanEval).

Protocol (per model):
  gen    N samples/task at the model's serving sampling (checkpointed JSONL);
  hard   one "think-harder" sample/task (thinking enabled / larger budget);
  embed  judge-prompt last-token states via /embedding (answer + trace);
  eval   in-domain head on train tasks, then on the held-out test tasks:

    A  single (sample 0)
    B  think-harder single
    C  head-gated sequential (accept first sample with p >= thr, tuned on train)
    D  best-of-N by head
    majority / oracle references

  plus the natural-error catch on the model's own answers and the frozen
  J-Bench head applied to them.

Usage:
    community_eval.py --server http://127.0.0.1:8897 --prefix scion-community \
        --tasks experiments/cascade/tasks-community.json --phase gen
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import threading
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
DEFAULT_TASKS = CASCADE / "tasks-community.json"
SYSTEM = "You are a decision function. Reply with the answer label only."
ANSWER_SYSTEM = ("You are a precise assistant. Answer the user's task. For code "
                 "questions return only the requested Python function in a "
                 "single code block.")
PROMPT = ("system: You are a decision function. Reply with the answer "
          "label only.\nuser: Question: {task}\nCandidate answer: "
          "{candidate}\nIs the candidate answer correct?\nA. Yes\n"
          "B. No\nassistant (generation prompt, thinking disabled)")


def post(url: str, path: str, body: dict, timeout: float = 3600.0, tries: int = 3) -> dict:
    # The pod's multi-slot server can return 503 under load and a long prefill can reset a connection; a
    # sample is only lost when all retries fail.
    last: Exception | None = None
    for attempt in range(tries):
        req = urllib.request.Request(url.rstrip("/") + path,
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:   # urllib.error.HTTPError / URLError / timeouts / bad JSON
            last = e
            if attempt + 1 < tries:
                time.sleep(2.0 * (attempt + 1))
    raise last  # type: ignore[misc]


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


def gen_one(server: str, task: dict, *, user: str, sample: int, args,
            thinking: bool, max_tokens: int) -> dict:
    kwargs: dict = {} if thinking else {"enable_thinking": False}
    if args.reasoning_effort:
        kwargs["reasoning_effort"] = args.reasoning_effort
    body = {"model": args.model_name,
            "messages": [{"role": "system", "content": ANSWER_SYSTEM},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": args.temperature,
            "chat_template_kwargs": kwargs}
    if args.top_p is not None:
        body["top_p"] = args.top_p
    if args.top_k is not None:
        body["top_k"] = args.top_k
    if args.min_p is not None:
        body["min_p"] = args.min_p
    if args.presence_penalty is not None:
        body["presence_penalty"] = args.presence_penalty
    if args.repetition_penalty is not None:
        body["repetition_penalty"] = args.repetition_penalty
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
    lock = threading.Lock()
    t0 = time.time()
    jobs: list[tuple[dict, int]] = []
    for task in tasks:
        for k in range(len(done.get(task["id"], [])), args.samples):
            jobs.append((task, k))
    total = len(jobs)
    state = {"done": 0, "ok": 0}

    def run(job: tuple[dict, int]) -> None:
        task, k = job
        rec = gen_one(args.server, task, user=task["prompt"], sample=k,
                      args=args, thinking=args.thinking,
                      max_tokens=args.max_tokens)
        with lock:
            sf.write(json.dumps(rec) + "\n")
            sf.flush()
            state["done"] += 1
            state["ok"] += int(bool(rec["ok"]))
            if state["done"] % 25 == 0 or state["done"] == total:
                print(f"[{state['done']}/{total}] samples ok={state['ok']}/{state['done']} "
                      f"{time.time()-t0:.0f}s", flush=True)

    if args.concurrency <= 1:
        for job in jobs:
            run(job)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            list(ex.map(run, jobs))
    sf.close()
    print(f"wrote {path} ({total} new samples)")


def phase_hard(args, tasks: list[dict], prefix: str) -> None:
    path = args.out_dir / f"{prefix}-hard.jsonl"
    done = {rec["task_id"] for rec in load_jsonl(path)}
    sf = open(path, "a")
    lock = threading.Lock()
    t0 = time.time()
    jobs = [task for task in tasks if task["id"] not in done]
    total = len(jobs)
    state = {"done": 0}

    def run(task: dict) -> None:
        rec = gen_one(args.server, task, user=task["prompt"], sample=0,
                      args=args, thinking=args.hard_thinking,
                      max_tokens=args.hard_max_tokens)
        with lock:
            sf.write(json.dumps(rec) + "\n")
            sf.flush()
            state["done"] += 1
            if state["done"] % 25 == 0 or state["done"] == total:
                print(f"[{state['done']}/{total}] hard {time.time()-t0:.0f}s", flush=True)

    if args.concurrency <= 1:
        for task in jobs:
            run(task)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            list(ex.map(run, jobs))
    sf.close()
    print(f"wrote {path} ({total} new records)")


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
                       ("hard", f"{prefix}-hard.jsonl")):
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
    lock = threading.Lock()
    state = {"done": 0}

    def emb_one(item: dict):
        task = tasks_by[item["task_id"]]
        a = embed_prompt(args.server, task["prompt"], item["answer"])
        v = embed_prompt(args.server, task["prompt"], item["trace"])
        with lock:
            state["done"] += 1
            if state["done"] % 100 == 0 or state["done"] == len(items):
                print(f"[{state['done']}/{len(items)}] {time.time()-t0:.0f}s", flush=True)
        return a, v

    if args.concurrency <= 1:
        pairs = [emb_one(item) for item in items]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            pairs = list(ex.map(emb_one, items))
    for a, v in pairs:
        ha.append(a)
        hv.append(v)
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
    print(f"wrote {out} ({len(items)} x {ha[0].shape[0]})")


# ------------------------------------------------------------------ eval
def score_head(X, model, plat):
    return sigmoid(plat[0] * ridge_score(X, model) + plat[1])


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
    hard = {tid[i]: i for i, k in enumerate(kind) if k == "hard"}
    tr = [i for i in sm if split[i] == "train"]
    te = [i for i in sm if split[i] == "test"]

    fits = {}
    for name, X in (("answer", XA), ("trace", XV)):
        model = ridge_fit(X[tr], label[tr])
        plat = platt(ridge_score(X[tr], model), label[tr])
        p = score_head(X, model, plat)
        fits[name] = {"model": model, "plat": plat, "p": p,
                      "train": head_metrics(p[tr], label[tr]),
                      "test": head_metrics(p[te], label[te])}
        t = fits[name]["test"]
        auc_txt = f"{t['auc']:.3f}" if t["auc"] is not None else "n/a"
        print(f"in-domain head ({name}): test AUC {auc_txt} "
              f"BA {t['best_ba']['bal_acc']:.3f}", flush=True)
    primary = fits[args.primary]
    p_all = primary["p"]

    pools: dict[str, list[int]] = defaultdict(list)
    for i in sm:
        pools[tid[i]].append(i)
    for v in pools.values():
        v.sort(key=lambda i: sample[i])
    pools = {t: v for t, v in pools.items() if len(v) >= args.samples}

    text: dict[str, str] = {}
    for rec in load_jsonl(args.out_dir / f"{prefix}-samples.jsonl"):
        text[f"{rec['task_id']}|sample|{rec['sample']}"] = rec["content"]

    tr_tasks = [t for t, rows in pools.items() if split[rows[0]] == "train"]
    te_tasks = [t for t, rows in pools.items() if split[rows[0]] == "test"]

    def gated(rows, thr):
        used = 0
        for i in rows:
            used += 1
            if p_all[i] >= thr:
                return i, used
        return max(rows, key=lambda i: p_all[i]), used

    best_thr, best_key = 0.5, None
    for step in range(30, 100, 5):
        thr = step / 100
        correct = sum(int(label[gated(pools[t], thr)[0]]) for t in tr_tasks)
        used = sum(gated(pools[t], thr)[1] for t in tr_tasks)
        key = (correct, -used)
        if best_key is None or key > best_key:
            best_key, best_thr = key, thr

    def arm_stats(tids, pick_fn) -> dict:
        correct = n = 0
        tok = 0.0
        for t in tids:
            i, used, tk = pick_fn(t)
            if i is None:
                continue
            n += 1
            correct += int(label[i])
            tok += tk
        return {"n": n, "correct": correct, "acc": correct / max(1, n),
                "tokens": tok, "tok_per_correct": tok / max(1, correct)}

    def pick_A(t):
        i = pools[t][0]
        return i, 1, float(toks[i])

    def pick_D(t):
        i = max(pools[t], key=lambda j: p_all[j])
        return i, len(pools[t]), sum(float(toks[j]) for j in pools[t])

    def pick_C(t):
        i, used = gated(pools[t], best_thr)
        return i, used, sum(float(toks[j]) for j in pools[t][:used])

    def pick_B(t):
        i = hard.get(t)
        if i is None:
            return None, 0, 0.0
        return i, 1, float(toks[i])

    def pick_maj(t):
        canon = [canonical(tasks_by[t], text[keys[j]]) for j in pools[t]]
        reps: dict[str, list[int]] = {}
        for k, c in enumerate(canon):
            if c is not None:
                reps.setdefault(c, []).append(k)
        if reps:
            maj = max(reps, key=lambda c: len(reps[c]))
            if len(reps[maj]) > 1:
                j = pools[t][reps[maj][0]]
                return j, 1, float(toks[j])
        return pick_A(t)

    def oracle_count(tids):
        return sum(1 for t in tids if any(label[j] for j in pools[t]))

    arms = {
        "A_single": arm_stats(te_tasks, pick_A),
        "A_single_all": arm_stats(list(pools), pick_A),
        "B_think_hard": arm_stats(te_tasks, pick_B),
        "B_think_hard_all": arm_stats(list(pools), pick_B),
        "C_gated": {**arm_stats(te_tasks, pick_C),
                    "threshold": best_thr},
        "C_gated_all": {**arm_stats(list(pools), pick_C),
                        "threshold": best_thr},
        "D_best_of_n": arm_stats(te_tasks, pick_D),
        "D_best_of_n_all": arm_stats(list(pools), pick_D),
        "majority": arm_stats(te_tasks, pick_maj),
        "majority_all": arm_stats(list(pools), pick_maj),
    }
    oracle = {"test": {"correct": oracle_count(te_tasks), "n": len(te_tasks)},
              "all": {"correct": oracle_count(list(pools)), "n": len(pools)}}
    oracle["test"]["acc"] = oracle["test"]["correct"] / max(1, oracle["test"]["n"])
    oracle["all"]["acc"] = oracle["all"]["correct"] / max(1, oracle["all"]["n"])

    # per-sample metrics + natural catch (in-domain and J-Bench heads)
    first = [i for i in sm if sample[i] == 0]
    catch = {"indomain": {
        "first": {f"thr_{t}": catch_stats(primary["p"][first], label[first], t)
                  for t in (0.5,)},
        "all": {"metrics": head_metrics(primary["p"][sm], label[sm]),
                "at_0.5": threshold_stats(list(primary["p"][sm]),
                                          list(label[sm]), 0.5)}}}
    jb_states = args.out_dir / f"probe-states-{args.jbench_prefix}.npz"
    if jb_states.exists():
        zj = np.load(jb_states, allow_pickle=True)
        Xj = zj["L1"].astype(np.float64)
        yj = np.asarray(zj["labels"], dtype=np.float64)
        sj = [str(s) for s in zj["splits"]]
        jtr = [i for i, s in enumerate(sj) if s == "train"]
        jte = [i for i, s in enumerate(sj) if s == "test"]
        jmodel = ridge_fit(Xj[jtr], yj[jtr])
        jplat = platt(ridge_score(Xj[jtr], jmodel), yj[jtr])
        jp = score_head(XA, jmodel, jplat)
        catch["jbench"] = {
            "test": head_metrics(score_head(Xj[jte], jmodel, jplat), yj[jte]),
            "first": {f"thr_{t}": catch_stats(jp[first], label[first], t)
                      for t in (0.5,)},
            "all": {"metrics": head_metrics(jp[sm], label[sm])}}
        # J-Bench head as the chooser too
        arms["D_best_of_n_jbench"] = arm_stats(
            te_tasks, lambda t: (max(pools[t], key=lambda j: jp[j]),
                                 len(pools[t]),
                                 sum(float(toks[j]) for j in pools[t])))

    # export the in-domain head
    tr_label = label[tr]
    head = {
        "name": f"{prefix}-{args.primary}-decision-head",
        "model": args.model_label,
        "runtime": args.runtime,
        "readout": {"kind": "ridge-to-labels", "position": "last prompt token",
                    "layer": "final", "dim": int(XA.shape[1]),
                    "candidate": args.primary,
                    "calibration": "Platt on train scores"},
        "prompt": PROMPT,
        "training": {"bench": f"{args.tasks.name} (train tasks)",
                     "n_train": len(tr),
                     "n_pos": int(tr_label.sum()),
                     "n_neg": int(len(tr) - tr_label.sum())},
        "weights": {"w": [round(float(v), 8) for v in primary["model"][0]],
                    "mu": [round(float(v), 6) for v in primary["model"][1][0]],
                    "sd": [round(float(v), 6) for v in primary["model"][1][1]],
                    "b": float(primary["model"][1][2]),
                    "platt": [float(primary["plat"][0]),
                              float(primary["plat"][1])]},
        "test": {"n": len(te), "auc": primary["test"]["auc"],
                 "best_ba": primary["test"]["best_ba"],
                 "at_0.5": primary["test"]["at_0.5"]},
    }
    head_path = args.out_dir / f"{prefix}-indomain-head.json"
    head_path.write_text(json.dumps(head))
    print(f"wrote {head_path} ({head_path.stat().st_size/1024:.0f} KB)")

    result = {
        "prefix": prefix, "model": args.model_label,
        "sampling": {"temperature": args.temperature, "max_tokens": args.max_tokens,
                     "thinking": args.thinking},
        "hard": {"thinking": args.hard_thinking,
                 "max_tokens": args.hard_max_tokens},
        "primary_candidate": args.primary,
        "n_tasks": len(pools), "n_samples": len(sm),
        "indomain_heads": {name: {"train": f["train"], "test": f["test"]}
                           for name, f in fits.items()},
        "arms": arms, "oracle": oracle, "catch": catch,
    }
    out_path = args.out_dir / f"{prefix}-eval.json"
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in arms.items()}, indent=1))
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default="http://127.0.0.1:8897")
    ap.add_argument("--phase", default="all",
                    choices=("gen", "hard", "embed", "eval", "all"))
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--model-label", default="")
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--runtime", default="")
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    ap.add_argument("--out-dir", type=Path, default=R)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel HTTP requests; the server must run with -np at least this large")
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction,
                    default=False)
    ap.add_argument("--reasoning-effort", default="")
    ap.add_argument("--hard-thinking", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--hard-max-tokens", type=int, default=2048)
    ap.add_argument("--tail-chars", type=int, default=1000)
    ap.add_argument("--primary", default="answer",
                    choices=("answer", "trace"))
    ap.add_argument("--jbench-prefix", default="scion35b")
    args = ap.parse_args()

    tasks = json.loads(args.tasks.read_text())["tasks"]
    if args.limit:
        tasks = tasks[:args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.model_label:
        args.model_label = args.prefix

    if args.phase in ("gen", "all"):
        phase_gen(args, tasks, args.prefix)
    if args.phase in ("hard", "all"):
        phase_hard(args, tasks, args.prefix)
    if args.phase in ("embed", "all"):
        phase_embed(args, tasks, args.prefix)
    if args.phase in ("eval", "all"):
        phase_eval(args, tasks, args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
