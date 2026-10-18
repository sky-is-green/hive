#!/usr/bin/env python3
"""Dual generation: run one prompt through two independently-hosted models at the same time.

This is the mechanical test for the tiered-routing idea — 35B (Scion-35B-A3B, card 0) and
Bonsai-2 27B ternary (card 1) — answering the same prompts side by side, so we learn three things
that no offline probe has told us:

  1. does the system actually co-host two engines on two cards without one starving the other,
  2. what the per-model latency is when both are churning, and
  3. how often the two disagree, which is the raw material for the arbiter that comes next.

Agreement is computed on the *extracted final answer* (the checker-consumable part), not the raw
text, so two models that agree in substance but differ in prose count as agreement.

    .venv/bin/python experiments/cascade/dual_generation.py \
        --a http://127.0.0.1:8082 --a-name 35b \
        --b http://127.0.0.1:8083 --b-name bonsai27b \
        --bench <bench.jsonl> --limit 66 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from checkers import check  # noqa: E402
from consistency_check import check_answer  # noqa: E402

BOXED = re.compile(r"\\boxed\{(.+?)\}")
FINAL_NUM = re.compile(r"final answer[^0-9\-+]*([-+]?\d[\d,]*(?:\.\d+)?)", re.I | re.S)
LETTER = re.compile(r"\b([A-D])\b")


def post(url: str, path: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def generate(url: str, prompt_text: str, system: str | None,
             max_tokens: int, temperature: float, thinking: bool) -> tuple[str, str | None, int | None]:
    """OpenAI-shaped chat completion, tolerant of engines that lack thinking control.

    Returns (text, finish_reason, completion_tokens).  finish_reason matters here: a smaller model
    that runs to the token cap never emits its final answer, and scoring that as "wrong" would
    measure the harness's budget rather than the model.
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt_text}]
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": temperature}
    try:
        out = post(url, "/v1/chat/completions", {**body, "chat_template_kwargs":
                                                 {"enable_thinking": thinking}})
    except Exception:
        out = post(url, "/v1/chat/completions", body)
    ch = out["choices"][0]
    return (ch["message"]["content"], ch.get("finish_reason"),
            (out.get("usage") or {}).get("completion_tokens"))


def final_answer(text: str) -> str | None:
    """The checker-consumable answer: boxed, else 'Final answer: n', else a bare choice letter."""
    m = BOXED.findall(text or "")
    if m:
        return m[-1].strip()
    m = FINAL_NUM.findall(text or "")
    if m:
        return m[-1].strip().replace(",", "")
    m = LETTER.findall((text or "").upper())
    if m:
        return m[-1]
    return None


def agree(a: str | None, b: str | None) -> bool | None:
    if a is None or b is None:
        return None
    return a.strip().replace(" ", "") == b.strip().replace(" ", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-name", default="model-a")
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-name", default="model-b")
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=1,
                    help="items in flight; every item fires BOTH models at once (default 1)")
    ap.add_argument("--tasks", type=Path, nargs="+", default=None,
                    help="task spec files carrying checker specs")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in args.bench.read_text().splitlines() if l.strip()]
    if args.limit:
        items = items[:args.limit]

    tasks: dict[str, dict] = {}
    for spec_file in (args.tasks or []):
        if not spec_file.exists():
            print(f"warn: task specs not found: {spec_file}")
            continue
        d = json.loads(spec_file.read_text())
        for it in (d if isinstance(d, list) else (d.get("items") or d.get("tasks") or [])):
            if "checker" in it:
                tasks.setdefault(str(it.get("id") or it.get("task_id")), it)

    system = ("Solve the problem, showing your work. End your answer with 'Final answer: <number>'."
              if True else None)

    def call(tag: str, url: str, prompt_text: str) -> tuple[str, dict]:
        t0 = time.time()
        try:
            txt, finish, ntok = generate(url, prompt_text, system, args.max_tokens,
                                         args.temperature, False)
            return tag, {"text": txt, "ms": (time.time() - t0) * 1000,
                         "final": final_answer(txt), "ok": True,
                         "finish_reason": finish, "completion_tokens": ntok}
        except Exception as exc:  # noqa: BLE001
            return tag, {"text": "", "ms": (time.time() - t0) * 1000, "final": None,
                         "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    inner = ThreadPoolExecutor(max_workers=2)   # both models on each prompt at the same time

    def one(item: dict) -> dict:
        prompt_text = item["prompt"]
        futs = [inner.submit(call, tag, url, prompt_text)
                for tag, url in (("a", args.a), ("b", args.b))]
        res: dict = dict(f.result() for f in futs)
        row = {"item_id": item["item_id"], "task_id": item.get("task_id"),
               "label": item.get("label"), "origin": item.get("origin"),
               args.a_name: res["a"], args.b_name: res["b"]}
        row["agree"] = agree(res["a"]["final"], res["b"]["final"])
        spec = tasks.get(item.get("task_id"))
        if spec:
            for tag in ("a", "b"):
                try:
                    ok = check(spec, res[tag]["text"])
                except Exception:  # noqa: BLE001
                    ok = None
                res[tag]["checker_pass"] = ok
                res[tag]["consistency_flag"] = check_answer(res[tag]["text"])["flagged"]
        return row

    t0 = time.time()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for n, row in enumerate(pool.map(one, items), 1):
            rows.append(row)
            if n % 10 == 0 or n == len(items):
                print(f"[{n}/{len(items)}] {time.time() - t0:.0f}s "
                      f"agree={sum(1 for r in rows if r['agree'])}", flush=True)

    scored = [r for r in rows if "checker_pass" in r.get(args.a_name, {})]
    summary = {
        "items": len(rows),
        "models": [args.a_name, args.b_name],
        "wall_s": time.time() - t0,
        "agreement": {
            "both_answered": sum(1 for r in rows if r["agree"] is not None),
            "agreed": sum(1 for r in rows if r["agree"] is True),
            "disagreed": sum(1 for r in rows if r["agree"] is False),
        },
        "latency_ms": {name: {
            "median": sorted(r[name]["ms"] for r in rows)[len(rows) // 2],
            "mean": sum(r[name]["ms"] for r in rows) / len(rows),
            "p90": sorted(r[name]["ms"] for r in rows)[int(len(rows) * 0.9)],
            "errors": sum(1 for r in rows if not r[name]["ok"]),
        } for name in (args.a_name, args.b_name)},
        # A model that never says "Final answer" was cut off by --max-tokens, not wrong; these two
        # counters keep that separable from the accuracy numbers below when the budget is tight.
        "budget": {name: {
            "hit_cap": sum(1 for r in rows if r[name].get("finish_reason") == "length"),
            "no_final": sum(1 for r in rows if r[name].get("final") is None),
        } for name in (args.a_name, args.b_name)},
    }
    if scored:
        truth = [r["label"] for r in scored]
        for name in (args.a_name, args.b_name):
            wrong = sum(1 for r in scored if (r[name]["checker_pass"] or False) != (r["label"] == 1))
            summary.setdefault("checker_accuracy", {})[name] = {
                "n": len(scored), "wrong_vs_label": wrong,
                "wrong_rate": wrong / len(scored),
                "false_reject": sum(1 for r in scored if r["label"] == 1 and not r[name]["checker_pass"]),
                "false_accept": sum(1 for r in scored if r["label"] == 0 and r[name]["checker_pass"]),
                "consistency_flagged": sum(1 for r in scored if r[name]["consistency_flag"]),
            }
        both = [r for r in scored if r["agree"] is not None]
        summary["on_disagreements"] = {
            "n": len(both),
            "a_right": sum(1 for r in both if (r[args.a_name]["checker_pass"] or False) ==
                           (r["label"] == 1)),
            "b_right": sum(1 for r in both if (r[args.b_name]["checker_pass"] or False) ==
                           (r["label"] == 1)),
            "neither": sum(1 for r in both if (r[args.a_name]["checker_pass"] or False) !=
                           (r["label"] == 1) and (r[args.b_name]["checker_pass"] or False) !=
                           (r["label"] == 1)),
        }
        del truth

    (args.out / "dual-generation.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (args.out / "dual-generation-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())