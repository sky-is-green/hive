#!/usr/bin/env python
"""Harvest organic flashnext-176B answers for the decision gate (stage 2).

The distribution fix: the frozen J-Bench negatives are checker-verified
*perturbations* of correct answers, but the gate must police the answers the
resident 176B actually produces.  This script asks the live serve on port 8081
for every benchmark task, labels each answer with the task's own checker
(``checkers.check``), keeps organic wrong answers as the gold negatives, and
amplifies correct **train**-side answers with the exact perturbation recipes
the J-Bench used (``judge_bench.perturbations``, checker-verified wrong).
Test-side tasks stay organic-only, so the harvested test slice is a
fresh-distribution measurement and never synthetic.

Thinking is disabled via ``chat_template_kwargs`` so answers arrive as complete
``content`` (not truncated mid-``reasoning_content``); truncated or empty
generations are counted and dropped, never labeled.

Usage, from the repo root:

    .venv/bin/python experiments/cascade/harvest.py --limit 3 --samples 1   # smoke
    .venv/bin/python experiments/cascade/harvest.py                        # full run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cascade.checkers import check  # noqa: E402
from experiments.cascade.judge_bench import DECOYS, perturbations  # noqa: E402

TASKS_FILE = REPO_ROOT / "experiments" / "cascade" / "tasks-bench.json"
OUT_DIR = REPO_ROOT / "experiments" / "cascade" / "results" / "decision-harvest-20261017"
SERVE = "http://127.0.0.1:8081"
MODEL = "scion-flashnext-176b-a6b-q6k"


def _probe_health() -> dict:
    try:
        with urllib.request.urlopen(f"{SERVE}/health", timeout=5) as resp:
            return {"url": f"{SERVE}/health", "status": resp.status}
    except Exception as exc:  # noqa: BLE001 — informational
        return {"url": f"{SERVE}/health", "status": None, "error": str(exc)}


def _serve_model_info() -> dict:
    try:
        with urllib.request.urlopen(f"{SERVE}/v1/models", timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _decoys(tasks: list[dict]) -> list[str]:
    pool = list(DECOYS)
    pool += [
        e
        for t in tasks
        for e in (
            t["checker"].get("expect")
            if isinstance(t["checker"].get("expect"), list)
            else [t["checker"].get("expect")]
        )
        if isinstance(e, str) and e.strip()
    ]
    return pool


def _chat(prompt: str, temperature: float, max_tokens: int) -> dict:
    import requests

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.time()
    resp = requests.post(f"{SERVE}/v1/chat/completions", json=payload, timeout=900)
    resp.raise_for_status()
    data = resp.json()
    data["_elapsed_s"] = time.time() - started
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="first N tasks (smoke)")
    ap.add_argument("--samples", type=int, default=2,
                    help="sampled answers per task (plus 1 greedy)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-neg", type=int, default=3,
                    help="perturbed negatives per correct train answer")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    args = ap.parse_args()

    tasks = json.loads(TASKS_FILE.read_text())["tasks"]
    if args.limit:
        tasks = tasks[: args.limit]
    decoys = _decoys(tasks)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "harvest-bench.jsonl"
    manifest_path = out / "harvest-manifest.json"
    log_path = out / "harvest.log"
    log = log_path.open("w")

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    health_before = _probe_health()
    say(f"[harvest] serve {health_before} | model {MODEL}")
    say(f"[harvest] tasks {len(tasks)} | samples/task {args.samples} + 1 greedy | "
        f"temp {args.temp} | max_tokens {args.max_tokens} | thinking off")

    items: list[dict] = []
    stats = {
        "requests": 0, "kept": 0, "truncated": 0, "empty": 0, "errors": 0,
        "dup_skipped": 0, "correct": 0, "incorrect": 0, "perturbed": 0,
        "per_bucket": {}, "per_origin": {},
    }
    seen_per_task: dict[str, set[str]] = {}
    t0 = time.time()

    def flush(manifest_extra: dict | None = None) -> None:
        payload = "\n".join(json.dumps(i, sort_keys=True) for i in items) + "\n"
        jsonl_path.write_text(payload)
        manifest = {
            "created": time.strftime("%Y-%m-%d", time.gmtime()),
            "purpose": ("organic answers from the live 176B for each J-Bench task; "
                        "checker labels; train-side perturbation amplification; "
                        "test-side organic-only"),
            "serve": {"health_before": health_before, "model": _serve_model_info()},
            "config": {"samples": args.samples, "temp": args.temp,
                       "max_tokens": args.max_tokens, "enable_thinking": False,
                       "max_neg": args.max_neg, "limit": args.limit},
            "tasks": len(tasks),
            "items": len(items),
            "stats": stats,
            "elapsed_s": round(time.time() - t0, 1),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
        if manifest_extra:
            manifest.update(manifest_extra)
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    for n, task in enumerate(tasks, 1):
        tid = task["id"]
        seen = seen_per_task.setdefault(tid, set())
        specs = [("g", 0.0)] + [(f"s{i}", args.temp) for i in range(1, args.samples + 1)]
        for tag, temp in specs:
            try:
                data = _chat(task["prompt"], temp, args.max_tokens)
            except Exception as exc:  # noqa: BLE001 — keep harvesting
                stats["errors"] += 1
                say(f"  ! {tid}:{tag} request failed: {exc}")
                if stats["errors"] >= 10:
                    flush({"aborted": "too many request errors"})
                    say("[harvest] aborting: too many errors")
                    return 2
                time.sleep(5)
                continue
            stats["requests"] += 1
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = (message.get("content") or "").strip()
            finish = choice.get("finish_reason")
            usage = data.get("usage") or {}
            if finish != "stop":
                stats["truncated"] += 1
                continue
            if not content or content in seen:
                stats["empty" if not content else "dup_skipped"] += 1
                continue
            seen.add(content)
            label = bool(check(task, content))
            origin = "harvested:greedy" if temp == 0.0 else f"harvested:t{tag}"
            item = {
                "item_id": f"{tid}:h{tag}",
                "task_id": tid, "bucket": task["bucket"], "split": task["split"],
                "prompt": task["prompt"], "candidate": content,
                "label": int(label), "origin": origin, "source": "harvest",
                "gen": {"temperature": temp, "completion_tokens": usage.get("completion_tokens"),
                        "ms": round(data["_elapsed_s"] * 1000)},
            }
            items.append(item)
            stats["kept"] += 1
            stats["correct" if label else "incorrect"] += 1
            stats["per_bucket"].setdefault(task["bucket"], {"correct": 0, "incorrect": 0})
            stats["per_bucket"][task["bucket"]]["correct" if label else "incorrect"] += 1
            stats["per_origin"][origin] = stats["per_origin"].get(origin, 0) + 1

            if label and task["split"] == "train":
                k = 0
                for pert_label, cand in perturbations(task, content, decoys):
                    if k >= args.max_neg:
                        break
                    if cand in seen or check(task, cand):
                        continue
                    seen.add(cand)
                    k += 1
                    stats["perturbed"] += 1
                    items.append({
                        "item_id": f"{tid}:h{tag}:neg{k}",
                        "task_id": tid, "bucket": task["bucket"], "split": task["split"],
                        "prompt": task["prompt"], "candidate": cand, "label": 0,
                        "origin": f"harvested-perturbed:{pert_label}", "source": "harvest",
                        "gen": {"temperature": temp, "perturbation": pert_label},
                    })

        if n % 5 == 0 or n == len(tasks):
            elapsed = time.time() - t0
            say(f"[{n}/{len(tasks)}] kept {stats['kept']} "
                f"(+{stats['perturbed']} perturbed), wrong {stats['incorrect']}, "
                f"trunc {stats['truncated']}, err {stats['errors']} | {elapsed:.0f}s")
        if n % args.checkpoint_every == 0:
            flush()

    health_after = _probe_health()
    flush({"serve_health_after": health_after, "done": True})
    say(f"[harvest] done: {stats['kept']} organic ({stats['incorrect']} wrong) + "
        f"{stats['perturbed']} perturbed | {time.time()-t0:.0f}s | {health_after}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
