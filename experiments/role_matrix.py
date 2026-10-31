#!/usr/bin/env python
"""Role-specific model matrix for Hive.

Each task names a Hive role and carries a deterministic checker. Chat models are
called through their OpenAI-compatible endpoints; image tasks only run on models
marked vision-capable. This is a role scorecard, not a generic benchmark.
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.cascade.checkers import check  # noqa: E402

TASKS = REPO / "experiments" / "roles" / "tasks-roles.json"
OUTDIR = REPO / "experiments" / "role-matrix"

MODELS = {
    "scion35": {
        "base": "http://127.0.0.1:8082/v1",
        "model": "scion-35b-a3b-pq2_0-corr",
        "extra": {},
        "vision": False,
    },
    "lfm3": {
        "base": "http://127.0.0.1:8155/v1",
        "model": "lfm3",
        "extra": {"chat_template_kwargs": {"enable_thinking": False}},
        "vision": False,
    },
    "bonsai27": {
        "base": "http://127.0.0.1:8163/v1",
        "model": "bonsai",
        "extra": {},
        "vision": False,
    },
    "qwen35": {
        "base": "http://127.0.0.1:8162/v1",
        "model": "Qwen3.8-35B-A3B-IQ2_M",
        "extra": {},
        "vision": False,
    },
    "lfm3vl": {
        "base": "http://127.0.0.1:8160/v1",
        "model": "LFM3-VL-2B-Q8_0",
        "extra": {},
        "vision": True,
    },
}

SYSTEM = {
    "planner": "You are the Hive planner. Return only the requested JSON.",
    "controller": "You are the Hive controller. Return only the requested JSON.",
    "router": "You are the Hive router. Answer with exactly one letter.",
    "face": "You are the Hive face. Answer directly and concisely.",
    "agency": "You are the Hive agency role. Answer with exactly one letter.",
    "mechanics": "You are the Hive mechanics role. Return only the requested JSON.",
    "judge": "You are the Hive judge. Answer with exactly one letter.",
    "rag": "You are the Hive RAG reader. Use only the provided passage.",
    "risk": "You are the Hive risk classifier. Answer with exactly one letter.",
    "vision": "You are the Hive vision reader. Answer briefly.",
    "placement": "You are the Hive placement planner. Return only the requested answer.",
    "critic": "You are the Hive critic. Check the proposal and answer yes or no.",
}


def post(base: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def image_data_url(path: str) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    suffix = Path(path).suffix.lower().lstrip(".") or "png"
    return f"data:image/{suffix};base64,{data}"


def run_one(model_id: str, model: dict[str, Any], task: dict[str, Any],
            timeout: float) -> dict[str, Any]:
    if task.get("image") and not model["vision"]:
        return {"ok": None, "skipped": True, "answer": "", "error": "vision-only task",
                "wall_s": 0.0, "tokens": 0}
    content: Any = task["prompt"]
    if task.get("image"):
        content = [
            {"type": "text", "text": task["prompt"]},
            {"type": "image_url", "image_url": {"url": image_data_url(task["image"])}},
        ]
    body = {
        "model": model["model"],
        "messages": [
            {"role": "system", "content": SYSTEM.get(task["role"], "You are a Hive role.")},
            {"role": "user", "content": content},
        ],
        "max_tokens": 2560 if task["role"] in ("planner", "controller") else 1024,
        "temperature": 0.1,
        "stream": False,
    }
    body.update(model.get("extra") or {})
    if task["checker"].get("type") in ("json", "ordered"):
        body["response_format"] = {"type": "json_object"}
    started = time.perf_counter()
    try:
        out = post(model["base"], "/chat/completions", body, timeout)
        wall = time.perf_counter() - started
        choice = (out.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        answer = message.get("content") or ""
        if not str(answer).strip():
            answer = message.get("reasoning_content") or ""
        answer = str(answer)
        try:
            ok = bool(check(task, answer))
        except Exception as exc:  # noqa: BLE001 - a bad output is a failed task
            return {"ok": False, "answer": answer[:1200], "error": f"checker: {exc}",
                    "wall_s": round(wall, 3), "tokens": (out.get("usage") or {}).get("completion_tokens", 0)}
        return {
            "ok": ok,
            "answer": answer[:1200],
            "error": None,
            "wall_s": round(wall, 3),
            "tokens": (out.get("usage") or {}).get("completion_tokens", 0),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "answer": "", "error": str(exc)[:300],
                "wall_s": round(time.perf_counter() - started, 3), "tokens": 0}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("ok") is not None]
    if not scored:
        return {"n": 0}
    lat = [r["wall_s"] for r in scored]
    return {
        "n": len(scored),
        "accuracy": round(sum(bool(r["ok"]) for r in scored) / len(scored), 4),
        "median_s": round(statistics.median(lat), 3),
        "errors": sum(1 for r in scored if r.get("error")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(TASKS))
    parser.add_argument("--models", default="scion35,lfm3,lfm3vl")
    parser.add_argument("--roles", default="")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    tasks = json.loads(Path(args.tasks).read_text())["tasks"]
    roles = {r.strip() for r in args.roles.split(",") if r.strip()}
    if roles:
        tasks = [t for t in tasks if t["role"] in roles]
    selected = [m for m in args.models.split(",") if m in MODELS]
    if not selected:
        raise SystemExit(f"no known models in {args.models!r}")

    rows = []
    for model_id in selected:
        model = MODELS[model_id]
        for task in tasks:
            for sample in range(args.samples):
                result = run_one(model_id, model, task, args.timeout)
                row = {"model": model_id, "role": task["role"], "task_id": task["id"],
                       "sample": sample, **result}
                rows.append(row)
                status = "skip" if result.get("skipped") else ("ok" if result["ok"] else "FAIL")
                print(f"  {model_id:10s} {task['id']:16s} {status:4s} "
                      f"{result['wall_s']:6.2f}s {str(result.get('error') or '')[:60]}", flush=True)

    by_model = defaultdict(list)
    by_role = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
        by_role[(row["model"], row["role"])].append(row)
    summary = {
        "models": {mid: summarise(rs) for mid, rs in by_model.items()},
        "by_role": {f"{mid}/{role}": summarise(rs) for (mid, role), rs in by_role.items()},
        "roles": sorted({t["role"] for t in tasks}),
        "samples": args.samples,
        "rows": rows,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUTDIR / f"role-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print("\nby model")
    for mid, s in summary["models"].items():
        print(f"  {mid:10s} n={s.get('n', 0):2d} acc={s.get('accuracy', 0):.3f} "
              f"median={s.get('median_s', 0):.2f}s")
    print("by role")
    for key, s in summary["by_role"].items():
        print(f"  {key:22s} n={s.get('n', 0):2d} acc={s.get('accuracy', 0):.3f} "
              f"median={s.get('median_s', 0):.2f}s")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
