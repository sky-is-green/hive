#!/usr/bin/env python
"""Evaluate a local HF LFM3 checkpoint on the 74-task role suite.

Same prompts, system messages and deterministic checkers as
``experiments/role_matrix.py``; the difference is that generation runs in
process through transformers instead of an OpenAI endpoint, so a freshly
fine-tuned checkpoint can be scored before it is converted to GGUF.

Vision tasks are skipped for text-only checkpoints and reported as such.

    HIP_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
      ~/.unsloth/studio/unsloth_studio/bin/python experiments/eval_role_sft.py \
        --model-dir ../train-runs/lfm3-role-sft-v1/merged \
        --out ../train-runs/lfm3-role-sft-v1/eval-role.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import check  # noqa: E402
from experiments.role_matrix import SYSTEM  # noqa: E402

TASKS = REPO / "experiments" / "roles" / "tasks-roles.json"


def summarise(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("ok") is not None]
    if not scored:
        return {"n": 0, "skipped": len(rows)}
    lat = [r["wall_s"] for r in scored]
    return {
        "n": len(scored),
        "skipped": len(rows) - len(scored),
        "accuracy": round(sum(bool(r["ok"]) for r in scored) / len(scored), 4),
        "median_s": round(statistics.median(lat), 3),
        "tokens": sum(r.get("tokens", 0) for r in scored),
    }


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tasks", default=str(TASKS))
    parser.add_argument("--out", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--roles", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--vision", action="store_true", help="do not skip image tasks")
    parser.add_argument("--max-new-tokens", type=int, default=0,
                        help="override; default 512 for planner/controller, 192 otherwise")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tasks = json.loads(Path(args.tasks).read_text())["tasks"]
    roles = {r.strip() for r in args.roles.split(",") if r.strip()}
    if roles:
        tasks = [t for t in tasks if t["role"] in roles]
    if args.limit:
        tasks = tasks[:args.limit]

    print(f"loading {args.model_dir} on {args.device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, trust_remote_code=True)
    model.to(args.device)
    model.eval()

    rows: list[dict] = []
    for task in tasks:
        if task.get("image") and not args.vision:
            rows.append({"role": task["role"], "task_id": task["id"], "ok": None,
                         "skipped": True, "answer": "", "wall_s": 0.0, "tokens": 0,
                         "error": "vision task skipped"})
            print(f"  {task['id']:16s} skip (vision)", flush=True)
            continue
        messages = [
            {"role": "system", "content": SYSTEM.get(task["role"], "You are a Hive role.")},
            {"role": "user", "content": task["prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        limit = args.max_new_tokens or (512 if task["role"] in ("planner", "controller") else 192)
        started = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=limit,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else 1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        wall = time.perf_counter() - started
        new_ids = output[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        tokens = len(new_ids)
        try:
            ok = bool(check(task, answer))
        except Exception as exc:  # noqa: BLE001 - a bad answer is a failed task
            ok = False
            answer = f"{answer}\n[checker error: {exc}]"
        rows.append({"role": task["role"], "task_id": task["id"], "ok": ok,
                     "answer": answer, "wall_s": round(wall, 3), "tokens": tokens,
                     "error": None})
        print(f"  {task['id']:16s} {'ok' if ok else 'FAIL':4s} {wall:6.2f}s "
              f"{answer[:70]!r}", flush=True)

    by_model = summarise(rows)
    by_role: dict[str, dict] = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["role"]].append(row)
    for role, rs in grouped.items():
        by_role[role] = summarise(rs)
    summary = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "device": args.device,
        "temperature": args.temperature,
        "overall": by_model,
        "by_role": by_role,
        "rows": rows,
    }
    print("\noverall", json.dumps(by_model))
    print("by role")
    for role, stats in by_role.items():
        print(f"  {role:12s} n={stats.get('n', 0):2d} "
              f"acc={stats.get('accuracy', 0):.3f} median={stats.get('median_s', 0):.2f}s")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=1) + "\n")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
