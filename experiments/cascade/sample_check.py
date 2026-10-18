"""Sample k fresh local drafts on mined-hard tasks and checker-label them.

The escalation lane harvests ladder-resolved items (flagged / escalated /
human-gated).  Those tasks are mined hard tasks.  For each, this script
samples k drafts at temperature 0.7 from the local tier, checker-labels
each (labels are eval-only), extracts the 0.8B readout states, and appends
to a labeled corpus on the *silent* distribution.

This is the seed for re-auditing / retraining the learned gate: it mints
labels for the exact distribution the gate must operate on (fragile corrects
+ confident wrongs), which the stored 1315-item corpus does not cover.

Output: <out>/corpus.jsonl (one line per (task, draft): run, task_id, bucket,
prompt, draft, label, ms) + <out>/corpus_states.npz (arrays L0.5/L0.75/L1,
shape [N, 1024]) + per-run <out>/<run>/sample_check.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import check
from experiments.cascade.run_cascade import ScionTier, answer_prompt


def select_tasks(report, tasks, args):
    """Pick the task list for this run based on mode."""
    if args.all_tasks:
        return list(tasks.values())
    if args.sample_tasks and args.sample_tasks > 0:
        rng = random.Random(args.seed)
        pool = list(tasks.values())
        rng.shuffle(pool)
        return pool[: args.sample_tasks]
    # default: mined-hard tasks (ladder-touched)
    out = []
    seen = set()
    for r in report["records"]:
        if r.get("gated"):
            continue
        if r.get("scion_ok") is None:
            continue
        if not (r.get("escalated") or r.get("flagged") or r.get("human_gate") is not None):
            continue
        tid = r["id"]
        if tid in seen:
            continue
        seen.add(tid)
        task = tasks.get(tid)
        if task is None:
            continue
        out.append(task)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs (each with run/report.json + tasks-organic66.json)")
    ap.add_argument("--k", type=int, default=4, help="drafts per task")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--scion-base", default="http://127.0.0.1:8081/v1")
    ap.add_argument("--out", default=str(REPO / "experiments/cascade/results/silent-labeled-corpus"))
    ap.add_argument("--limit", type=int, default=0, help="cap number of tasks (smoke)")
    ap.add_argument("--all-tasks", action="store_true", help="sample on every task in the tasks file")
    ap.add_argument("--sample-tasks", type=int, default=0, help="randomly sample N tasks (representative slice)")
    ap.add_argument("--seed", type=int, default=42, help="rng seed for --sample-tasks")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from harness.cascade.learned_gate import LearnedGate

    scion = ScionTier(args.scion_base, max_tokens=args.max_tokens, temperature=args.temperature)
    if not scion.alive():
        print(f"[sample_check] scion serve not alive at {args.scion_base}", file=sys.stderr)
        sys.exit(2)

    gate = LearnedGate()  # loads backbone lazily

    # Accumulate states + metadata across the whole corpus.
    all_meta = []
    L05, L075, L1 = [], [], []

    for run_dir in args.runs:
        run_dir = Path(run_dir)
        report = json.loads((run_dir / "run" / "report.json").read_text())
        tasks = {t["id"]: t for t in json.loads((run_dir / "tasks-organic66.json").read_text())["tasks"]}
        hts = select_tasks(report, tasks, args)
        if args.limit:
            hts = hts[: args.limit]
        run_name = run_dir.name
        print(f"[{run_name}] {len(hts)} hard tasks, k={args.k}")

        run_records = []
        for i, task in enumerate(hts, 1):
            tid = task['id']
            prompt = task["prompt"]
            drafts = []
            for d in range(args.k):
                t0 = time.perf_counter()
                try:
                    messages = answer_prompt(task)
                    out = scion.chat(messages, max_tokens=args.max_tokens, temperature=args.temperature)
                    draft = (out.get("content") or "").strip()
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{i}/{len(hts)}] {tid} draft {d+1} ERROR: {exc}", file=sys.stderr)
                    draft = ""
                ms_gen = (time.perf_counter() - t0) * 1000.0
                if not draft:
                    drafts.append({"draft": "", "label": False, "ms_gen": ms_gen, "error": "empty"})
                    print(f"  [{i}/{len(hts)}] {tid} draft {d+1}: EMPTY ({ms_gen:.0f}ms)")
                    continue
                label = bool(check(task, draft))
                # extract states
                try:
                    feats = gate.states(prompt, draft)
                    l05 = feats["L0.5"]; l075 = feats["L0.75"]; l1 = feats["L1"]
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{i}/{len(hts)}] {tid} draft {d+1} state ERROR: {exc}", file=sys.stderr)
                    drafts.append({"draft": draft, "label": label, "ms_gen": ms_gen, "error": str(exc)})
                    continue
                ms_state = 0.0
                L05.append(l05); L075.append(l075); L1.append(l1)
                meta = {
                    "run": run_name,
                    "task_id": tid,
                    "bucket": task.get("bucket"),
                    "prompt": prompt,
                    "draft": draft,
                    "label": label,  # True = checker passes (correct)
                    "ms_gen": round(ms_gen, 1),
                }
                all_meta.append(meta)
                drafts.append({"draft": draft, "label": label, "ms_gen": round(ms_gen, 1)})
                tag = "OK  " if label else "WRONG"
                print(f"  [{i}/{len(hts)}] {tid} draft {d+1}: {tag} ({ms_gen:.0f}ms gen)")

            run_records.append({"task_id": tid, "bucket": task.get("bucket"),
                                "n_drafts": len(drafts), "drafts": drafts})

        (out_dir / run_name).mkdir(parents=True, exist_ok=True)
        (out_dir / run_name / "sample_check.json").write_text(
            json.dumps({"run": run_name, "k": args.k, "temperature": args.temperature,
                        "tasks": run_records}, indent=1))

    # Write merged corpus.
    if all_meta:
        (out_dir / "corpus.jsonl").write_text(
            "\n".join(json.dumps(m, ensure_ascii=False) for m in all_meta) + "\n")
        np.savez(out_dir / "corpus_states.npz",
                 L0_5=np.asarray(L05, dtype=np.float32),
                 L0_75=np.asarray(L075, dtype=np.float32),
                 L1=np.asarray(L1, dtype=np.float32))
        n_wrong = sum(1 for m in all_meta if not m["label"])
        n_right = sum(1 for m in all_meta if m["label"])
        print(f"\n[done] corpus: {len(all_meta)} drafts ({n_right} correct, {n_wrong} wrong) "
              f"-> {out_dir}/corpus.jsonl + corpus_states.npz")
    else:
        print("\n[done] no drafts collected", file=sys.stderr)


if __name__ == "__main__":
    main()
