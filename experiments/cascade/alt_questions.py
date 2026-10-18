#!/usr/bin/env python
"""Ask different questions of the same Intern-Decision state.

The gate's holistic verdict collapses arithmetic slips, edge cases, and
requirement completeness into one judgement. This probe keeps the model, the
state, and the deployment noul interface, and only changes the question — the
cheap "recompute differently" tier. Scores are P(question true) (inverted for
the error question) and are compared with the frozen L0.5 probe on the organic
test-66.

Usage:
    .venv/bin/python experiments/cascade/alt_questions.py \
        --corpus experiments/cascade/results/decision-harvest-r2-20261017/harvest-bench-r2.jsonl \
        --out experiments/cascade/results/decision-alt-questions-20261018
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cascade.decision import InternDecisionJudge  # noqa: E402
from experiments.cascade.decision_readout import _apply_saved_heads  # noqa: E402
from experiments.cascade.readout_judge import metrics  # noqa: E402

QUESTIONS = {
    "holistic": "The candidate answer is correct and complete for the task.",
    "arithmetic": "The candidate answer's arithmetic and final numerical result are correct.",
    "edge_cases": "The candidate answer handles the task's edge cases and boundary conditions correctly.",
    "error": "The candidate answer contains at least one concrete error that changes the result.",
    "requirements": "The candidate answer satisfies every stated requirement of the task.",
}
ERROR_DIRECTION = {"error"}  # score = 1 - p

DEFAULT_CORPUS = REPO_ROOT / "experiments/cascade/results/decision-harvest-r2-20261017/harvest-bench-r2.jsonl"
R2_DIR = REPO_ROOT / "experiments/cascade/results/decision-readout-r2-20261017"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def health() -> dict:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=5) as r:
            return {"url": "http://127.0.0.1:8081/health", "status": r.status}
    except Exception as exc:  # noqa: BLE001
        return {"url": "http://127.0.0.1:8081/health", "error": str(exc)}


def frozen_l05(items: list[dict]) -> np.ndarray | None:
    """Frozen r2 L0.5 p for the same items, if the artifacts are present."""
    states = R2_DIR / "readout-states-intern-decision-0.8b-r2.npz"
    heads = R2_DIR / "readout-head-intern-decision-0.8b-r2.npz"
    if not (states.exists() and heads.exists()):
        return None
    z = np.load(states)
    pos = {i: k for k, i in enumerate(z["item_ids"].astype(str))}
    if any(i["item_id"] not in pos for i in items):
        return None
    k = np.array([pos[i["item_id"]] for i in items])
    h = np.load(heads)
    s = ((z["L0.5"][k] - h["ridge_L0.5_mu"]) / h["ridge_L0.5_sd"]) @ h["ridge_L0.5_w"] + float(h["ridge_L0.5_bias"])
    a, b = h["ridge_L0.5_platt"]
    return 1 / (1 + np.exp(-(a * s + b)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="internlm/Intern-Decision-0.8B")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "experiments/cascade/results/decision-alt-questions-20261018")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = [i for i in load_jsonl(args.corpus)
             if i["split"] == "test" and i["origin"].startswith("harvested:")]
    if args.limit:
        items = items[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[alt] {len(items)} organic test items x {len(QUESTIONS)} questions", flush=True)

    before = health()
    judge = InternDecisionJudge(args.model, device=args.device, dtype=args.dtype)
    print(f"[alt] serve before: {before}", flush=True)

    p: dict[str, list[float]] = {q: [] for q in QUESTIONS}
    ms: list[float] = []
    t0 = time.time()
    for n, item in enumerate(items, 1):
        state = {"task": item["prompt"], "candidate_answer": item["candidate"]}
        for q, text in QUESTIONS.items():
            prob, latency = judge.noul(state, text)
            p[q].append(float(prob))
            ms.append(float(latency))
        if n % 10 == 0 or n == len(items):
            print(f"[{n}/{len(items)}] {time.time()-t0:.0f}s", flush=True)

    after = health()
    y = np.array([int(i["label"]) for i in items])
    ids = [i["item_id"] for i in items]
    print(f"\n[alt] serve after: {after} | {time.time()-t0:.0f}s | "
          f"median {statistics.median(ms):.0f} ms/question", flush=True)

    results: dict[str, dict] = {}
    print(f"\n{'question':14s} {'AUC':>6s} {'caught':>7s} {'FR':>5s}  wrongs p (min..max)")
    for q in QUESTIONS:
        score = np.array(p[q])
        if q in ERROR_DIRECTION:
            score = 1.0 - score
        rows = [{"label": int(it["label"]), "p_correct": float(s)} for it, s in zip(items, score)]
        m = metrics(rows)
        at = m["at_0.5"]
        caught = int(sum(1 for j in range(len(y)) if y[j] == 0 and score[j] < 0.5))
        wrong_p = score[y == 0]
        auc = f"{m['auc']:.3f}" if m["auc"] is not None else "n/a  "
        print(f"{q:14s} {auc} {caught:5d}/9 {at['false_rejects']:4d}/57  "
              f"{wrong_p.min():.3f}..{wrong_p.max():.3f}")
        results[q] = {"auc": (round(m["auc"], 4) if m["auc"] is not None else None),
                      "acc": at["correct"], "fa": at["false_accepts"],
                      "fr": at["false_rejects"], "caught": caught,
                      "wrongs_p": {i: round(float(s), 4)
                                   for i, s, lab in zip(ids, score, y) if lab == 0}}

    l05 = frozen_l05(items)
    if l05 is not None:
        print("\n[alt] union with frozen L0.5 @0.55 (reject if either rejects):")
        l05_rej = l05 < 0.55
        for q in QUESTIONS:
            score = 1.0 - np.array(p[q]) if q in ERROR_DIRECTION else np.array(p[q])
            rej = l05_rej | (score < 0.5)
            caught = int(sum(1 for j in range(len(y)) if y[j] == 0 and rej[j]))
            fr = int(sum(1 for j in range(len(y)) if y[j] == 1 and rej[j]))
            print(f"  L0.5 + {q:12s} caught {caught}/9  FR {fr:2d}/57")
            results[q]["union_l05_55"] = {"caught": caught, "fr": fr}
        results["_l05"] = {"p": [round(float(x), 6) for x in l05]}
    else:
        print("\n[alt] frozen L0.5 artifacts not found; skipping unions")

    out = {"model": args.model, "corpus": str(args.corpus),
           "n": len(items), "questions": QUESTIONS,
           "error_direction": sorted(ERROR_DIRECTION),
           "elapsed_s": round(time.time() - t0, 1),
           "median_ms": round(statistics.median(ms), 1),
           "serve": {"before": before, "after": after},
           "results": results,
           "p": {q: [round(float(x), 6) for x in p[q]] for q in QUESTIONS},
           "item_ids": ids, "labels": [int(x) for x in y]}
    dest = args.out / "alt-questions.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[alt] wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
