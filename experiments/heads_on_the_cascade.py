#!/usr/bin/env python3
"""The 176B's decision heads, compared on the campaign's own 28 tasks.

Two heads ship for the 176B-A6B:

  the J-Bench head   `scion-176b-a6b-decision-head.json`   (AUC 0.9941 on its
                     own frozen test split - 116 items of the judge bench)
  the community head `scion176b-community-indomain-head.json` (per-sample
                     AUC 0.852, within-pool 0.965, over 1395 training samples
                     of the community task set)

The dataset's own record says the in-domain refit beats the frozen
J-Bench head for selection and veto on this body, and that "validate per
model" is the rule.  Neither head has been measured on the cascade's
organic set, and neither has been measured against the *other* model's
answers at the operating point the cascade would use.

This scores both heads, live, on the 28 organic tasks and the 35B's own
answers to them, and reports the separation and the cost at a few
thresholds.

Usage: heads_on_the_cascade.py [--out JSON]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARM = REPO / "experiments/cascade/results/cascade-arm-escalation-20261020/report.json"
TASKS = REPO / "experiments/cascade/tasks-organic-28.json"
WORK = Path("/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/scion/decision-heads-work")
HEADS = {
    "the J-Bench head": WORK / "scion-176b-a6b-decision-head.json",
    "the community head": WORK / "overnight/scion176b-community-indomain-head.json",
}
BASE = "http://127.0.0.1:8081"


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for a in pos for b in neg if a > b) + \
        0.5 * sum(1 for a in pos for b in neg if a == b)
    return round(wins / (len(pos) * len(neg)), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    recs = json.loads(ARM.read_text())["records"]
    tasks = {t["id"]: t for t in json.loads(TASKS.read_text())["tasks"]}

    # the states of the two heads, applied client-side to the live readouts
    import numpy as np
    heads = {}
    for name, path in HEADS.items():
        h = json.loads(path.read_text())
        w = h["weights"]
        heads[name] = {
            "w": np.array(w["w"]), "mu": np.array(w["mu"]), "sd": np.array(w["sd"]),
            "b": float(w["b"]), "a": float(w["platt"][0]), "c": float(w["platt"][1]),
            "threshold": float((h.get("test") or {}).get("threshold") or 0.5),
            "path": str(path),
        }

    rows = []
    for r in recs:
        task = tasks.get(r["id"])
        if task is None or not (r.get("scion_answer") or "").strip():
            continue
        # The readout: the judge's prompt, rendered by the model's own chat
        # template (the head was fitted on the *templated* prompt - a raw
        # string is a different prompt, and the head reads nothing from it).
        # The vector is raw float32: the head was fitted on it unnormalised,
        # and /v1/embeddings L2-normalises by default.
        try:
            req = urllib.request.Request(
                BASE + "/apply-template",
                data=json.dumps({"messages": _judge_messages(
                    task["prompt"], r["scion_answer"]),
                    "chat_template_kwargs": {"enable_thinking": False}}).encode(),
                headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                prompt = json.loads(resp.read())["prompt"]
        except Exception as exc:  # noqa: BLE001
            print("  %-14s the template failed: %s" % (r["id"][:14], exc), flush=True)
            continue
        req = urllib.request.Request(
            BASE + "/embedding",
            data=json.dumps({"content": prompt, "embd_normalize": -1}).encode(),
            headers={"content-type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                d = json.loads(resp.read())
            vec = d[0]["embedding"][0]
        except Exception as exc:  # noqa: BLE001
            print("  %-14s the readout failed: %s" % (r["id"][:14], exc), flush=True)
            continue
        row = {"id": r["id"], "the_35B_ok": bool(r.get("scion_ok")),
               "the_readout_s": round(time.time() - t0, 2)}
        for name, h in heads.items():
            v = np.array(vec, dtype=np.float64)
            z = float(h["w"] @ ((v - h["mu"]) / h["sd"]) + h["b"])
            p = 1.0 / (1.0 + np.exp(-(h["a"] * z + h["c"])))
            row[name] = p
        rows.append(row)
        print("  %-14s the 35B %-5s  the J-Bench p=%.4f  the community p=%.4f"
              % (r["id"][:14], "ok" if row["the_35B_ok"] else "WRONG",
                 row["the J-Bench head"], row["the community head"]), flush=True)

    out = {"n": len(rows), "the_heads": {}}
    for name in heads:
        pos = [x[name] for x in rows if x["the_35B_ok"]]
        neg = [x[name] for x in rows if not x["the_35B_ok"]]
        out["the_heads"][name] = {
            "the_threshold": heads[name]["threshold"],
            "the_auc": auc(pos, neg),
            "the_mean_p_on_the_correct": round(statistics.mean(pos), 4) if pos else None,
            "the_mean_p_on_the_wrong": round(statistics.mean(neg), 4) if neg else None,
        }
        for thr in (0.5, 0.8, 0.9, 0.92, 0.95):
            caught = sum(1 for x in rows if not x["the_35B_ok"] and x[name] < thr)
            kept = sum(1 for x in rows if x["the_35B_ok"] and x[name] >= thr)
            out["the_heads"][name]["at_%.2f" % thr] = {
                "the_errors_caught": caught, "the_errors": len(neg),
                "the_correct_kept": kept, "the_correct": len(pos),
            }
    out["the_mean_readout_s"] = round(statistics.mean([x["the_readout_s"] for x in rows]), 2)
    print("\n" + json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    return 0


def _judge_messages(task: str, candidate: str) -> list[dict]:
    """The judge's prompt, in the same shape the head was fitted on.

    The engine's own ``verdict()`` builds exactly this: the system turn,
    the question, the candidate, and the two labels.
    """
    return [
        {"role": "system", "content": "You are a decision function. Reply with the answer label only."},
        {"role": "user", "content": (f"Question: {task}\nCandidate answer: {candidate}\n"
                                     "Is the candidate answer correct?\nA. Yes\nB. No\n")},
    ]


if __name__ == "__main__":
    raise SystemExit(main())
