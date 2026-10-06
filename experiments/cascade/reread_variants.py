#!/usr/bin/env python
"""Re-read variants: does including the reasoning tail change the verdict read?

For every sample of the correction run (and the hinted run if present), the
judge prompt is rebuilt with ``candidate = reasoning tail + answer`` instead of
the answer alone.  Compares, on held-out tasks:

  frozen head, answer-only      (baseline, from the saved samples)
  frozen head, reasoning+answer (prompt-format transfer)
  refit head, reasoning+answer  (same task, fitted on the variant)

Usage:
    reread_variants.py --server http://127.0.0.1:8897
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from probe_judge import ridge_fit, ridge_score  # noqa: E402
from readout_judge import auc, order_body  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"
SYSTEM = "You are a decision function. Reply with the answer label only."


def post(url: str, path: str, body: dict, timeout: float = 600.0) -> dict:
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
    ap.add_argument("--out", type=Path, default=R / "reread-variant.json")
    ap.add_argument("--tail-chars", type=int, default=1000)
    args = ap.parse_args()

    head = json.loads(args.head.read_text())
    W = np.asarray(head["weights"]["w"])
    MU = np.asarray(head["weights"]["mu"])
    SD = np.asarray(head["weights"]["sd"])
    B = head["weights"]["b"]
    PA, PC = head["weights"]["platt"]
    frozen = (W, MU, SD, B)

    tasks = {t["id"]: t for t in json.loads(
        (CASCADE / "tasks-bench.json").read_text())["tasks"]}
    samples = [json.loads(line) for line in args.samples.read_text().splitlines()]
    rows, embs = [], []
    t0 = time.time()
    for i, rec in enumerate(samples, 1):
        reasoning = (rec.get("reasoning") or "").strip()
        candidate = rec["content"]
        if reasoning:
            candidate = reasoning[-args.tail_chars:] + "\n\n" + candidate
        prompt = post(args.server, "/apply-template", {
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": order_body(
                             {"prompt": tasks[rec["task_id"]]["prompt"],
                              "candidate": candidate}, False)}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        emb = post(args.server, "/embedding", {"content": prompt,
                                               "embd_normalize": -1})
        embs.append(np.asarray(emb[0]["embedding"][0], dtype=np.float64))
        rows.append({"task_id": rec["task_id"], "sample": rec["sample"],
                     "split": rec["split"], "label": int(bool(rec["ok"])),
                     "p_answer": rec["p_correct"]})
        if i % 50 == 0:
            print(f"[{i}/{len(samples)}] {time.time()-t0:.0f}s", flush=True)

    X = np.stack(embs)
    rows_by: dict[str, list] = defaultdict(list)
    for i, r in enumerate(rows):
        r["p_variant_frozen"] = float(ridge_score(X[i][None, :], frozen)[0])
        rows_by[r["task_id"]].append(i)
    # refit on train samples (variant format)
    tr = [i for i, r in enumerate(rows) if r["split"] == "train"]
    te = [i for i, r in enumerate(rows) if r["split"] == "test"]
    y = np.asarray([r["label"] for r in rows], dtype=np.float64)
    model = ridge_fit(X[tr], y[tr])
    s_all = ridge_score(X, model)
    for i, r in enumerate(rows):
        r["p_variant_fit"] = float(s_all[i])

    def metrics(key):
        return {"auc_test": auc([r[key] for r in rows if r["split"] == "test"],
                                [r["label"] for r in rows if r["split"] == "test"])}

    def bestof4(key):
        ok = n = 0
        for tid, idx in rows_by.items():
            if rows[idx[0]]["split"] != "test":
                continue
            n += 1
            ok += int(any(rows[i]["label"] for i in idx))
            # covered by oracle only; selection accuracy computed below
        return n

    def selection(key):
        ok = n = 0
        for tid, idx in rows_by.items():
            if rows[idx[0]]["split"] != "test":
                continue
            n += 1
            pick = max(idx, key=lambda i: rows[i][key])
            ok += int(rows[pick]["label"])
        return {"n": n, "correct": ok, "acc": ok / max(1, n)}

    out = {
        "n_samples": len(rows),
        "auc_test": {"p_answer": metrics("p_answer")["auc_test"],
                     "p_variant_frozen": metrics("p_variant_frozen")["auc_test"],
                     "p_variant_fit": metrics("p_variant_fit")["auc_test"]},
        "selection_bestof4_test": {"answer_frozen": selection("p_answer"),
                                   "variant_frozen": selection("p_variant_frozen"),
                                   "variant_fit": selection("p_variant_fit")},
    }
    args.out.write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    print(json.dumps(out, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
