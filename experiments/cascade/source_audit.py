#!/usr/bin/env python
"""Source-swap audit: does the decision head read correctness, or authorship?

Two answer sources for the same 100 bench tasks:
  * ``scion``  — the generator's own answers (scion-answers.json, checker-labelled)
  * ``strata`` — the recorded answers from another model (bench reports)

The head is fitted exactly as in probe_scion.py (train split of J-Bench, which
is entirely Strata-sourced) and then applied **frozen** to both sources.  If it
reads correctness, per-source AUC/acceptance should match; if it reads
authorship, the generator's own answers score differently at the same label.

The prompted leg runs separately (readout_judge.py --server-chat over the
scion-items JSONL) for the behavioral self-preference baseline.

Usage:
    source_audit.py --server http://127.0.0.1:8897 \
        --scion-answers results/jev-readout-20261006/scion-answers.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
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
STRATA_REPORTS = [
    "results/strata-bench-train-20261003/report.json",
    "results/strata-bench-test-labels-20261003/report.json",
]


def post(url: str, path: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def embed(server: str, item: dict) -> np.ndarray:
    prompt = post(server, "/apply-template", {
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": order_body(item, False)}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    emb = post(server, "/embedding", {"content": prompt, "embd_normalize": -1})
    return np.asarray(emb[0]["embedding"][0], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--scion-answers", type=Path, default=R / "scion-answers.json")
    ap.add_argument("--states", type=Path, default=R / "probe-states-scion35b.npz")
    ap.add_argument("--out", type=Path, default=R / "source-audit.json")
    args = ap.parse_args()

    # frozen head: refit exactly as probe_scion.py does (deterministic)
    z = np.load(args.states, allow_pickle=True)
    ids = [str(x) for x in z["item_ids"]]
    labels = np.asarray(z["labels"], dtype=np.float64)
    splits = [str(s) for s in z["splits"]]
    X = z["L1"].astype(np.float64)
    tr = [i for i, s in enumerate(splits) if s == "train"]
    model = ridge_fit(X[tr], labels[tr])

    scion = json.loads(args.scion_answers.read_text())["records"]
    strata: dict[str, dict] = {}
    for rel in STRATA_REPORTS:
        report = json.loads((CASCADE / rel).read_text())
        for rec in report.get("records", []):
            if rec.get("scion_answer"):
                strata[rec["id"]] = rec

    rows = []
    t0 = time.time()
    for rec in scion:
        task_id = rec["id"]
        other = strata.get(task_id)
        if other is None:
            continue
        base = {"task_id": task_id, "bucket": rec["bucket"],
                "split": rec["split"], "prompt": rec["prompt"]}
        own = {**base, "source": "scion", "candidate": rec["scion_answer"],
               "label": int(bool(rec["scion_ok"]))}
        foreign = {**base, "source": "strata", "candidate": other["scion_answer"],
                   "label": int(bool(other.get("scion_ok")))}
        for item in (own, foreign):
            item["score"] = float(ridge_score(embed(args.server, item)[None, :],
                                              model)[0])
        rows.append({"task_id": task_id, "bucket": rec["bucket"],
                     "split": rec["split"],
                     "scion": {"candidate": own["candidate"], "label": own["label"],
                               "score": own["score"]},
                     "strata": {"candidate": foreign["candidate"],
                                "label": foreign["label"],
                                "score": foreign["score"]}})
        if len(rows) % 10 == 0:
            print(f"[{len(rows)}] {time.time()-t0:.0f}s", flush=True)

    out = {"n": len(rows), "head": "scion35b L1 ridge (refit, deterministic)"}
    for src in ("scion", "strata"):
        s = [r[src] for r in rows]
        out[src] = {
            "n": len(s),
            "auc": auc([x["score"] for x in s], [x["label"] for x in s]),
            "mean_score_correct": float(np.mean([x["score"] for x in s if x["label"]])),
            "mean_score_wrong": float(np.mean([x["score"] for x in s if not x["label"]])),
            "accept_rate": float(np.mean([x["score"] >= 0.5 for x in s])),
        }
    # self-preference delta = own minus foreign at equal label
    def delta(label: int) -> float | None:
        a = [r["scion"]["score"] for r in rows if r["scion"]["label"] == label]
        b = [r["strata"]["score"] for r in rows if r["strata"]["label"] == label]
        return float(np.mean(a) - np.mean(b)) if a and b else None
    out["self_preference_delta"] = {"correct": delta(1), "wrong": delta(0)}
    paired = [r for r in rows if r["scion"]["label"] and r["strata"]["label"]]
    out["paired_correct"] = {
        "n": len(paired),
        "mean_delta": float(np.mean([r["scion"]["score"] - r["strata"]["score"]
                                     for r in paired])) if paired else None,
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
