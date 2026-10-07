#!/usr/bin/env python
"""Controlled attribution test: identical text, own vs other attribution.

Each candidate is presented twice, with the only difference being who it is
attributed to ("Your own answer" vs "Another model's answer").  If the head or
the prompted judge reads correctness, the score must not move; if it reads
authorship, the "own" presentation scores higher.  Both answer sources are
tested (Scion's own answers and Strata's recorded answers), and both readouts
(frozen head + prompted).

Usage:
    attribution_audit.py --server http://127.0.0.1:8897
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from probe_judge import ridge_fit, ridge_score  # noqa: E402
from readout_judge import _is_label  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"
SYSTEM = "You are a decision function. Reply with the answer label only."
STRATA_REPORTS = [
    "results/strata-bench-train-20261003/report.json",
    "results/strata-bench-test-labels-20261003/report.json",
]
ATTRIB = {"own": "Your own", "other": "Another model's"}


def post(url: str, path: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def user_body(item: dict, attrib: str) -> str:
    return (f"Question: {item['prompt']}\n"
            f"{ATTRIB[attrib]} answer: {item['candidate']}\n"
            "Is the candidate answer correct?\nA. Yes\nB. No\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--states", type=Path, default=R / "probe-states-scion35b.npz")
    ap.add_argument("--answers", type=Path, default=R / "scion-answers.json",
                    help="own-answers records (id/split/prompt/scion_answer/"
                         "scion_ok), as written by gen_answers.py or "
                         "bonsai_eval.py --phase items")
    ap.add_argument("--states-out", type=Path, default=None,
                    help="embedding cache path (default: <out dir>/"
                         "attribution-states.npz)")
    ap.add_argument("--out", type=Path, default=R / "attribution-audit.json")
    args = ap.parse_args()

    z = np.load(args.states, allow_pickle=True)
    labels = np.asarray(z["labels"], dtype=np.float64)
    splits = [str(s) for s in z["splits"]]
    X = z["L1"].astype(np.float64)
    tr = [i for i, s in enumerate(splits) if s == "train"]
    model = ridge_fit(X[tr], labels[tr])

    scion = json.loads(args.answers.read_text())["records"]
    strata: dict[str, dict] = {}
    for rel in STRATA_REPORTS:
        report = json.loads((CASCADE / rel).read_text())
        for rec in report.get("records", []):
            if rec.get("scion_answer"):
                strata[rec["id"]] = rec

    rows = []
    embs: dict[str, np.ndarray] = {}
    t0 = time.time()
    for rec in scion:
        other = strata.get(rec["id"])
        if other is None:
            continue
        for src, cand, label in (("scion", rec["scion_answer"], int(bool(rec["scion_ok"]))),
                                 ("strata", other["scion_answer"], int(bool(other.get("scion_ok"))))):
            item = {"prompt": rec["prompt"], "candidate": cand}
            entry = {"task_id": rec["id"], "source": src, "label": label}
            for attrib in ("own", "other"):
                prompt = post(args.server, "/apply-template", {
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": user_body(item, attrib)}],
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
                emb = post(args.server, "/embedding",
                           {"content": prompt, "embd_normalize": -1})
                vec = np.asarray(emb[0]["embedding"][0], dtype=np.float32)
                embs[f"{rec['id']}:{src}:{attrib}"] = vec
                score = float(ridge_score(vec[None, :], model)[0])
                cp = post(args.server, "/completion", {
                    "prompt": prompt, "n_predict": 1, "temperature": 0.0,
                    "seed": 1337, "n_probs": 20, "cache_prompt": False}
                ).get("completion_probabilities") or []
                probs = {e["token"]: math.exp(e["logprob"])
                         for e in (cp[0].get("top_logprobs", []) if cp else [])}
                yes = sum(v for k, v in probs.items()
                          if _is_label(k, "A", "Yes", False))
                no = sum(v for k, v in probs.items()
                         if _is_label(k, "B", "No", False))
                entry[f"head_{attrib}"] = score
                entry[f"prompted_{attrib}"] = (yes / (yes + no)
                                               if yes + no > 0 else 0.5)
            rows.append(entry)
        if len(rows) % 20 == 0:
            print(f"[{len(rows)}] {time.time()-t0:.0f}s", flush=True)

    def mean(key: str, label: int) -> float | None:
        v = [r[key] for r in rows if r["label"] == label]
        return float(np.mean(v)) if v else None

    out = {"n": len(rows), "rows": rows}
    keys = list(embs.keys())
    states_out = args.states_out or args.out.with_name("attribution-states.npz")
    np.savez_compressed(states_out,
                        keys=np.array(keys),
                        emb=np.stack([embs[k] for k in keys]).astype(np.float16))
    for readout in ("head", "prompted"):
        out[readout] = {}
        for label in (1, 0):
            own = mean(f"{readout}_own", label)
            other = mean(f"{readout}_other", label)
            out[readout][f"label_{label}"] = {
                "own": own, "other": other,
                "delta_own_minus_other": (own - other)
                if own is not None and other is not None else None}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
