#!/usr/bin/env python3
"""Gate replay (M3 of LEARNED-GATE-PLAN.md): the production scorer over committed artifacts.

No serving changes, no runner re-run: one model load, then three scorings
that reuse committed data —

1. the harvest-66 reproduction (the study body): every stored organic draft
   through ``harness.cascade.learned_gate`` (the path-B scorer, frozen heads),
   its per-item p compared with the r2 extraction's fitted p, and the 0.5-cut
   catch/flag counts compared with ``MARGINAL.md``'s committed table;
2. arm A (``results/runner-cascade-sampled-20261018``) and arm B (``...
   -nothink-20261018``): every served draft through the gold-free
   deterministic detectors, and — blind-spot-only, exactly the runner's rule
   — the silent drafts through the gate.  This is the go/no-go data: the
   learned gate on FRESH drafts at the frozen 0.5 cut, counts of marginal
   catches (silent wrongs the detectors missed and the gate fires on) and
   extra flags (silent corrects it would fire on).

The labels are the committed row truth (``scion_ok`` — eval-only use of the
checkers; the scored objects stay gold-free: question + candidate).

    .venv/bin/python experiments/cascade/gate_replay.py \
        --out experiments/cascade/results/gate-replay-20261018
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import fusion_probe as fp  # noqa: E402
from cascade import detect, runtime_spec  # noqa: E402 - the gold-free runtime guard + detectors

from harness.cascade.learned_gate import (  # noqa: E402
    DEFAULT_HEAD,
    LearnedGate,
    ReadoutHead,
)
THRESHOLD = 0.5
#: the recall dial runs beside the default head — one extraction, two heads
HEAD_SEQ = ("L0.75", "L0.5")

ARMS = {
    "armA-sampled": REPO / "experiments/cascade/results/runner-cascade-sampled-20261018",
    "armB-nothink": REPO / "experiments/cascade/results/runner-cascade-sampled-nothink-20261018",
}
HARVEST = REPO / "experiments/cascade/results/decision-harvest-r2-20261017/harvest-bench-r2.jsonl"
SPECS = REPO / "experiments/cascade/tasks-bench.json"


def arm_replay(g: LearnedGate, arm_dir: Path, report_path: Path,
               heads: dict[str, ReadoutHead]) -> dict:
    """Blind-spot-only replay over one committed arm report."""
    report = json.loads(report_path.read_text())
    tasks = {t["id"]: t for t in
             json.loads((arm_dir / "tasks-organic66.json").read_text())["tasks"]}
    per_head = {name: {"caught": {}, "false_flags": {}, "fires": 0, "scored": 0}
                for name in heads}
    silent_rows, detector_rows, truncated_rows = [], 0, []
    t0 = time.perf_counter()
    for row in report["records"]:
        if row.get("gated"):  # router sent it to the API: no draft to score
            continue
        rs = runtime_spec(tasks[row["id"]])
        if row.get("truncated"):
            # the runner's truncation rule: a still-cut draft is rejected by
            # the harness ceiling and never reaches the gate (no answer to
            # score) — the replay honours the same rule.
            truncated_rows.append(row["id"])
        if detect(str(row["scion_answer"] or ""), rs):
            detector_rows += 1
            continue
        if row.get("truncated"):
            continue
        silent_rows.append(row)
        task = tasks[row["id"]]
        feats = dict(g.states(str(task["prompt"]), str(row["scion_answer"] or "")))
        wrong = row["scion_ok"] is False
        for name, head in heads.items():
            p = head.score(feats)
            entry = per_head[name]
            entry["scored"] += 1
            if p < THRESHOLD:
                entry["fires"] += 1
                if wrong:
                    entry["caught"][row["id"]] = round(p, 6)
                else:
                    entry["false_flags"][row["id"]] = round(p, 6)
    scored_s = time.perf_counter() - t0
    by_head = {}
    for name, entry in per_head.items():
        by_head[name] = {
            "threshold": THRESHOLD,
            "silent": len(silent_rows),
            "fires": entry["fires"],
            "marginal_catch": len(entry["caught"]),
            "marginal_catch_ids": list(entry["caught"]),
            "extra_flags": len(entry["false_flags"]),
            "extra_flag_ids": list(entry["false_flags"]),
            "silent_wrong_ids": [row["id"] for row in silent_rows
                                 if row["scion_ok"] is False],
            "silent_correct_ids": [row["id"] for row in silent_rows
                                   if row["scion_ok"] is not False],
            "fires_p": {**entry["caught"], **entry["false_flags"]},
        }
    return {
        "rows": len(report["records"]),
        "detector_flagged": detector_rows,
        "truncated_no_score": truncated_rows,
        "silent": len(silent_rows),
        "silent_scoring_s": round(scored_s, 1),
        "scored_per_silent_ms": round(scored_s * 1000.0 / max(len(silent_rows), 1), 1),
        "heads": by_head,
    }


def harvest_replay(g: LearnedGate, heads: dict[str, ReadoutHead]) -> dict:
    """The stored organic-66 through the production scorer, both heads."""
    meta, fitted = fp.load_r2()
    ids = [str(i) for i in meta["item_ids"]]
    want_mask = (meta["sources"].astype(str) == "harvest") \
        & (meta["splits"].astype(str) == "test")
    organic = [ids[i] for i in range(len(ids)) if want_mask[i]]
    rows = {r["item_id"]: r for r in
            (json.loads(l) for l in HARVEST.read_text().splitlines() if l.strip())}
    specs = {t["id"]: t for t in json.loads(SPECS.read_text())["tasks"]}
    missing = [i for i in organic if i not in rows]
    if missing:
        raise SystemExit(f"no candidate for {missing}")

    labels, det = {}, {}
    max_dp = {name: 0.0 for name in heads}
    scores: dict[str, dict[str, float]] = {}
    t0 = time.perf_counter()
    for item in organic:
        row = rows[item]
        task = specs[str(row.get("task_id"))]
        rs = runtime_spec(task)
        feats = dict(g.states(str(row["prompt"]), str(row["candidate"])))
        scores[item] = {}
        for name, head in heads.items():
            p = head.score(feats)
            scores[item][name] = p
            store = fitted[name][ids.index(item)]
            max_dp[name] = max(max_dp[name], abs(p - float(store)))
        det[item] = [f["detector"] for f in
                     detect(str(row["candidate"]), rs)]
        labels[item] = int(row["label"])
    scored_s = time.perf_counter() - t0

    y = np.array([labels[i] for i in organic])
    out = {
        "n": len(organic),
        "scoring_s": round(scored_s, 1),
        "max_abs_dp_vs_stored": {k: round(v, 9) for k, v in max_dp.items()},
        "detectors": {
            "caught_wrong": sorted(i for i in organic
                                   if y[organic.index(i)] == 0 and det[i]),
            "false_flags": sorted(i for i in organic
                                  if y[organic.index(i)] == 1 and det[i]),
        },
        "heads": {},
    }
    detected = np.array([bool(det[i]) for i in organic])
    for name in heads:
        p = np.array([scores[i][name] for i in organic])
        miss = [(i, round(float(scores[i][name]), 6))
                for i, yi, d in zip(organic, y, detected)
                if yi == 0 and not d and scores[i][name] < THRESHOLD]
        flags = [(i, round(float(scores[i][name]), 6))
                 for i, yi, d in zip(organic, y, detected)
                 if yi == 1 and not d and scores[i][name] < THRESHOLD]
        out["heads"][name] = {
            "threshold": THRESHOLD,
            "marginal_at_cut": {"caught": [i for i, _ in miss],
                                "paid": [i for i, _ in flags]},
            "fired": miss + flags,
            "auc": round(float(fp.auc(y, p)), 4),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[replay] loading the path-B backbone ({' '.join(HEAD_SEQ)} heads) ...",
          flush=True)
    g = LearnedGate()  # head object inside; states() drives the backbone
    heads = {name: ReadoutHead(name) for name in HEAD_SEQ}

    print("[replay] harvest-66 reproduction ...", flush=True)
    harvest = harvest_replay(g, heads)
    print(json.dumps({"n": harvest["n"],
                      "max|dp|": harvest["max_abs_dp_vs_stored"]}), flush=True)

    arms = {}
    for name, arm_dir in ARMS.items():
        print(f"[replay] {name} ...", flush=True)
        arms[name] = arm_replay(g, arm_dir, arm_dir / "run/report.json", heads)
        b = arms[name]["heads"][DEFAULT_HEAD]
        print(f"  silent {b['silent']} | fires {b['fires']} "
              f"| catches {b['marginal_catch']} | extra flags {b['extra_flags']}",
              flush=True)

    result = {
        "threshold": THRESHOLD,
        "heads": list(HEAD_SEQ),
        "default_head": DEFAULT_HEAD,
        "harvest": harvest,
        "arms": arms,
    }
    (args.out / "replay.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"[replay] wrote {args.out / 'replay.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
