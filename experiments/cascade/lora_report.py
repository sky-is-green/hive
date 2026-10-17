#!/usr/bin/env python
"""Report a lora_verdict run against the frozen-head references.

Applies the selection rule and pass/marginal/fail rules fixed in
results/lora-verdict-20261017/PRE-REGISTRATION.md before the first eval:

- epoch to report: best organic-test AUC, unless its bench correct count is
  more than 2 below the best-bench epoch's (then the best-bench epoch);
- pass: organic AUC >= 0.84 and bench AUC >= 0.95, or caught >= 6/9 at FA <= 4;
- marginal: organic AUC 0.80-0.84 at bench >= 0.95, or 5/9 caught at FA <= 3;
- fail: bench AUC < 0.92 or caught <= 4/9.

Usage:
    .venv/bin/python experiments/cascade/lora_report.py \
        --lora experiments/cascade/results/lora-verdict-20261017/lora-intern-decision-0.8b-lora-r2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
R2 = REPO_ROOT / "experiments/cascade/results/decision-readout-r2-20261017"
DEFAULT_LORA = REPO_ROOT / "experiments/cascade/results/lora-verdict-20261017/lora-intern-decision-0.8b-lora-r2.json"
DEFAULT_CORPUS = REPO_ROOT / "experiments/cascade/results/decision-harvest-r2-20261017/harvest-bench-r2.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = ap.parse_args()

    run = json.loads(args.lora.read_text())
    evals = run["evals"]
    corpus = load_jsonl(args.corpus)
    org_test = [i for i in corpus if i["split"] == "test"
                and i["origin"].startswith("harvested:")]
    org_ids = [i["item_id"] for i in org_test]
    wrong_ids = [i["item_id"] for i in org_test if i["label"] == 0]

    epochs = [t for t in evals if t.startswith("epoch")]
    best_bench = max(epochs, key=lambda t: (evals[t]["judge-bench"]["auc"],
                                            evals[t]["judge-bench"]["acc"]))
    best_org = max(epochs, key=lambda t: evals[t]["organic-test"]["auc"])
    if (evals[best_org]["judge-bench"]["acc"]
            >= evals[best_bench]["judge-bench"]["acc"] - 2):
        chosen = best_org
    else:
        chosen = best_bench

    ref = json.loads((R2 / "readout-intern-decision-0.8b-r2.json").read_text())
    ref_rows = {}
    for name in ("mlp-concat", "L0.5", "L0.75"):
        b = ref["readouts"][name]["by_source"]["bench"]["at_0.5"]
        h = ref["readouts"][name]["by_source"]["harvest"]["at_0.5"]
        ref_rows[name] = {
            "bench_auc": round(ref["readouts"][name]["by_source"]["bench"]["auc"], 4),
            "bench_acc": b["correct"], "bench_fa": b["false_accepts"],
            "bench_fr": b["false_rejects"],
            "org_auc": round(ref["readouts"][name]["by_source"]["harvest"]["auc"], 4),
            "org_fa": h["false_accepts"], "org_fr": h["false_rejects"],
            "caught": 9 - h["false_accepts"],
        }
    raw_bench = ref["baseline"]["by_source"]["bench"]
    raw_org = ref["baseline"]["by_source"]["harvest"]
    raw_row = {
        "bench_auc": round(raw_bench["auc"], 4),
        "bench_acc": raw_bench["at_0.5"]["correct"],
        "bench_fa": raw_bench["at_0.5"]["false_accepts"],
        "bench_fr": raw_bench["at_0.5"]["false_rejects"],
        "org_auc": round(raw_org["auc"], 4),
        "org_fa": raw_org["at_0.5"]["false_accepts"],
        "org_fr": raw_org["at_0.5"]["false_rejects"],
        "caught": 9 - raw_org["at_0.5"]["false_accepts"],
    }

    print(f"run {run['name']} — train {run['train_items']} items, "
          f"{run['seconds']} s")
    print(f"\n{'epoch':8s} {'bench AUC':>9s} {'bench acc':>9s} {'FA/FR':>6s} "
          f"{'org AUC':>8s} {'org acc':>8s} {'FA/FR':>6s} {'caught':>6s} "
          f"{'ext FR':>6s}")
    for tag in epochs + ["final"]:
        if tag not in evals:
            continue
        b, o = evals[tag]["judge-bench"], evals[tag]["organic-test"]
        e = evals[tag].get("harvest-testmath", {})
        print(f"{tag:8s} {b['auc']:9.3f} {b['acc']:5d}/116 "
              f"{b['fa']:2d}/{b['fr']:2d} {o['auc']:8.3f} {o['acc']:5d}/66 "
              f"{o['fa']:2d}/{o['fr']:2d} {o['wrongs_caught']:4d}/9 "
              f"{e.get('fr', 0):5d}/4")

    c = evals[chosen]
    b, o = c["judge-bench"], c["organic-test"]
    org_auc, bench_auc = o["auc"], b["auc"]
    caught, fa = o["wrongs_caught"], o["fa"]
    if (org_auc >= 0.84 and bench_auc >= 0.95) or (caught >= 6 and fa <= 4):
        verdict = "PASS — adapted-backbone refit + gate trial"
    elif (org_auc >= 0.80 and bench_auc >= 0.95) or (caught == 5 and fa <= 3):
        verdict = "MARGINAL — one more data round, no gate trial"
    elif bench_auc < 0.92 or caught <= 4:
        verdict = "FAIL — stop representational work, choose operating point"
    else:
        verdict = "BETWEEN — read the item table before deciding"
    print(f"\nselected epoch: {chosen} (best org {best_org}, best bench {best_bench})")
    print(f"verdict: {verdict}")

    print("\nreferences (frozen backbone, commit 7bfa7fb):")
    print(f"{'readout':12s} {'bench AUC':>9s} {'bench acc':>9s} "
          f"{'org AUC':>8s} {'caught':>6s}")
    for name, r in [("raw", raw_row)] + list(ref_rows.items()):
        print(f"{name:12s} {r['bench_auc']:9.3f} {r['bench_acc']:5d}/116 "
              f"{r['org_auc']:8.3f} {r['caught']:4d}/9")

    print(f"\norganic test wrongs under {chosen} (C = rejected):")
    p = o["p"]
    for iid in wrong_ids:
        pi = p[org_ids.index(iid)]
        print(f"  {iid:24s} {pi:.4f} {'C' if pi < 0.5 else ''}")

    out = {"run": run["name"], "selected_epoch": chosen,
           "best_org_epoch": best_org, "best_bench_epoch": best_bench,
           "verdict": verdict, "references": {"raw": raw_row, **ref_rows},
           "selected": c,
           "epochs": {t: {"judge-bench": evals[t]["judge-bench"],
                          "organic-test": evals[t]["organic-test"],
                          "harvest-testmath": evals[t].get("harvest-testmath")}
                      for t in epochs}}
    dest = args.lora.parent / "lora-comparison.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
