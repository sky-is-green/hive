#!/usr/bin/env python3
"""Marginal catch: what a learned gate adds on top of the deterministic detectors.

The blind-spot question, answered offline on the stored harvest-organic-66
candidates (no serve, no GPU): recompute the deterministic catches
(consistency + code_exec) on the stored drafts, then ask of each learned
signal what it adds **beyond** those detectors at a false-flag budget of
0..3 extra corrects: wrongs caught that the detectors missed, and the extra
flags paid for them.  This is the number that decides whether a learned gate
is worth wiring as a third trigger.

    .venv/bin/python experiments/cascade/blindspot_marginal.py \
        --out experiments/cascade/results/scion176b-self-organic-20261018/marginal-catch.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = REPO.parent
sys.path.insert(0, str(HERE))

import fusion_probe as fp  # noqa: E402
from cascade import detect, runtime_spec  # noqa: E402 - the gold-free runtime guard + detectors
from eval_self_head import HEAD_COMMUNITY, HEAD_JBENCH, WINDOW, head_probe, load_states  # noqa: E402

HARVEST = REPO / "experiments/cascade/results/decision-harvest-r2-20261017/harvest-bench-r2.jsonl"
SPECS = REPO / "experiments/cascade/tasks-bench.json"
BUDGETS = (0, 1, 2, 3, 5)


def marginal(y: np.ndarray, p: np.ndarray, already: np.ndarray, budgets) -> dict:
    """Best catch of wrongs not already caught, per cap on newly-flagged corrects.

    ``already`` marks items the detectors flag; the sweep charges only corrects
    that the learned signal flags and the detectors did not.
    """
    order = np.argsort(p)  # low p = flag first
    out: dict[str, dict] = {}
    for k in budgets:
        best = {"marginal_caught": 0, "extra_correct_flags": 0, "threshold": None,
                "caught_ids": [], "innocent_ids": []}
        caught: list[str] = []
        extra: list[str] = []
        for idx in order:
            if y[idx] == 0:
                if not already[idx]:
                    caught.append(idx)
            elif not already[idx]:
                extra.append(idx)
            if len(extra) <= k:
                if len(caught) > best["marginal_caught"]:
                    best = {"marginal_caught": len(caught), "extra_correct_flags": len(extra),
                            "threshold": float(p[idx]), "caught_ids": list(caught),
                            "innocent_ids": list(extra)}
            else:
                break
        out[str(k)] = {**best, "caught_ids": [int(i) for i in best["caught_ids"]],
                       "innocent_ids": [int(i) for i in best["innocent_ids"]]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    meta, p08 = fp.load_r2()
    ids = [str(i) for i in meta["item_ids"]]
    labels = meta["labels"].astype(int)
    splits, sources = meta["splits"], meta["sources"]

    rows = [json.loads(l) for l in HARVEST.read_text().splitlines() if l.strip()]
    by_id = {str(r["item_id"]): r for r in rows}
    specs = {t["id"]: t for t in json.loads(SPECS.read_text())["tasks"]}

    organic = [i for i in ids if sources[ids.index(i)] == "harvest" and splits[ids.index(i)] == "test"]
    print(f"organic-66: {len(organic)} items, {(labels[[ids.index(i) for i in organic]] == 0).sum()} wrong")
    missing = [i for i in organic if i not in by_id]
    if missing:
        raise SystemExit(f"no candidate for {missing}")
    label_source = {i: int(by_id[i]["label"]) for i in organic}
    mismatch = [i for i in organic
                if "label" in by_id[i] and label_source[i] != int(labels[ids.index(i)])]
    if mismatch:
        print(f"WARNING: labels disagree on {mismatch}")

    # --- deterministic detectors on the stored drafts (gold-free) -----------
    caught: dict[str, list[str]] = {}
    for i in organic:
        row = by_id[i]
        spec = specs.get(str(row.get("task_id")))
        if spec is None:
            raise SystemExit(f"no spec for {i} (task_id={row.get('task_id')})")
        findings = detect(str(row["candidate"]), runtime_spec(spec))
        caught[i] = [f["detector"] for f in findings]
    det_caught = np.array([bool(caught[i]) for i in organic])
    y = np.array([int(labels[ids.index(i)]) for i in organic])
    print(f"detectors: caught {int(((y == 0) & det_caught).sum())} of {int((y == 0).sum())} wrongs, "
          f"false-flagged {int(((y == 1) & det_caught).sum())} corrects")
    from collections import Counter
    print("  by detector:", Counter(d for i in organic for d in caught[i]))

    # --- learned signals ----------------------------------------------------
    states, _, _ = load_states(WINDOW)
    probs = {
        "176b-own-community": head_probe("c", json.loads(HEAD_COMMUNITY.read_text()), states, organic),
        "176b-own-jbench": head_probe("j", json.loads(HEAD_JBENCH.read_text()), states, organic),
        "0.8B-L0.5": {i: float(p08["L0.5"][ids.index(i)]) for i in organic},
        "0.8B-mlp-concat": {i: float(p08["mlp-concat"][ids.index(i)]) for i in organic},
    }

    out = {"slice": "harvest organic-66 (stored candidates)", "n": len(organic),
           "wrong": int((y == 0).sum()),
           "detectors": {"caught_wrongs": int(((y == 0) & det_caught).sum()),
                         "false_flags": int(((y == 1) & det_caught).sum()),
                         "per_item": {i: caught[i] for i in organic}},
           "signals": {}}

    wrong_ids = [i for i, v in zip(organic, y) if v == 0]
    missed = [i for i in wrong_ids if not caught[i]]
    print(f"detector-missed wrongs: {len(missed)}: {missed}")

    for name, pr in probs.items():
        p = np.array([pr[i] for i in organic])
        total = fp.pareto(y, p, list(BUDGETS))
        marg = marginal(y, p, det_caught, BUDGETS)
        out["signals"][name] = {
            "auc": fp.auc(y, p),
            "total_pareto": total,
            "marginal": {k: {**v, "caught": [organic[i] for i in v["caught_ids"]],
                             "extra_flags": [organic[i] for i in v["innocent_ids"]]}
                         for k, v in marg.items()},
        }
        print(f"\n{name}: auc {fp.auc(y, p):.3f}")
        print("  total (all wrongs)   " + " | ".join(
            f"cap{k}: {v['caught']}/{v['correct_rejected']}" for k, v in total.items()))
        print("  marginal over dets   " + " | ".join(
            f"cap{k}: +{v['marginal_caught']} (extra flags {v['extra_correct_flags']})"
            for k, v in sorted(marg.items(), key=lambda kv: int(kv[0]))))
        for k in ("0", "1", "2", "3"):
            v = marg[k]
            if v["marginal_caught"]:
                print(f"    cap{k} catches: {[organic[i] for i in v['caught_ids']]} "
                      f"paying {[organic[i] for i in v['innocent_ids']]}")

    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
