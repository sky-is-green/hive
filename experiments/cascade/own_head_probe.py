#!/usr/bin/env python3
"""The generator's own head on the frozen J-Bench, offline.

The 176B's judge states for the 378 frozen J-Bench items were pulled through the Strata engine
during the decision-head port (`scion/decision-heads-work/overnight/probe-states-main176b.npz`).
This script applies the 176B's own heads to those states and compares them against the partner
signals on the same 116-item test split: Ternary-Bonsai-2-27B's frozen J-Bench head (the committed
baseline), the Intern-Decision-0.8B ridge/MLP heads, and the raw answer-slot margin.  Nothing here
touches the engine or the serve; every array is read from disk.

    .venv/bin/python experiments/cascade/own_head_probe.py --out <path.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
SCION = WORK / "scion"
STATES_176B = SCION / "decision-heads-work/overnight/probe-states-main176b.npz"
HEAD_176B_JBENCH = SCION / "decision-heads/scion-176b-a6b-decision-head.json"
HEAD_176B_COMMUNITY = SCION / "decision-heads-work/overnight/scion176b-community-indomain-head.json"
STATES_BONSAI = REPO / "experiments/cascade/results/jev-readout-20261006/probe-states-bonsai27b.npz"
HEAD_BONSAI = REPO / "experiments/cascade/results/jev-readout-20261006/bonsai-2-27b-decision-head.json"
R2 = REPO / "experiments/cascade/results/decision-readout-r2-20261017"


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def apply_head(head: dict, vecs: np.ndarray) -> np.ndarray:
    w = np.asarray(head["weights"]["w"], dtype=np.float64)
    mu = np.asarray(head["weights"]["mu"], dtype=np.float64)
    sd = np.asarray(head["weights"]["sd"], dtype=np.float64)
    b = float(head["weights"]["b"])
    a, c = (float(v) for v in head["weights"].get("platt", [1.0, 0.0]))
    return sigmoid(a * (((vecs - mu) / sd) @ w + b) + c)


def stats(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = y.astype(int)
    return {
        "n": int(y.size),
        "wrong": int((y == 0).sum()),
        "auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None,
        "at_0.5": {
            "threshold": 0.5,
            "false_accepts": int(((y == 0) & (p >= 0.5)).sum()),
            "false_rejects": int(((y == 1) & (p < 0.5)).sum()),
        },
        "at_stored": None if threshold == 0.5 else {
            "threshold": float(threshold),
            "false_accepts": int(((y == 0) & (p >= threshold)).sum()),
            "false_rejects": int(((y == 1) & (p < threshold)).sum()),
        },
    }


def load_zeroehead() -> dict[str, np.ndarray]:
    z = np.load(R2 / "readout-states-intern-decision-0.8b-r2.npz", allow_pickle=False)
    h = np.load(R2 / "readout-head-intern-decision-0.8b-r2.npz", allow_pickle=False)

    def ridge(prefix: str, x: np.ndarray) -> np.ndarray:
        s = ((x.astype(np.float64) - h[f"{prefix}_mu"]) / h[f"{prefix}_sd"]) @ h[f"{prefix}_w"] \
            + float(h[f"{prefix}_bias"])
        a, b = h[f"{prefix}_platt"]
        return sigmoid(a * s + b)

    xc = np.concatenate([z["L0.5"], z["L0.75"], z["L1"]], axis=1)
    hid = np.maximum(((xc.astype(np.float64) - h["mlp_mu"]) / h["mlp_sd"]) @ h["mlp_0.weight"].T
                     + h["mlp_0.bias"], 0.0)
    return {
        "item_ids": np.array([str(i) for i in z["item_ids"]]),
        "labels": z["labels"].astype(int),
        "splits": np.array([str(s) for s in z["splits"]]),
        "sources": np.array([str(s) for s in z["sources"]]),
        "L0.5": ridge("ridge_L0.5", z["L0.5"]),
        "MLP": sigmoid((hid @ h["mlp_2.weight"].T + h["mlp_2.bias"]).ravel()),
        "p_raw": z["p_raw"].astype(np.float64),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    z = np.load(STATES_176B, allow_pickle=True)
    ids = [str(i) for i in z["item_ids"]]
    x176 = z["L1"].astype(np.float64)
    labels = z["labels"].astype(int)
    splits = np.array([str(s) for s in z["splits"]])
    test = splits == "test"

    signals: dict[str, dict] = {}

    head = json.loads(HEAD_176B_JBENCH.read_text())
    thr = float((head.get("test") or {}).get("threshold", 0.5))
    signals["176b-own-jbench"] = {"path": str(HEAD_176B_JBENCH), "p": apply_head(head, x176),
                                  "stored_threshold": thr}
    head = json.loads(HEAD_176B_COMMUNITY.read_text())
    thr = float((head.get("test") or {}).get("threshold", 0.5))
    signals["176b-own-community"] = {"path": str(HEAD_176B_COMMUNITY), "p": apply_head(head, x176),
                                     "stored_threshold": thr}

    zb = np.load(STATES_BONSAI, allow_pickle=True)
    idsb = [str(i) for i in zb["item_ids"]]
    xb = zb["L1"].astype(np.float64)
    if idsb != ids:
        raise SystemExit("bench item ids differ between the 176B and Bonsai states")
    head = json.loads(HEAD_BONSAI.read_text())
    thr = float((head.get("test") or {}).get("threshold", 0.5))
    signals["bonsai-frozen-partner"] = {"path": str(HEAD_BONSAI), "p": apply_head(head, xb),
                                        "stored_threshold": thr}

    z8 = load_zeroehead()
    pos = {str(i): n for n, i in enumerate(z8["item_ids"])}
    order = np.array([pos[i] for i in ids])
    if not np.array_equal(z8["labels"][order], labels):
        raise SystemExit("bench labels differ between the 176B and 0.8B states")
    for name in ("L0.5", "MLP", "p_raw"):
        signals[f"080-{name}"] = {"path": str(R2), "p": z8[name][order], "stored_threshold": 0.5}

    out: dict = {
        "states_176b": str(STATES_176B),
        "labels_match_across_sources": True,
        "slices": {},
        "signals": {},
    }
    for sname, mask in (("bench-train-262", ~test), ("bench-test-116", test)):
        out["slices"][sname] = {"n": int(mask.sum()), "wrong": int((labels[mask] == 0).sum())}
    for name, sig in signals.items():
        entry = {"path": sig["path"], "stored_threshold": sig["stored_threshold"]}
        for sname, mask in (("bench-train-262", ~test), ("bench-test-116", test)):
            entry[sname] = stats(labels[mask], sig["p"][mask], sig["stored_threshold"])
        out["signals"][name] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    print(f"{'signal':22s} {'train AUC':>9s} {'test AUC':>8s} {'test FA/FX@0.5':>14s} {'FA/FX@stored':>12s}")
    for name, entry in out["signals"].items():
        tr, te = entry["bench-train-262"], entry["bench-test-116"]
        fa, fx = te["at_0.5"]["false_accepts"], te["at_0.5"]["false_rejects"]
        store = te["at_stored"]
        s = f"{store['false_accepts']}/{store['false_rejects']}" if store else "-"
        print(f"{name:22s} {tr['auc']:9.4f} {te['auc']:8.4f} {fa:>7d}/{fx:<6d} {s:>12s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
