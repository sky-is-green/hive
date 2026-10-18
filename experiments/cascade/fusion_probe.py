#!/usr/bin/env python3
"""Zero-extraction fusion probe over existing verifier signals.

Signals per item (all on the r2 authoritative splits/labels):
  - Bonsai-27B frozen J-Bench head          (judge-prompt final hidden state)
  - Bonsai-27B community in-domain head     (natural-error trained, same prompt)
  - Intern-Decision-0.8B ridge heads        (L0.5, L0.75, L1, concat, MLP)
  - raw answer-slot margin p_raw            (zero-training)

No model runs, no extraction: everything is recomputed from committed NPZs.
Fusion is exploratory leave-one-out logistic regression on the held-out test
split; label it accordingly in any report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
R2 = REPO / "experiments/cascade/results/decision-readout-r2-20261017"
JEV = REPO / "experiments/cascade/results/jev-readout-20261006"
PARTNER = REPO / "experiments/cascade/results/partner-bonsai-organic-20261018"


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_r2() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    z = np.load(R2 / "readout-states-intern-decision-0.8b-r2.npz", allow_pickle=False)
    meta = {
        k: z[k].astype(str) if z[k].dtype.kind in "US" else z[k]
        for k in ("item_ids", "labels", "splits", "buckets", "origins", "sources", "p_raw")
    }
    h = np.load(R2 / "readout-head-intern-decision-0.8b-r2.npz", allow_pickle=False)
    p: dict[str, np.ndarray] = {}

    def ridge(prefix: str, x: np.ndarray) -> np.ndarray:
        s = ((x - h[f"{prefix}_mu"]) / h[f"{prefix}_sd"]) @ h[f"{prefix}_w"] \
            + float(h[f"{prefix}_bias"])
        a, b = h[f"{prefix}_platt"]
        return sigmoid(a * s + b)

    p["L0.5"] = ridge("ridge_L0.5", z["L0.5"].astype(np.float64))
    p["L0.75"] = ridge("ridge_L0.75", z["L0.75"].astype(np.float64))
    p["L1"] = ridge("ridge_L1", z["L1"].astype(np.float64))
    xc = np.concatenate([z["L0.5"], z["L0.75"], z["L1"]], axis=1).astype(np.float64)
    p["ridge-concat"] = ridge("ridge_concat", xc)
    hid = np.maximum(((xc - h["mlp_mu"]) / h["mlp_sd"]) @ h["mlp_0.weight"].T
                     + h["mlp_0.bias"], 0.0)
    p["mlp-concat"] = sigmoid((hid @ h["mlp_2.weight"].T + h["mlp_2.bias"]).ravel())
    p["p_raw"] = z["p_raw"].astype(np.float64)
    return meta, p


def load_bonsai() -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    def one(path: Path) -> dict[str, np.ndarray]:
        z = np.load(path, allow_pickle=True)
        return {str(i): z["L1"][n].astype(np.float64)
                for n, i in enumerate(z["item_ids"])}

    states = one(JEV / "probe-states-bonsai27b.npz")
    for k, v in one(PARTNER / "probe-states-bonsai-transfer.npz").items():
        states.setdefault(k, v)
    heads = {
        "frozen": json.loads((JEV / "bonsai-2-27b-decision-head.json").read_text()),
        "community": json.loads((JEV / "bonsai-community-indomain-head.json").read_text()),
    }
    return states, heads


def apply_head(head: dict, vecs: np.ndarray) -> np.ndarray:
    w = np.asarray(head["weights"]["w"], dtype=np.float64)
    mu = np.asarray(head["weights"]["mu"], dtype=np.float64)
    sd = np.asarray(head["weights"]["sd"], dtype=np.float64)
    b = float(head["weights"]["b"])
    a, c = (float(v) for v in head["weights"]["platt"])
    return sigmoid(a * (((vecs - mu) / sd) @ w + b) + c)


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if y.size == 0 or len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, p))


def pareto(y: np.ndarray, p: np.ndarray, budgets: list[int]) -> dict[str, dict]:
    """Max wrongs caught for each cap on corrects rejected, sweeping thresholds."""
    wrongs = y == 0
    corrects = y == 1
    order = np.argsort(p)  # low p = reject first
    out: dict[str, dict] = {}
    for k in budgets:
        best = {"caught": 0, "correct_rejected": 0, "threshold": None}
        # cumulative rejection as threshold rises
        caught = 0
        rejected_correct = 0
        for idx in order:
            if y[idx] == 0:
                caught += 1
            else:
                rejected_correct += 1
            if rejected_correct <= k:
                if caught > best["caught"]:
                    best = {"caught": caught, "correct_rejected": rejected_correct,
                            "threshold": float(p[idx])}
            else:
                break
        out[str(k)] = best
    return out


def loo_fusion(F: np.ndarray, y: np.ndarray, C: float = 1.0) -> np.ndarray:
    n = len(y)
    pred = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        mu, sd = F[m].mean(0), F[m].std(0) + 1e-9
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit((F[m] - mu) / sd, y[m])
        pred[i] = clf.predict_proba(((F[i] - mu) / sd)[None, :])[0, 1]
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    meta, p08 = load_r2()
    states, heads = load_bonsai()
    ids = [str(i) for i in meta["item_ids"]]
    labels = meta["labels"].astype(int)

    p_bonsai: dict[str, np.ndarray] = {}
    id_arr = np.asarray(ids)
    covered = np.array([i in states for i in ids])
    for name, head in heads.items():
        p = np.full(len(ids), np.nan)
        p[covered] = apply_head(head, np.stack([states[i] for i in id_arr[covered]]))
        p_bonsai[name] = p

    # sanity vs r2 comparison JSON for the 9 organic wrongs
    r2 = json.loads((R2 / "r2-readout-comparison.json").read_text())
    for head_name, key in (("L0.5", "L0.5"), ("L0.75", "L0.75"), ("L1", "L1"),
                           ("ridge-concat", "ridge-concat"), ("mlp-concat", "mlp-concat")):
        for item, expect in r2["p"][key].items():
            got = float(p08[head_name][ids.index(item)])
            if abs(got - expect) > 2e-3:
                print(f"WARN r2 mismatch {head_name} {item}: {got:.4f} vs {expect:.4f}")

    splits = meta["splits"]
    sources = meta["sources"]
    buckets = meta["buckets"]
    test = splits == "test"
    organic = test & (sources == "harvest")
    bench_test = test & (sources == "bench")
    qa_harvest = (buckets == "qa") & (sources == "harvest")

    slices = {
        "test-182": test,
        "bench-116": bench_test,
        "organic-66": organic,
        "qa-harvest-75": qa_harvest,
    }

    signals = {**p08, **{f"bonsai-{k}": v for k, v in p_bonsai.items()}}
    base_rows: dict[str, dict] = {}
    for name, pv in signals.items():
        row: dict[str, dict] = {}
        needs_bonsai = name.startswith("bonsai-")
        for sname, mask in slices.items():
            m = mask & covered if needs_bonsai else mask
            if needs_bonsai and (mask & ~covered).any():
                raise SystemExit(f"{sname}: Bonsai states missing for "
                                 f"{int((mask & ~covered).sum())} items")
            y, p = labels[m], pv[m]
            row[sname] = {
                "n": int(m.sum()),
                "wrong": int((y == 0).sum()),
                "auc": auc(y, p),
                "pareto_organic": pareto(y, p, [0, 3, 5, 9, 14]) if sname == "organic-66" else None,
            }
            if sname == "organic-66":
                row[sname]["catch_at_0.5"] = int(((y == 0) & (p < 0.5)).sum())
                row[sname]["correct_rejected_at_0.5"] = int(((y == 1) & (p < 0.5)).sum())
        base_rows[name] = row

    # fusion feature sets (logits), LOO on the held-out test split (Bonsai-covered)
    test_c = test & covered
    y = labels[test_c]
    F_all = {
        "community": logit(p_bonsai["community"][test_c]),
        "frozen": logit(p_bonsai["frozen"][test_c]),
        "L0.5": logit(p08["L0.5"][test_c]),
        "L0.75": logit(p08["L0.75"][test_c]),
        "L1": logit(p08["L1"][test_c]),
        "ridge-concat": logit(p08["ridge-concat"][test_c]),
        "mlp-concat": logit(p08["mlp-concat"][test_c]),
        "p_raw": logit(p08["p_raw"][test_c]),
    }
    combos = {
        "community": ["community"],
        "L0.5": ["L0.5"],
        "mlp": ["mlp-concat"],
        "L0.5+mlp": ["L0.5", "mlp-concat"],
        "community+L0.5": ["community", "L0.5"],
        "community+mlp": ["community", "mlp-concat"],
        "community+L0.5+mlp": ["community", "L0.5", "mlp-concat"],
        "community+L0.5+mlp+p_raw": ["community", "L0.5", "mlp-concat", "p_raw"],
        "all": list(F_all),
    }
    org_mask = organic[test_c]
    fusion_rows: dict[str, dict] = {}
    for name, feats in combos.items():
        F = np.stack([F_all[f] for f in feats], axis=1)
        pv = loo_fusion(F, y)
        fusion_rows[name] = {
            "features": feats,
            "auc_test": auc(y, pv),
            "auc_organic": auc(y[org_mask], pv[org_mask]),
            "pareto_organic": pareto(y[org_mask], pv[org_mask], [0, 3, 5, 9, 14]),
        }

    out = {
        "signals": base_rows,
        "fusion_loo_test182": fusion_rows,
        "slices": {k: {"n": int(v.sum()),
                       "wrong": int((labels[v] == 0).sum()),
                       "correct": int((labels[v] == 1).sum())}
                   for k, v in slices.items()},
        "coverage": {"r2_items": len(ids),
                     "bonsai_states": len(states),
                     "test_with_bonsai": int(test_c.sum()),
                     "test_total": int(test.sum())},
        "note": "exploratory; fusion trained by LOO on the held-out test itself",
    }
    (args.outdir / "fusion-probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v["organic-66"]["pareto_organic"] for k, v in base_rows.items()
                      if k in ("bonsai-community", "bonsai-frozen", "L0.5", "mlp-concat")}, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "features"}
                      for k, v in fusion_rows.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
