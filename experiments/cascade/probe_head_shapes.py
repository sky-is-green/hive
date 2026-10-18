#!/usr/bin/env python3
"""Is the 176B's organic readout underfit?  Ridge vs small encoders on the harvest judge states.

The Window B head is a plain ridge+Platt on the 2560-d final-token judge state, and it needed
lambda=10000 on 871 items.  On the 0.8B the equivalent MLP beat the ridge on organic by a wide
margin (0.844 vs 0.688 for ridge-concat), so the natural question is whether the 176B's own states
are also underfit by a linear readout.

Selection is on the 871 harvest TRAIN items under GroupKFold(5) by task_id — 470 wrongs, which is
the only properly powered measurement we have.  The 66 organic test items carry 9 wrongs, so their
AUC has a standard error around 0.08: they confirm a config chosen on CV, they cannot choose one.

    .venv/bin/python experiments/cascade/probe_head_shapes.py \
        --out experiments/cascade/results/scion176b-harvest-20261018/head-shapes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import fusion_probe as fp  # noqa: E402
from decision_readout import _mlp_fit, _mlp_predict, platt, ridge_fit, ridge_score  # noqa: E402

WINDOW_B = REPO / "experiments/cascade/results/scion176b-harvest-20261018"
R2 = REPO / "experiments/cascade/results/decision-readout-r2-20261017"
STATES = WINDOW_B / "probe-states-scion176b-harvest.npz"
LAMBDA_GRID = (1.0, 100.0, 1000.0, 10000.0, 100000.0)


def fit_predict(kind: str, param: float, Xf, yf, Xu):
    """Fit one head on (Xf, yf) and return calibrated probabilities on Xu."""
    if kind == "ridge":
        model = ridge_fit(Xf, yf, lam=param)
        a, b = platt(ridge_score(Xf, model), yf)
        s = ridge_score(Xu, model)
        return 1.0 / (1.0 + np.exp(-(a * s + b)))
    # mlp: hidden width = param; _mlp_predict already standardises and applies the sigmoid
    net, stats = _mlp_fit(Xf, yf, seed=1337, epochs=int(MLPE), hidden=int(param))
    return _mlp_predict(Xu, net, stats)


MLPE = 400


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    z = np.load(STATES, allow_pickle=True)
    ids = [str(i) for i in z["item_ids"]]
    X = z["L1"].astype(np.float64)
    y = z["labels"].astype(int).astype(np.float64)
    splits = z["splits"].astype(str)
    task = np.array([i.split(":")[0] for i in ids])
    tr, te = np.where(splits == "train")[0], np.where(splits == "test")[0]
    print(f"harvest states {len(ids)}  train {len(tr)} ({int((y[tr]==0).sum())} wrong)  "
          f"test {len(te)} ({int((y[te]==0).sum())} wrong)")

    configs = ([("ridge", lam) for lam in LAMBDA_GRID]
               + [("mlp", 64), ("mlp", 128), ("mlp", 256), ("mlp", 512)])
    folds = list(GroupKFold(n_splits=5).split(tr, y[tr], task[tr]))
    rows: dict[str, dict] = {}
    for kind, param in configs:
        name = f"{kind}-{param:g}"
        aucs, nll = [], []
        for fit_l, hold_l in folds:
            f, h = tr[fit_l], tr[hold_l]
            p = fit_predict(kind, param, X[f], y[f], X[h])
            aucs.append(fp.auc(y[h], p))
            nll.append(float(-np.mean(y[h] * np.log(p + 1e-9) + (1 - y[h]) * np.log(1 - p + 1e-9))))
        aucs = np.array(aucs, dtype=float)
        rows[name] = {"kind": kind, "param": float(param),
                      "cv_auc_mean": float(aucs.mean()),
                      "cv_auc_std": float(aucs.std(ddof=1)),
                      "cv_auc_sem": float(aucs.std(ddof=1) / np.sqrt(len(aucs))),
                      "cv_nll_mean": float(np.mean(nll))}
        print(f"  {name:12s} grouped-CV AUC {aucs.mean():.4f} +/- {rows[name]['cv_auc_sem']:.4f} "
              f"(folds {' '.join(f'{a:.3f}' for a in aucs)})")

    best = max(rows, key=lambda k: rows[k]["cv_auc_mean"])
    print(f"\nCV-selected: {best}")
    kind = rows[best]["kind"]
    param = rows[best]["param"]
    p_te = fit_predict(kind, param, X[tr], y[tr], X[te])
    p_tr = fit_predict(kind, param, X[tr], y[tr], X[tr])
    test = {"n": len(te), "wrong": int((y[te] == 0).sum()),
            "auc": fp.auc(y[te], p_te),
            "catch_at_0.5": int(((y[te] == 0) & (p_te < 0.5)).sum()),
            "correct_rejected_at_0.5": int(((y[te] == 1) & (p_te < 0.5)).sum()),
            "pareto": fp.pareto(y[te], p_te, [0, 3, 5, 9, 14]),
            "train_insample_auc": fp.auc(y[tr], p_tr)}
    pa = test["pareto"]
    print(f"  organic-66 (9 wrongs, SE~0.08): AUC {test['auc']:.3f}  "
          f"catch@0.5 {test['catch_at_0.5']}/9  CR@0.5 {test['correct_rejected_at_0.5']}/57")
    print("  frontier catch/CR @ 0/3/5/9/14: "
          + " ".join(f"{pa[str(c)]['caught']}/{pa[str(c)]['correct_rejected']}"
                     for c in (0, 3, 5, 9, 14)))

    zero8 = zeroeight_check()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "states": str(STATES), "n": len(ids), "dim": int(X.shape[1]),
        "selection": "GroupKFold(5) by task_id on the 871 harvest train items",
        "note": "the 66-item test has 9 wrongs (AUC SE ~0.08); it confirms, it does not select",
        "configs": rows, "selected": best, "selected_test": test,
        "p_test": {ids[n]: float(v) for n, v in zip(te, p_te)},
        "zeroeightb": zero8,
    }, indent=1))
    print(f"wrote {args.out}")
    return 0


def zeroeight_check() -> dict:
    """Same question on the 0.8B, where the MLP beat the ridge 0.844 vs 0.688.

    Those 0.8B heads were fit at ridge_fit's default lambda=1, while the 176B head needed 10000 on a
    comparable 2560-d input.  If the 0.8B ridge was simply under-regularised, the "MLP beats ridge"
    result is a regularisation artefact, not a capacity effect -- and it is the belief the whole
    head-selection history rests on, so it is worth 30 seconds to check.
    """
    z = np.load(R2 / "readout-states-intern-decision-0.8b-r2.npz", allow_pickle=False)
    Xc = np.concatenate([z[k] for k in ("L0.5", "L0.75", "L1")], axis=1).astype(np.float64)
    y = z["labels"].astype(int).astype(np.float64)
    sp = z["splits"].astype(str)
    src = z["sources"].astype(str)
    task = z["task_ids"].astype(str)
    tr = np.where(sp == "train")[0]
    te = np.where((sp == "test") & (src == "harvest"))[0]
    print(f"\n0.8B concat {Xc.shape[1]}-d: train {len(tr)} ({int((y[tr]==0).sum())} wrong), "
          f"organic test {len(te)} ({int((y[te]==0).sum())} wrong)")
    folds = list(GroupKFold(n_splits=5).split(tr, y[tr], task[tr]))
    out: dict = {}
    for kind, param in ([("ridge", lam) for lam in LAMBDA_GRID] + [("mlp", 64)]):
        name = f"0.8B/{kind}-{param:g}"
        aucs = []
        for fit_l, hold_l in folds:
            f, h = tr[fit_l], tr[hold_l]
            aucs.append(fp.auc(y[h], fit_predict(kind, param, Xc[f], y[f], Xc[h])))
        a = np.array(aucs, dtype=float)
        sem = a.std(ddof=1) / np.sqrt(len(a))
        out[name] = {"cv_auc_mean": float(a.mean()), "cv_auc_sem": float(sem)}
        extra = ""
        if kind == "mlp" or param in (1.0, LAMBDA_GRID[-1]):
            p = fit_predict(kind, param, Xc[tr], y[tr], Xc[te])
            extra = f"  organic-66 AUC {fp.auc(y[te], p):.3f}"
        print(f"  {name:18s} grouped-CV AUC {a.mean():.4f} +/- {sem:.4f}{extra}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())