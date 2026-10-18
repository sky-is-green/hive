#!/usr/bin/env python3
"""Fusion v1 — the purpose-trained gate, offline, honest out-of-fold on the r2 train split.

The deployed 0.8B heads were fit on the r2 train split (1,133 items), so their train-split
probabilities are in-sample and cannot be used to fit a second-stage gate.  This script recomputes
**out-of-fold** base probabilities on the train split (GroupKFold by task_id, 5 folds — test tasks
are disjoint from train, so a task-disjoint OOF mirrors the test condition) using the *same* fit
code as `decision_readout.py` (ridge+Platt per depth, and the 64-unit MLP on the concat).  The test
split keeps the deployed-head probabilities (honest: the heads never saw a test task).

Then a single logistic fusion is fit on the OOF train features and evaluated once on test-182,
split by source / bucket / task family, with the organic catch-vs-correct-rejection frontier.

The 176B's own heads (frozen J-Bench + in-domain community) exist for all 182 test items — bench-116
through the port states, organic-66 through Window A — but only 332/1,133 train items (bench-262 +
the 70 harvest-qa train items), never the organic harvest train rows.  So the honest no-self fusion
is the debit-deployable primary; the with-self variants are fit on that self-available train subset
and are labelled as such (Window B is what makes the organic-inclusive self fusion honest).

    .venv/bin/python experiments/cascade/fusion_v1.py --outdir experiments/cascade/results/fusion-v1-20261018
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = REPO.parent
sys.path.insert(0, str(HERE))

import fusion_probe as fp  # noqa: E402
from decision_readout import _mlp_fit, _mlp_predict, platt, ridge_fit, ridge_score  # noqa: E402

R2 = REPO / "experiments/cascade/results/decision-readout-r2-20261017"
PORT = WORK / "scion/decision-heads-work/overnight/probe-states-main176b.npz"
WINDOW = REPO / ("experiments/cascade/results/scion176b-self-organic-20261018/"
                 "probe-states-scion176b-transfer.npz")
HEAD_JBENCH = WORK / "scion/decision-heads/scion-176b-a6b-decision-head.json"
HEAD_COMMUNITY = WORK / "scion/decision-heads-work/overnight/scion176b-community-indomain-head.json"
COMMUNITY_SAMPLES = WORK / "scion/decision-heads-work/overnight/scion176b-community-samples.jsonl"

DEPTHS = ("L0.5", "L0.75", "L1")


def load_r2() -> dict:
    z = np.load(R2 / "readout-states-intern-decision-0.8b-r2.npz", allow_pickle=False)
    return {
        "ids": [str(i) for i in z["item_ids"]],
        "labels": z["labels"].astype(int),
        "splits": np.array([str(s) for s in z["splits"]]),
        "sources": np.array([str(s) for s in z["sources"]]),
        "buckets": np.array([str(s) for s in z["buckets"]]),
        "origins": np.array([str(s) for s in z["origins"]]),
        "task_ids": np.array([str(s) for s in z["task_ids"]]),
        "X": {k: z[k].astype(np.float64) for k in DEPTHS},
        "Xc": np.concatenate([z[k] for k in DEPTHS], axis=1).astype(np.float64),
        "p_raw": z["p_raw"].astype(np.float64),
    }


def deployed_probs() -> dict[str, np.ndarray]:
    """Deployed-head probabilities on all 1,315 items (in-sample on train; honest on test)."""
    _, p = fp.load_r2()
    return {"L0.5": p["L0.5"], "L0.75": p["L0.75"], "L1": p["L1"],
            "ridge-concat": p["ridge-concat"], "MLP": p["mlp-concat"], "p_raw": p["p_raw"]}


def oof_base_probs(z: dict, folds: int, seed: int) -> dict[str, np.ndarray]:
    """Out-of-fold base probabilities on the train split; deployed probs elsewhere."""
    n = len(z["labels"])
    dep = deployed_probs()
    oof = {k: np.full(n, np.nan) for k in ("L0.5", "L0.75", "L1", "ridge-concat", "MLP")}
    train = z["splits"] == "train"
    tr = np.where(train)[0]
    y = z["labels"].astype(np.float64)
    gkf = GroupKFold(n_splits=folds)
    for fit_l, hold_l in gkf.split(tr, y[tr], z["task_ids"][tr]):
        fit, hold = tr[fit_l], tr[hold_l]
        for key in DEPTHS:
            model = ridge_fit(z["X"][key][fit], y[fit])
            a, b = platt(ridge_score(z["X"][key][fit], model), y[fit])
            oof[key][hold] = 1.0 / (1.0 + np.exp(-(a * ridge_score(z["X"][key][hold], model) + b)))
        model = ridge_fit(z["Xc"][fit], y[fit])
        a, b = platt(ridge_score(z["Xc"][fit], model), y[fit])
        oof["ridge-concat"][hold] = 1.0 / (1.0 + np.exp(-(a * ridge_score(z["Xc"][hold], model) + b)))
        net, stats = _mlp_fit(z["Xc"][fit], y[fit], seed=seed, epochs=400)
        oof["MLP"][hold] = _mlp_predict(z["Xc"][hold], net, stats)
    # deposit deployed probabilities where OOF is not defined (the test split)
    for key in oof:
        miss = np.isnan(oof[key])
        oof[key][miss] = dep[key][miss]
    oof["p_raw"] = dep["p_raw"]
    return oof


def load_self_probs(ids: list[str]) -> dict[str, np.ndarray]:
    """The 176B's own heads, applied to every state on disk (port + Window A)."""
    heads = {"self-jbench": json.loads(HEAD_JBENCH.read_text()),
             "self-community": json.loads(HEAD_COMMUNITY.read_text())}
    vecs: dict[str, np.ndarray] = {}
    for path in (PORT, WINDOW):
        z = np.load(path, allow_pickle=True)
        key = "L1" if "L1" in z.files else "H"
        for n, i in enumerate(z["item_ids"]):
            vecs[str(i)] = z[key][n].astype(np.float64)
    out: dict[str, np.ndarray] = {}
    pos = {i: n for n, i in enumerate(ids)}
    for name, head in heads.items():
        p = np.full(len(ids), np.nan)
        have = [i for i in ids if i in vecs]
        p[[pos[i] for i in have]] = fp.apply_head(head, np.stack([vecs[i] for i in have]))
        out[name] = p
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    wrong = y == 0
    return {
        "n": int(y.size), "wrong": int(wrong.sum()), "correct": int((y == 1).sum()),
        "auc": fp.auc(y, p),
        "catch_at_0.5": int((wrong & (p < 0.5)).sum()),
        "correct_rejected_at_0.5": int(((y == 1) & (p < 0.5)).sum()),
    }


def fit_logit(F: np.ndarray, y: np.ndarray, C: float = 1.0):
    mu, sd = F.mean(0), F.std(0) + 1e-9
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit((F - mu) / sd, y)
    return clf, mu, sd


def apply_logit(clf, mu, sd, F: np.ndarray) -> np.ndarray:
    return clf.predict_proba((F - mu) / sd)[:, 1]


def community_train_tasks() -> set[str]:
    """Task ids the 176B in-domain (community) head was fit on — its own train split."""
    out = set()
    for line in COMMUNITY_SAMPLES.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if str(row.get("split")) == "train":
                out.add(str(row["task_id"]))
    return out


def slices(z: dict) -> dict[str, np.ndarray]:
    sp, src, bk = z["splits"], z["sources"], z["buckets"]
    test = sp == "test"
    fam = np.array([t.split("-")[0] for t in z["task_ids"]])
    org = test & (src == "harvest")
    out = {"test-182": test,
           "bench-116": test & (src == "bench"),
           "organic-66": org}
    for f in sorted(set(fam[test].tolist())):
        out[f"organic-{f}"] = org & (fam == f)
    # the self-community head's own train tasks: in-domain vs clean reading of the same items
    ct = community_train_tasks()
    in_dom = np.array([t in ct for t in z["task_ids"]])
    out["organic-66-indomain"] = org & in_dom
    out["organic-66-clean"] = org & ~in_dom
    out["bench-116-indomain"] = test & (src == "bench") & in_dom
    out["bench-116-clean"] = test & (src == "bench") & ~in_dom
    out["test-bench-qa"] = test & (bk == "qa")
    return out


BASE_SETS = {
    "L0.5": ["L0.5"],
    "MLP": ["MLP"],
    "L0.5+MLP": ["L0.5", "MLP"],
    "L0.5+MLP+p_raw": ["L0.5", "MLP", "p_raw"],
    "L0.75+L0.5+MLP+p_raw": ["L0.75", "L0.5", "MLP", "p_raw"],
    "all-base": ["L0.5", "L0.75", "L1", "ridge-concat", "MLP", "p_raw"],
}
SELF_SETS = {
    "base4+self-community": ["L0.5", "MLP", "p_raw", "self-community"],
    "base4+self-jbench": ["L0.5", "MLP", "p_raw", "self-jbench"],
    "base4+self-both": ["L0.5", "MLP", "p_raw", "self-community", "self-jbench"],
}


def feature_matrix(feats: list[str], probs: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    return np.stack([fp.logit(probs[f][idx]) for f in feats], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    z = load_r2()
    ids, labels = z["ids"], z["labels"]
    train = z["splits"] == "train"
    test = z["splits"] == "test"
    sl = slices(z)

    oof = oof_base_probs(z, args.folds, args.seed)
    selfp = load_self_probs(ids)
    self_avail = ~np.isnan(selfp["self-community"])
    print(f"items {len(ids)} train {int(train.sum())} test {int(test.sum())} | "
          f"self available train {int((train & self_avail).sum())} test {int((test & self_avail).sum())}")

    # sanity: the OOF feature should track the deployed in-sample feature but be lower on train
    sanity = {}
    for key in ("L0.5", "MLP", "ridge-concat"):
        sanity[key] = {
            "auc_train_oof": fp.auc(labels[train], oof[key][train]),
            "auc_train_insample": fp.auc(labels[train], deployed_probs()[key][train]),
            "auc_test_deployed": fp.auc(labels[test], oof[key][test]),
        }
        print(f"  {key:13s} train OOF AUC {sanity[key]['auc_train_oof']:.3f} "
              f"vs in-sample {sanity[key]['auc_train_insample']:.3f}; "
              f"test {sanity[key]['auc_test_deployed']:.3f}")

    prob = {**oof, **selfp}
    tr_idx = np.where(train)[0]
    te_idx = np.where(test)[0]
    y = labels

    def run_fit(feats: list[str], fit_idx: np.ndarray) -> tuple[dict, np.ndarray]:
        clf, mu, sd = fit_logit(feature_matrix(feats, prob, fit_idx), y[fit_idx])
        p_all = np.full(len(ids), np.nan)
        p_all[te_idx] = apply_logit(clf, mu, sd, feature_matrix(feats, prob, te_idx))
        p_all[fit_idx] = apply_logit(clf, mu, sd, feature_matrix(feats, prob, fit_idx))
        entry = {"features": feats, "fit_items": int(fit_idx.size),
                 "coef": clf.coef_.ravel().tolist()}
        for sname, mask in sl.items():
            entry[sname] = metrics(y[mask], p_all[mask])
            if sname == "organic-66":
                entry[sname]["pareto_organic"] = fp.pareto(y[mask], p_all[mask], [0, 3, 5, 9, 14])
        return entry, p_all

    # ---- honest no-self fusion: fit on OOF train, evaluate once on test-182 ----
    primary = "L0.5+MLP"
    fusion: dict[str, dict] = {}
    for name, feats in BASE_SETS.items():
        entry, _ = run_fit(feats, tr_idx)
        fusion[name] = entry
        o = entry["organic-66"]
        print(f"base {name:22s} test {entry['test-182']['auc']:.3f} "
              f"bench {entry['bench-116']['auc']:.3f} organic {o['auc']:.3f} "
              f"(catch@0.5 {o['catch_at_0.5']}/9 CR {o['correct_rejected_at_0.5']}/57)")

    # ---- with-self variants: fit on the self-available train subset (bench-262 + harvest-qa-70) ----
    tr_self = tr_idx[self_avail[tr_idx]]
    with_self: dict[str, dict] = {}
    for name, feats in SELF_SETS.items():
        entry, _ = run_fit(feats, tr_self)
        with_self[name] = entry
        o = entry["organic-66"]
        print(f"self {name:22s} test {entry['test-182']['auc']:.3f} "
              f"bench {entry['bench-116']['auc']:.3f} organic {o['auc']:.3f} "
              f"(catch@0.5 {o['catch_at_0.5']}/9 CR {o['correct_rejected_at_0.5']}/57)")

    # ---- ablations: same fit subset, with and without the self feature; and where self is clean ----
    tr_bench = tr_idx[(z["sources"][tr_idx] == "bench")]
    tr_qa = tr_idx[(z["sources"][tr_idx] == "harvest") & (z["buckets"][tr_idx] == "qa")]
    base4 = ["L0.5", "MLP", "p_raw"]
    diag_specs = [
        ("alltrain/base4", base4, tr_idx),
        ("selftrain/base4", base4, tr_self),
        ("selftrain/base4+self-community", base4 + ["self-community"], tr_self),
        ("benchtrain/base4", base4, tr_bench),
        ("benchtrain/base4+self-community", base4 + ["self-community"], tr_bench),
        ("benchtrain/base4+self-jbench", base4 + ["self-jbench"], tr_bench),
        ("qatrain/base4", base4, tr_qa),
        ("qatrain/base4+self-community", base4 + ["self-community"], tr_qa),
    ]
    diagnostics: dict[str, dict] = {}
    for name, feats, fit_idx in diag_specs:
        entry, _ = run_fit(feats, fit_idx)
        diagnostics[name] = entry
        o, b = entry["organic-66"], entry["bench-116"]
        print(f"diag {name:34s} n={entry['fit_items']:4d} test {entry['test-182']['auc']:.3f} "
              f"bench {b['auc']:.3f} organic {o['auc']:.3f}")

    # ---- exploratory reference: leave-one-out fusion on test-182 itself (previous reports) ----
    y_te = labels[te_idx]
    org = np.array([z["sources"][n] == "harvest" for n in te_idx])
    loo_feats = {
        "base4(L0.5+MLP+p_raw)": ["L0.5", "MLP", "p_raw"],
        "base4+self-community": ["L0.5", "MLP", "p_raw", "self-community"],
        "base4+self-jbench": ["L0.5", "MLP", "p_raw", "self-jbench"],
        "base4+self-both": ["L0.5", "MLP", "p_raw", "self-community", "self-jbench"],
    }
    loo: dict[str, dict] = {}
    for name, feats in loo_feats.items():
        F = feature_matrix(feats, prob, te_idx)
        pv = fp.loo_fusion(F, y_te)
        loo[name] = {"features": feats, "auc_test": fp.auc(y_te, pv),
                     "auc_organic": fp.auc(y_te[org], pv[org]),
                     "pareto_organic": fp.pareto(y_te[org], pv[org], [0, 3, 5, 9, 14])}
        print(f"LOO  {name:22s} test {loo[name]['auc_test']:.3f} organic {loo[name]['auc_organic']:.3f}")

    # ---- where the organic evidence comes from: the community head's own train tasks ----
    ct = community_train_tasks()
    in_dom = np.array([t in ct for t in z["task_ids"]])
    org_mask = sl["organic-66"]
    wrong_ids = []
    for n in np.where(org_mask & (labels == 0))[0]:
        wrong_ids.append({"item": ids[n], "task": z["task_ids"][n],
                          "self_community_domain": "in-domain" if in_dom[n] else "clean",
                          "p_L0.5": float(oof["L0.5"][n]), "p_MLP": float(oof["MLP"][n]),
                          "p_self-community": float(selfp["self-community"][n])})
    print("\norganic-66 wrong items by self-community domain:")
    for w in wrong_ids:
        print(f"  {w['item']:26s} {w['self_community_domain']:10s} "
              f"L0.5 {w['p_L0.5']:.3f} MLP {w['p_MLP']:.3f} self-community {w['p_self-community']:.3f}")

    out = {
        "corpus": {"items": len(ids), "train": int(train.sum()), "test": int(test.sum()),
                   "test_wrong": int((labels[test] == 0).sum()),
                   "self_available_train": int((train & self_avail).sum()),
                   "self_available_test": int((test & self_avail).sum())},
        "method": {
            "oof": f"GroupKFold({args.folds}) by task_id on the train split; "
                   "ridge+Platt and the 64-unit MLP refit per fold with decision_readout helpers",
            "test_features": "deployed heads (fit on full train, test tasks disjoint from train)",
            "fusion": "single logistic regression on standardized logits, fit on train OOF, one test pass",
            "with_self_note": "self-176B features exist for all test items but only 332 train items "
                              "(bench-262 + harvest-qa-70); those variants are fit on that subset",
            "self_domain_note": "176b-own-community was fit on scion176b-community train tasks; "
                                "in-domain / clean slices split every test item by task overlap",
        },
        "oof_sanity": sanity,
        "slices": {k: {"n": int(m.sum()), "wrong": int((labels[m] == 0).sum()),
                       "correct": int((labels[m] == 1).sum())} for k, m in sl.items()},
        "organic_wrongs_by_self_domain": wrong_ids,
        "fusion_v1_no_self": fusion,
        "fusion_v1_with_self_subset_fit": with_self,
        "diagnostics": diagnostics,
        "exploratory_loo_test182": loo,
        "primary": primary,
    }
    (args.outdir / "fusion-v1.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {args.outdir / 'fusion-v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
