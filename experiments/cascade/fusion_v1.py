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
WINDOW_B = REPO / "experiments/cascade/results/scion176b-harvest-20261018"
HARVEST_HEAD = WINDOW_B / "harvest-refit-head.json"
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


def self_harvest_probs(ids: list[str], folds: int,
                       vecs: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    """`self-harvest`, the Window B refit head, used the same honest way as the base signals.

    The head is fit on the 871 harvest *train* items, so its score on those rows is in-sample
    (train AUC 0.996) and a fusion that learned to trust the feature from them would be fitting the
    head's memorisation -- wiring the exported head in naively read 0.924 organic against 0.889 for
    the honest version.  So: out-of-fold scores on the harvest train rows (GroupKFold by task_id,
    the same lambda the head was selected with), and the full-fit head everywhere else.  Everywhere
    else is honest too: the 66 organic test items are out of task (their tasks are disjoint from the
    head's), and the bench items were never in its training corpus at all.
    """
    path = WINDOW_B / "probe-states-scion176b-harvest.npz"
    if not path.exists() or not HARVEST_HEAD.exists():
        return None
    lam = float(json.loads(HARVEST_HEAD.read_text())["training"]["lambda"])
    z = np.load(path, allow_pickle=True)
    hid = [str(i) for i in z["item_ids"]]
    X = z["L1"].astype(np.float64)
    y = z["labels"].astype(int).astype(np.float64)
    splits = z["splits"].astype(str)
    task = np.array([i.split(":")[0] for i in hid])
    tr, te = np.where(splits == "train")[0], np.where(splits == "test")[0]
    p = np.full(len(hid), np.nan)
    for fit_l, hold_l in GroupKFold(n_splits=folds).split(tr, y[tr], task[tr]):
        f, h = tr[fit_l], tr[hold_l]
        model = ridge_fit(X[f], y[f], lam=lam)
        a, b = platt(ridge_score(X[f], model), y[f])
        p[h] = 1.0 / (1.0 + np.exp(-(a * ridge_score(X[h], model) + b)))
    model = ridge_fit(X[tr], y[tr], lam=lam)
    a, b = platt(ridge_score(X[tr], model), y[tr])
    full = lambda v: 1.0 / (1.0 + np.exp(-(a * ridge_score(v, model) + b)))  # noqa: E731
    p[te] = full(X[te])
    pos = {i: n for n, i in enumerate(ids)}
    out = np.full(len(ids), np.nan)
    n_harvest = 0
    for n, i in enumerate(hid):
        if i in pos:
            out[pos[i]] = p[n]
            n_harvest += 1
    # items outside the harvest corpus (bench) were never in the head's training set either
    hset = set(hid)
    rest = [i for i in ids if i not in hset and i in vecs]
    if rest:
        out[[pos[i] for i in rest]] = full(np.stack([vecs[i] for i in rest]))
    print(f"self-harvest: lam={lam:g}, OOF on {int((splits == 'train').sum())} harvest train rows, "
          f"full-fit on {int((splits == 'test').sum())} organic test + {len(rest)} out-of-corpus items")
    return {"self-harvest": out}


def load_self_probs(ids: list[str], folds: int = 5) -> dict[str, np.ndarray]:
    """The 176B's own heads, applied to every state on disk (port + Window A + Window B).

    Window B (all 937 harvest items) is what puts the self signal on the organic train side; the
    clean cross-task refit head it produces is included as `self-harvest` when present.
    """
    heads = {"self-jbench": json.loads(HEAD_JBENCH.read_text()),
             "self-community": json.loads(HEAD_COMMUNITY.read_text())}
    vecs: dict[str, np.ndarray] = {}
    sources = [PORT, WINDOW, WINDOW_B / "probe-states-scion176b-harvest.npz"]
    used = []
    for path in sources:
        if not path.exists():
            continue
        used.append(path.name)
        z = np.load(path, allow_pickle=True)
        key = "L1" if "L1" in z.files else "H"
        for n, i in enumerate(z["item_ids"]):
            vecs.setdefault(str(i), z[key][n].astype(np.float64))
    print(f"self signal state sources: {used}")
    out: dict[str, np.ndarray] = {}
    pos = {i: n for n, i in enumerate(ids)}
    for name, head in heads.items():
        p = np.full(len(ids), np.nan)
        have = [i for i in ids if i in vecs]
        p[[pos[i] for i in have]] = fp.apply_head(head, np.stack([vecs[i] for i in have]))
        out[name] = p
    harvest = self_harvest_probs(ids, folds, vecs)
    if harvest is not None:
        out.update(harvest)
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    wrong = y == 0
    return {
        "n": int(y.size), "wrong": int(wrong.sum()), "correct": int((y == 1).sum()),
        "auc": fp.auc(y, p),
        "catch_at_0.5": int((wrong & (p < 0.5)).sum()),
        "correct_rejected_at_0.5": int(((y == 1) & (p < 0.5)).sum()),
    }


def _fmt(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:.3f}"


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
    "base4+self-harvest": ["L0.5", "MLP", "p_raw", "self-harvest"],
    "base4+self-harvest+community": ["L0.5", "MLP", "p_raw", "self-harvest", "self-community"],
    "base4+self-harvest+jbench": ["L0.5", "MLP", "p_raw", "self-harvest", "self-jbench"],
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
    selfp = load_self_probs(ids, args.folds)
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
        # a feature only exists where its states were embedded; fit and score on that support
        avail = np.ones(len(ids), dtype=bool)
        for f in feats:
            if not np.isfinite(prob[f][fit_idx]).all():
                raise SystemExit(f"fit rows missing feature {f}")
            avail &= np.isfinite(prob[f])
        clf, mu, sd = fit_logit(feature_matrix(feats, prob, fit_idx), y[fit_idx])
        p_all = np.full(len(ids), np.nan)
        te_mask = test & avail
        p_all[te_mask] = apply_logit(clf, mu, sd,
                                     feature_matrix(feats, prob, np.where(te_mask)[0]))
        p_all[fit_idx] = apply_logit(clf, mu, sd, feature_matrix(feats, prob, fit_idx))
        entry = {"features": feats, "fit_items": int(fit_idx.size),
                 "scored_items": int(avail.sum()),
                 "coef": clf.coef_.ravel().tolist()}
        for sname, mask in sl.items():
            m = mask & avail
            entry[sname] = metrics(y[m], p_all[m])
            if sname == "organic-66":
                entry[sname]["pareto_organic"] = fp.pareto(y[m], p_all[m], [0, 3, 5, 9, 14])
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

    # ---- with-self variants: fit on the train rows where those self features exist.
    #      Before Window B that was bench-262 + harvest-qa-70 (no organic train row at all);
    #      with Window B the frozen community/jbench signals cover all 1,133 train rows and the
    #      clean harvest refit covers the 871 harvest train rows. ----
    with_self: dict[str, dict] = {}
    for name, feats in SELF_SETS.items():
        if any(f not in prob for f in feats):
            print(f"self {name:32s} skipped (feature not on disk)")
            continue
        need = np.ones(len(ids), dtype=bool)
        for f in feats:
            need &= np.isfinite(prob[f])
        fit_idx = tr_idx[need[tr_idx]]
        entry, _ = run_fit(feats, fit_idx)
        with_self[name] = entry
        o, b = entry["organic-66"], entry["bench-116"]
        print(f"self {name:32s} fit n={entry['fit_items']:4d} scored n={entry['scored_items']:4d} "
              f"organic {_fmt(o['auc'])} bench {_fmt(b['auc'])} "
              f"(organic catch@0.5 {o['catch_at_0.5']}/9 CR {o['correct_rejected_at_0.5']}/57)")

    # ---- ablations: same fit subset, with and without the self feature; and where self is clean ----
    tr_bench = tr_idx[(z["sources"][tr_idx] == "bench")]
    tr_qa = tr_idx[(z["sources"][tr_idx] == "harvest") & (z["buckets"][tr_idx] == "qa")]
    tr_harv = tr_idx[(z["sources"][tr_idx] == "harvest")]
    tr_self = tr_idx[np.isfinite(prob["self-community"])[tr_idx]]
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
    if np.isfinite(prob.get("self-harvest", np.full(len(ids), np.nan))).any():
        diag_specs += [
            ("harvtrain/base4", base4, tr_harv),
            ("harvtrain/base4+self-harvest", base4 + ["self-harvest"], tr_harv),
            ("harvtrain/base4+self-community", base4 + ["self-community"], tr_harv),
        ]
    diagnostics: dict[str, dict] = {}
    for name, feats, fit_idx in diag_specs:
        entry, _ = run_fit(feats, fit_idx)
        diagnostics[name] = entry
        o, b = entry["organic-66"], entry["bench-116"]
        print(f"diag {name:34s} n={entry['fit_items']:4d} "
              f"test {_fmt(entry['test-182']['auc'])} bench {_fmt(b['auc'])} "
              f"organic {_fmt(o['auc'])}")

    # ---- exploratory reference: leave-one-out fusion on test-182 itself (previous reports) ----
    y_te = labels[te_idx]
    org = np.array([z["sources"][n] == "harvest" for n in te_idx])
    loo_feats = {
        "base4(L0.5+MLP+p_raw)": ["L0.5", "MLP", "p_raw"],
        "base4+self-community": ["L0.5", "MLP", "p_raw", "self-community"],
        "base4+self-jbench": ["L0.5", "MLP", "p_raw", "self-jbench"],
        "base4+self-both": ["L0.5", "MLP", "p_raw", "self-community", "self-jbench"],
        "base4+self-harvest": ["L0.5", "MLP", "p_raw", "self-harvest"],
        "base4+self-harvest+community": ["L0.5", "MLP", "p_raw", "self-harvest", "self-community"],
    }
    loo: dict[str, dict] = {}
    for name, feats in loo_feats.items():
        if any(f not in prob for f in feats):
            print(f"LOO  {name:32s} skipped (feature not on disk)")
            continue
        ok = np.isfinite(np.stack([prob[f] for f in feats], axis=1)).all(axis=1)
        idx = te_idx[ok[te_idx]]
        yv = labels[idx]
        orgv = z["sources"][idx] == "harvest"
        F = feature_matrix(feats, prob, idx)
        pv = fp.loo_fusion(F, yv)
        loo[name] = {"features": feats, "n_scored": int(idx.size),
                     "auc_test": fp.auc(yv, pv),
                     "auc_organic": fp.auc(yv[orgv], pv[orgv]) if orgv.any() else None,
                     "pareto_organic": fp.pareto(yv[orgv], pv[orgv], [0, 3, 5, 9, 14])}
        print(f"LOO  {name:32s} n={idx.size:3d} test {_fmt(loo[name]['auc_test'])} "
              f"organic {_fmt(loo[name]['auc_organic'])}")

    # ---- where the organic evidence comes from: the community head's own train tasks ----
    ct = community_train_tasks()
    in_dom = np.array([t in ct for t in z["task_ids"]])
    org_mask = sl["organic-66"]
    wrong_ids = []
    signals = ["L0.5", "MLP", "p_raw", "self-community", "self-harvest"]
    for n in np.where(org_mask & (labels == 0))[0]:
        row = {"item": ids[n], "task": z["task_ids"][n],
               "self_community_domain": "in-domain" if in_dom[n] else "clean",
               "p": {s: (float(prob[s][n]) if np.isfinite(prob[s][n]) else None)
                     for s in signals if s in prob},
               "caught_by": [s for s in signals if s in prob and np.isfinite(prob[s][n])
                             and prob[s][n] < 0.5]}
        wrong_ids.append(row)
    print("\norganic-66 wrong items (p < 0.5 = rejected):")
    for w in wrong_ids:
        ps = " ".join(f"{s}={w['p'][s]:.3f}" if w["p"].get(s) is not None else f"{s}=  n/a"
                      for s in w["p"])
        print(f"  {w['item']:26s} {w['self_community_domain']:10s} {ps} | "
              f"caught by {','.join(w['caught_by']) or 'NOTHING'}")

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
            "with_self_note": "self-176B features are fit on whatever train rows they exist on: "
                              "the frozen community/jbench heads now cover all 1,133 train rows "
                              "(Window B), the clean harvest refit covers the 871 harvest train rows",
            "self_domain_note": "176b-own-community was fit on scion176b-community train tasks; "
                                "in-domain / clean slices split every test item by task overlap. "
                                "self-harvest is the Window B refit (train and test tasks disjoint).",
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
