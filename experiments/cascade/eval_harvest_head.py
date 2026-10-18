#!/usr/bin/env python3
"""Window B: the generator's own head, refit cleanly on the harvest corpus.

Window A measured `176b-own-community` (fit on the port's community corpus) on the organic-66 and
got 0.803 AUC, but 61% of organic-66 sits inside that head's own train tasks.  Window B embeds the
judge states for all 937 harvest items, which makes two things possible that Window A could not:

  1. a **clean cross-task** head: ridge+Platt fit on the 871 harvest *train* items and tested on
     the 66 organic *test* items, whose tasks are disjoint from the train tasks (verified here) —
     the honest number for "the generator reading its own natural errors, out of task";
  2. the 176B signal on the **organic train side**, which `fusion_v1.py` needs to fit a trustworthy
     weight instead of inheriting one from bench.

Also re-applies the two frozen heads to the new states and checks the cross-window states against
Window A's (the shared harvest rows must be identical).

    .venv/bin/python experiments/cascade/eval_harvest_head.py \
        --out experiments/cascade/results/scion176b-harvest-20261018/harvest-head-eval.json
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
WORK = REPO.parent
sys.path.insert(0, str(HERE))

import fusion_probe as fp  # noqa: E402
from fusion_v1 import metrics  # noqa: E402
from decision_readout import platt, ridge_fit, ridge_score  # noqa: E402
from probe_judge import best_ba, ece, no_leak_threshold, threshold_stats  # noqa: E402

WINDOW_B = REPO / "experiments/cascade/results/scion176b-harvest-20261018"
WINDOW_A = REPO / "experiments/cascade/results/scion176b-self-organic-20261018"
HEAD_JBENCH = WORK / "scion/decision-heads/scion-176b-a6b-decision-head.json"
HEAD_COMMUNITY = WORK / "scion/decision-heads-work/overnight/scion176b-community-indomain-head.json"
COMMUNITY_SAMPLES = WORK / "scion/decision-heads-work/overnight/scion176b-community-samples.jsonl"
LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
CAPS = (0, 3, 5, 9, 14)


def load_states(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    key = "L1" if "L1" in z.files else "H"
    ids = [str(i) for i in z["item_ids"]]
    tasks = np.array([str(t) for t in z["task_ids"]]) if "task_ids" in z.files \
        else np.array([i.split(":")[0] for i in ids])
    return {"ids": ids, "X": z[key].astype(np.float64),
            "labels": z["labels"].astype(int),
            "splits": np.array([str(s) for s in z["splits"]]),
            "task_ids": tasks}


def family_of(item_id: str) -> str:
    return item_id.split("-")[0]


def community_train_tasks() -> set[str]:
    out = set()
    for line in COMMUNITY_SAMPLES.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if str(row["split"]) == "train":
                out.add(str(row["task_id"]))
    return out


def seq_of(pa: dict) -> str:
    return " ".join(f"{pa[str(c)]['caught']}/{pa[str(c)]['correct_rejected']}" for c in CAPS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--states", type=Path,
                    default=WINDOW_B / "probe-states-scion176b-harvest.npz")
    args = ap.parse_args()

    st = load_states(args.states)
    ids, y, splits, X = st["ids"], st["labels"], st["splits"], st["X"]
    train, test = splits == "train", splits == "test"
    fam = np.array([family_of(i) for i in ids])
    tr, te = np.where(train)[0], np.where(test)[0]
    print(f"harvest states {len(ids)}  train {int(train.sum())}  test {int(test.sum())}  dim {X.shape[1]}")
    print(f"  organic-66: {int((y[test] == 0).sum())} wrong / {int(test.sum())}, "
          f"{len(set(st['task_ids'][test].tolist()))} tasks, "
          f"task overlap with train {len(set(st['task_ids'][train].tolist()) & set(st['task_ids'][test].tolist()))}")

    out: dict = {"states": str(args.states), "n": len(ids), "dim": int(X.shape[1]),
                 "splits": {"train": int(train.sum()), "test": int(test.sum()),
                            "test_wrong": int((y[test] == 0).sum())}}

    # ---- cross-window check: every row Window A embedded must come out identical here ----
    wa = load_states(WINDOW_A / "probe-states-scion176b-transfer.npz")
    pos_a = {i: n for n, i in enumerate(wa["ids"])}
    shared = [i for i in wa["ids"] if i in set(ids)]
    if shared:
        a = np.stack([wa["X"][pos_a[i]] for i in shared])
        b = np.stack([X[ids.index(i)] for i in shared])
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        bit = int((np.abs(a - b).max() == 0))
        out["cross_window"] = {"shared_rows": len(shared), "mean_cosine": float(cos.mean()),
                               "min_cosine": float(cos.min()), "bit_identical_rows": bit}
        print(f"cross-window vs Window A: {len(shared)} shared rows, mean cosine {cos.mean():.6f}, "
              f"bit-identical {bit}")

    # ---- frozen heads on the new states ----
    frozen_p: dict[str, np.ndarray] = {}
    frozen: dict[str, dict] = {}
    for name, path in (("176b-own-jbench", HEAD_JBENCH),
                       ("176b-own-community", HEAD_COMMUNITY)):
        p = fp.apply_head(json.loads(path.read_text()), X)
        frozen_p[name] = p
        entry = {"path": str(path), "harvest-train-871": {}, "harvest-test-66": {}}
        for sname, mask in (("harvest-train-871", train), ("harvest-test-66", test)):
            yy, pp = y[mask], p[mask]
            entry[sname] = metrics(yy, pp)
            if sname == "harvest-test-66":
                entry[sname].update({
                    "pareto": fp.pareto(yy, pp, list(CAPS)),
                    "best_ba": best_ba(list(pp), list(yy)),
                    "ece": ece(list(pp), list(yy)),
                    "no_leak": no_leak_threshold(list(pp), list(yy)),
                    "at_0.5_stats": threshold_stats(list(pp), list(yy), 0.5),
                    "by_family": {f: metrics(y[mask & (fam == f)], p[mask & (fam == f)])
                                  for f in sorted(set(fam[mask].tolist()))
                                  if (y[mask & (fam == f)] == 0).sum()
                                  and (y[mask & (fam == f)] == 1).sum()},
                })
        frozen[name] = entry
        t = entry["harvest-test-66"]
        print(f"frozen {name:20s} test AUC {t['auc']:.3f} (train {entry['harvest-train-871']['auc']:.3f}) "
              f"catch@0.5 {t['catch_at_0.5']}/9 CR {t['correct_rejected_at_0.5']}/57 | "
              f"{seq_of(t['pareto'])}")
    out["frozen_heads"] = frozen

    # ---- clean refit: lambda picked by grouped CV inside train, then fit on all train ----
    yf = y.astype(np.float64)
    cv: dict[str, dict] = {}
    for lam in LAMBDAS:
        aucs = []
        for fit_l, hold_l in GroupKFold(n_splits=5).split(tr, yf[tr], st["task_ids"][tr]):
            f, h = tr[fit_l], tr[hold_l]
            model = ridge_fit(X[f], yf[f], lam=lam)
            a, b = platt(ridge_score(X[f], model), yf[f])
            aucs.append(fp.auc(y[h], 1.0 / (1.0 + np.exp(-(a * ridge_score(X[h], model) + b)))))
        cv[str(lam)] = {"mean_auc": float(np.mean(aucs)), "folds": [float(v) for v in aucs]}
        print(f"  grouped-CV lambda {lam:7g}: AUC {np.mean(aucs):.3f}")
    best_lam = max(LAMBDAS, key=lambda l: cv[str(l)]["mean_auc"])
    print(f"  picked lambda {best_lam:g}")

    refits: dict[str, dict] = {}
    probs: dict[str, np.ndarray] = {}
    for tag, lam in (("default(lam=1)", 1.0), (f"cv-picked(lam={best_lam:g})", float(best_lam))):
        model = ridge_fit(X[tr], yf[tr], lam=lam)
        a, b = platt(ridge_score(X[tr], model), yf[tr])
        p_tr = 1.0 / (1.0 + np.exp(-(a * ridge_score(X[tr], model) + b)))
        p_te = 1.0 / (1.0 + np.exp(-(a * ridge_score(X[te], model) + b)))
        probs[tag] = p_te
        p_full = np.full(len(ids), np.nan)
        p_full[te] = p_te
        entry = {"lambda": float(lam), "platt": [float(a), float(b)],
                 "train_insample_auc": fp.auc(y[tr], p_tr),
                 "harvest-test-66": metrics(y[te], p_te)}
        entry["harvest-test-66"].update({
            "pareto": fp.pareto(y[te], p_te, list(CAPS)),
            "best_ba": best_ba(list(p_te), list(y[te])),
            "ece": ece(list(p_te), list(y[te])),
            "no_leak": no_leak_threshold(list(p_te), list(y[te])),
            "by_family": {f: metrics(y[test & (fam == f)], p_full[test & (fam == f)])
                          for f in sorted(set(fam[test].tolist()))
                          if (y[test & (fam == f)] == 0).sum() and (y[test & (fam == f)] == 1).sum()},
        })
        refits[tag] = entry
        t = entry["harvest-test-66"]
        print(f"refit {tag:22s} test AUC {t['auc']:.3f} (train in-sample {entry['train_insample_auc']:.3f}) "
              f"catch@0.5 {t['catch_at_0.5']}/9 CR {t['correct_rejected_at_0.5']}/57 | "
              f"{seq_of(t['pareto'])}")
    out["grouped_cv"] = cv
    out["refits"] = refits
    tag_best = f"cv-picked(lam={best_lam:g})"

    # ---- export the clean refit head in the standard head format, so fusion_v1.py can use it
    #      as a feature without refitting (single source of truth) ----
    model = ridge_fit(X[tr], yf[tr], lam=float(best_lam))
    w, (mu, sd, bias) = model
    a, b = platt(ridge_score(X[tr], model), yf[tr])
    head_path = args.out.parent / "harvest-refit-head.json"
    head_path.write_text(json.dumps({
        "name": "scion176b-harvest-refit-head",
        "model": "scion176b (harvest judge states, Window B)",
        "readout": {"kind": "ridge-to-labels", "position": "last prompt token",
                    "layer": "final", "dim": int(X.shape[1]), "candidate": "answer",
                    "calibration": "Platt on train scores"},
        "training": {"corpus": "decision-harvest-r2-20261017/harvest-bench-r2.jsonl",
                     "n_train": int(tr.size), "n_test": int(te.size),
                     "lambda": float(best_lam),
                     "lambda_selection": "GroupKFold(5) by task_id inside the train split",
                     "test_task_overlap_with_train": 0},
        "weights": {"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
                    "b": float(bias), "platt": [float(a), float(b)]},
        "test": {"n": int(te.size),
                 "auc": refits[tag_best]["harvest-test-66"]["auc"],
                 "at_0.5": {"catch": refits[tag_best]["harvest-test-66"]["catch_at_0.5"],
                            "correct_rejected":
                                refits[tag_best]["harvest-test-66"]["correct_rejected_at_0.5"]}},
    }, indent=1))
    out["head_file"] = str(head_path)
    print(f"wrote {head_path}")

    # ---- the 9 organic wrongs, every reading side by side ----
    ct = community_train_tasks()
    rows = []
    for k, n in enumerate(te):
        if y[n] != 0:
            continue
        rows.append({
            "item": ids[n], "task": st["task_ids"][n], "family": fam[n],
            "community_head_domain": "in-domain" if st["task_ids"][n] in ct else "clean",
            "p_harvest_refit": float(probs[tag_best][k]),
            "p_176b-own-community": float(frozen_p["176b-own-community"][n]),
            "p_176b-own-jbench": float(frozen_p["176b-own-jbench"][n]),
        })
    out["organic_wrongs"] = rows
    print("\norganic-66 wrong items:")
    for r in rows:
        print(f"  {r['item']:26s} {r['community_head_domain']:10s} refit {r['p_harvest_refit']:.3f}  "
              f"community {r['p_176b-own-community']:.3f}  jbench {r['p_176b-own-jbench']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())