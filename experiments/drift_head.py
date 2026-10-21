"""The drift decision, trained on top of the frozen stock (the JEV recipe).

Three fine-tunes lost to the stock on this job (multi-task 0.420, dedup
hinge 0.731, drift-only 0.277): moving the weights destroys the
pre-trained topical space.  The campaign's *other* result says what to do
instead - the 176B's judge is a head on frozen hidden states - so this
trains a head on the stock's frozen pair embeddings and leaves the
encoder alone.

**The precision the training needs.**

* The decision is a function of the *pair* (recent, older), so the
  features are the pair's embeddings, not one text.  A linear head on
  the concatenation cannot even represent the cosine (a quadratic form);
  the elementwise product can, and the difference term adds asymmetry.
* The split is by conversation: 41 train, 41 hold-out.  The scaler and
  the model are fitted on train only.
* The threshold is chosen on **train** (the max-F1 point) and applied to
  the hold-out - the deployable rule.  The oracle is reported for
  reference, never gated on.
* No lexical feature: the label *is* a lexical Jaccard, so a lexical
  input would be leakage.  The lengths are kept (available at inference)
  and the no-length arm is reported beside them, because the label is
  length-biased and the head can learn the bias.

The comparison is the stock's own cosine under the same threshold rule.

    python -m experiments.drift_head
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "encoder-suite" / "results" / "drift-head"
SEED = 20261020


def features(a: np.ndarray, b: np.ndarray, la: int, lb: int, kind: str):
    parts = []
    if kind in ("pair", "pairdiff"):
        parts.append(a * b)
    if kind == "pairdiff":
        parts.append(a - b)
    parts.append(np.array([float(a @ b)]))
    if kind.endswith("+len"):
        parts.append(np.array([np.log1p(la), np.log1p(lb)]))
    return np.concatenate(parts)


def build(kind: str, cases, vectors):
    n = len(cases)
    rows = []
    for i, case in enumerate(cases):
        a = vectors[i]
        b = vectors[n + i]
        rows.append(features(a, b, len(case["recent"]), len(case["older"]), kind))
    return np.vstack(rows)


def f1_at(scores, labels, threshold: float) -> dict:
    tp = fp = fn = 0
    for s, l in zip(scores, labels):
        if l and s >= threshold:
            tp += 1
        elif l:
            fn += 1
        elif s >= threshold:
            fp += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"the_threshold": round(float(threshold), 4),
            "the_precision": round(precision, 4),
            "the_recall": round(recall, 4),
            "the_f1": round(f1, 4),
            "the_fires": round((tp + fp) / max(len(labels), 1), 3)}


def best_threshold(scores, labels) -> dict:
    """The max-F1 threshold - fitted on the train split only."""
    order = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 401)))
    best = {"the_f1": -1.0}
    for threshold in order:
        row = f1_at(scores, labels, threshold)
        if row["the_f1"] > best["the_f1"]:
            best = row
    return best


def auc(scores, labels) -> float:
    from experiments.encoder_suite import _auc
    return round(_auc([float(s) for s in scores], [bool(x) for x in labels]), 4)


def main() -> int:
    from experiments.encoder_suite import build_drift_cases, load_conversations, make_encoder
    from experiments.qwen_encoder import split_conversations

    conversations = load_conversations(["harness_state",
                                        "../splinter-memory/harness_state"])
    train_convs, holdout = split_conversations(conversations, seed=SEED)
    train_cases = build_drift_cases(train_convs)
    test_cases = build_drift_cases(holdout)
    print(f"the cases: train {len(train_cases)} "
          f"({sum(1 for c in train_cases if c['drift'])} drift), "
          f"hold-out {len(test_cases)} "
          f"({sum(1 for c in test_cases if c['drift'])} drift)", flush=True)

    embed, _ = make_encoder("l3")
    print("  embedding the train cases", flush=True)
    train_vecs = embed([c["recent"] for c in train_cases]
                       + [c["older"] for c in train_cases])
    print("  embedding the hold-out cases", flush=True)
    test_vecs = embed([c["recent"] for c in test_cases]
                      + [c["older"] for c in test_cases])

    y_train = np.array([bool(c["drift"]) for c in train_cases])
    y_test = np.array([bool(c["drift"]) for c in test_cases])

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    arms = {}
    # the baseline: the stock's own cosine, same threshold rule
    train_cos = build("cos", train_cases, train_vecs).ravel()
    test_cos = build("cos", test_cases, test_vecs).ravel()
    arms["the cosine"] = {
        "the_auc": auc(test_cos, y_test),
        "the_train_best": best_threshold(train_cos, y_train),
        "the_holdout_at_train": f1_at(test_cos, y_test,
                                      best_threshold(train_cos, y_train)["the_threshold"]),
        "the_holdout_oracle": best_threshold(test_cos, y_test),
    }

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for kind in ("pair+len", "pairdiff+len", "pair", "pairdiff",
                 "cos+len", "cos"):
        X_train = build(kind, train_cases, train_vecs)
        X_test = build(kind, test_cases, test_vecs)
        best = None
        for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, class_weight="balanced",
                                   max_iter=3000, solver="lbfgs"))
            cv = cross_val_score(model, X_train, y_train, cv=kfold,
                                 scoring="roc_auc").mean()
            if best is None or cv > best[0]:
                best = (cv, c)
        cv, c = best
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced",
                               max_iter=3000, solver="lbfgs"))
        model.fit(X_train, y_train)
        train_scores = model.predict_proba(X_train)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        thr = best_threshold(train_scores, y_train)
        arms[kind] = {
            "the_features": int(X_train.shape[1]),
            "the_cv_auc": round(float(cv), 4),
            "the_c": c,
            "the_auc": auc(test_scores, y_test),
            "the_train_best": thr,
            "the_holdout_at_train": f1_at(test_scores, y_test, thr["the_threshold"]),
            "the_holdout_oracle": best_threshold(test_scores, y_test),
        }
        print(f"  {kind:14s} features {X_train.shape[1]:4d}  cv {cv:.4f}  "
              f"hold-out AUC {arms[kind]['the_auc']:.4f}  "
              f"F1 at train thr {arms[kind]['the_holdout_at_train']['the_f1']:.4f}  "
              f"(oracle {arms[kind]['the_holdout_oracle']['the_f1']:.4f})", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "drift-head.json").write_text(json.dumps(
        {"the_seed": SEED, "the_arms": arms}, indent=1))
    print(f"  wrote {OUT / 'drift-head.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
