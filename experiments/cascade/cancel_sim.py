#!/usr/bin/env python
"""Offline early-cancel simulation on generation-context monitor dumps.

For horizon k (trace-relative), trains the trajectory GRU on train tasks and
simulates stopping generation at k when the risk crosses a threshold
(score < thr means "likely wrong" -> cancel):

  caught_wrong / false_cancels / tokens saved = sum(trace_len - k) over cancelled

Thresholds are calibrated on the model's validation split (quantiles of correct
validation scores -> nominal false-cancel rates) and also swept for the full
curve.  Everything is reported on the held-out test tasks.

Memory-safe: only the 8 snapshot rows per stream are read, directly from each
window file via mmap -- the full dump is never loaded into RAM.

Usage:
    cancel_sim.py --windows .../streams/scion-bench-monitor-ctx \
        --out results/.../cancel-sim-scion.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from readout_judge import auc as auc_fn  # noqa: E402

torch.set_num_threads(4)

SEEDS = (0, 1, 2)
QUANTILES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)


def load_meta(windows: Path):
    meta = json.loads((windows / "meta.json").read_text())
    order = meta["order"]
    splits = np.array([meta["tasks"][o.rsplit(":", 1)[0]] for o in order])
    files = sorted((windows / "dump").glob("win_*_h.bin"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if len(files) != len(order):
        raise SystemExit(f"{len(files)} dumps for {len(order)} streams")
    seq = int(meta["seq"])
    dim = files[0].stat().st_size // (seq * 4)
    return (files, seq, dim, np.asarray(meta["lengths"]),
            np.asarray(meta["labels"], float), splits,
            np.asarray(meta["prompt_len"]),
            np.asarray(meta.get("truncated", [False] * len(order))))


def load_snapshots(files, seq, dim, idx, plen, k):
    """[len(idx), 8, dim] snapshot rows at trace-relative 0..k, via mmap."""
    t = np.linspace(0, k, 8).astype(int)
    Xs = np.zeros((len(idx), len(t), dim), dtype=np.float32)
    for j, i in enumerate(idx):
        mm = np.memmap(files[int(i)], dtype=np.float32, mode="r", shape=(seq, dim))
        Xs[j] = mm[int(plen[i]) + t]
        del mm
    return Xs


def gru_scores(Xs, yy, splits_i, seed, *, hidden=32, pca=32, epochs=300):
    tr = np.array([j for j in range(len(Xs)) if splits_i[j] == "train"])
    te = np.array([j for j in range(len(Xs)) if splits_i[j] == "test"])
    flat = Xs[tr].reshape(-1, Xs.shape[2])
    mu = flat.mean(0, keepdims=True)
    sd = flat.std(0, keepdims=True) + 1e-6
    Z = (Xs - mu) / sd
    S = Z[tr].reshape(-1, Z.shape[2])
    S = S - S.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(S, full_matrices=False)
    V = Vt[:pca].T
    P = Z @ V
    fmu = P[tr].reshape(-1, pca).mean(0, keepdims=True)
    fsd = P[tr].reshape(-1, pca).std(0, keepdims=True) + 1e-6
    P = (P - fmu) / fsd
    Xt = torch.tensor(P, dtype=torch.float32)
    yt = torch.tensor(yy, dtype=torch.float32)

    torch.manual_seed(seed)
    perm = np.random.RandomState(seed).permutation(len(tr))
    nval = max(1, int(0.2 * len(tr)))
    vi, ti = tr[perm[:nval]], tr[perm[nval:]]
    gru = nn.GRU(pca, hidden, batch_first=True)
    head = nn.Linear(hidden, 1)
    opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()),
                           lr=2e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        gru.train()
        head.train()
        opt.zero_grad()
        o, _ = gru(Xt[ti])
        loss = lossf(head(o[:, -1]).squeeze(1), yt[ti])
        loss.backward()
        opt.step()
        gru.eval()
        head.eval()
        with torch.no_grad():
            o, _ = gru(Xt[vi])
            sv = head(o[:, -1]).squeeze(1).numpy()
        a = auc_fn(list(sv), list(yy[vi]))
        if a is not None and a > best + 1e-4:
            best, bad = a, 0
            best_state = (gru.state_dict(), head.state_dict())
        else:
            bad += 1
            if bad >= 40:
                break
    if best_state:
        gru.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    gru.eval()
    head.eval()
    with torch.no_grad():
        o, _ = gru(Xt)
        scores = head(o[:, -1]).squeeze(1).numpy()
    return scores, vi, te


def metrics(scores, rel, yv, thr, k):
    cancel = scores < thr
    wrong = yv == 0
    correct = ~wrong
    caught = int((cancel & wrong).sum())
    fc = int((cancel & correct).sum())
    saved = float(((rel - k) * cancel).sum())
    return {"threshold": float(thr), "cancelled": int(cancel.sum()),
            "caught_wrong": caught, "wrong": int(wrong.sum()),
            "false_cancels": fc, "correct": int(correct.sum()),
            "tokens_saved": saved, "tokens_total": float(rel.sum()),
            "savings_pct": saved / max(1.0, float(rel.sum())),
            "acc_drop_drop_policy": fc / max(1, len(yv))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizons", default="64,128,256")
    args = ap.parse_args()

    files, seq, dim, lengths, y, splits, plen, trunc = load_meta(args.windows)
    rel = np.minimum(lengths - plen, 768)  # tokens actually in the trace window

    res = {"windows": str(args.windows), "horizons": {}}
    for k in [int(x) for x in args.horizons.split(",")]:
        idx = np.where((rel > k) & (~trunc))[0]
        Xs = load_snapshots(files, seq, dim, idx, plen, k)
        y_i, splits_i = y[idx], splits[idx]
        seed_rows = []
        for seed in SEEDS:
            scores, vi, te = gru_scores(Xs, y_i, splits_i, seed)
            vi_s, te_s = idx[vi], idx[te]
            val_correct = scores[vi][y[vi_s] > 0.5]
            row = {"curve": []}
            for q in QUANTILES:
                thr = float(np.quantile(val_correct, q)) if len(val_correct) else 0.0
                row["curve"].append({"q": q, **metrics(scores[te], rel[te_s], y[te_s], thr, k)})
            row["oracle_at_k"] = {
                "tokens_saved": float(((rel[te_s] - k) * (y[te_s] == 0)).sum()),
                "tokens_total": float(rel[te_s].sum())}
            row["n_test"] = int(len(te_s))
            row["n_wrong_test"] = int((y[te_s] == 0).sum())
            row["n_correct_test"] = int((y[te_s] == 1).sum())
            seed_rows.append(row)

        res["horizons"][str(k)] = {"n_eval": int(len(idx)), "seeds": seed_rows}
        print(f"k={k}: n={len(idx)}")
        for q_i, q in enumerate(QUANTILES):
            vals = [row["curve"][q_i] for row in seed_rows]
            caught = np.mean([v["caught_wrong"] for v in vals])
            fc = np.mean([v["false_cancels"] for v in vals])
            sv = np.mean([v["savings_pct"] for v in vals])
            print(f"   q={q:4.2f}: caught {caught:.1f}/"
                  f"{vals[0]['wrong']} wrong, FC {fc:.1f}/{vals[0]['correct']} "
                  f"correct, savings {sv*100:.1f}%")
        orc = seed_rows[0]["oracle_at_k"]
        print(f"   oracle: savings {orc['tokens_saved']/max(1.0,orc['tokens_total'])*100:.1f}%")

    args.out.write_text(json.dumps(res, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
