#!/usr/bin/env python
"""Non-linear / trajectory retest of the L3 online monitor (alignment-corrected).

The first monitor pass loaded ``win_%02d_h.bin`` with a string sort, which
misaligns streams and labels for ~97% of windows (next-token agreement 0.16
under the string sort vs 0.83 in numeric order).  This script reloads the same
dump in numeric order and asks the question the linear probe was meant to
answer: is there a *non-linear* or *trajectory* signal about final correctness
in the mid-stream hidden state that could gate an early abort?

Models per position, trained on train-task streams and scored on held-out
test-task streams:
  - ridge (linear reference, monitor_probe's protocol)
  - PCA-128 + torch MLP (early-stopped on an internal train split)
  - random-Fourier-features ridge (RBF kernel approximation)
  - GRU over 8 evenly spaced trajectory snapshots at two horizons

Emits ``monitor-nonlinear.json`` next to the other monitor artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from probe_judge import ridge_fit, ridge_score  # noqa: E402
from readout_judge import auc  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"
torch.set_num_threads(4)


def load_aligned(windows: Path):
    meta = json.loads((windows / "meta.json").read_text())
    order, lengths, labels = meta["order"], meta["lengths"], meta["labels"]
    splits = np.array([meta["tasks"][o.rsplit(":", 1)[0]] for o in order])
    files = sorted((windows / "dump").glob("win_*_h.bin"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if len(files) != len(order):
        raise SystemExit(f"{len(files)} dumps for {len(order)} streams")
    H = np.stack([np.fromfile(p, dtype=np.float32).reshape(meta["seq"], -1)
                  for p in files])
    return H, np.asarray(lengths), np.asarray(labels, float), splits


def standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def pca_reduce(Xtr, Xte, k=128):
    mu = Xtr.mean(0, keepdims=True)
    S = Xtr - mu
    _, _, Vt = np.linalg.svd(S, full_matrices=False)
    V = Vt[:k].T
    return S @ V, (Xte - mu) @ V


def fit_mlp(Ztr, ytr, Zte, *, hidden=64, epochs=400, lr=1e-3, wd=1e-2,
            seed=0, val_frac=0.2):
    torch.manual_seed(seed)
    n = len(ytr)
    perm = np.random.RandomState(seed).permutation(n)
    nval = max(1, int(n * val_frac))
    vi, ti = perm[:nval], perm[nval:]
    Xt = torch.tensor(Ztr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    model = nn.Sequential(nn.Linear(Ztr.shape[1], hidden), nn.ReLU(),
                          nn.Linear(hidden, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xt[ti]).squeeze(1), yt[ti])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            sv = model(Xt[vi]).squeeze(1).numpy()
        a = auc(list(sv), list(ytr[vi]))
        if a is not None and a > best + 1e-4:
            best, bad = a, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 40:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ste = model(torch.tensor(Zte, dtype=torch.float32)).squeeze(1).numpy()
        sv = model(Xt[vi]).squeeze(1).numpy()
    return ste, sv, ytr[vi]


def rff_ridge(Ztr, ytr, Zte, *, nfeat=512, seed=0):
    rng = np.random.RandomState(seed)
    d = Ztr.shape[1]
    W = rng.normal(0.0, np.sqrt(2.0 / d),
                   size=(d, nfeat // 2)).astype(np.float64)
    b = rng.uniform(0.0, 2.0 * np.pi, size=nfeat // 2)

    def feats(X):
        P = X @ W + b
        return np.concatenate([np.cos(P), np.sin(P)], axis=1) * np.sqrt(2.0 / nfeat)

    m = ridge_fit(feats(Ztr), ytr)
    return ridge_score(feats(Zte), m), ridge_score(feats(Ztr), m)


def operational(s_tr, y_tr, s_te, y_te, frac=0.10):
    if (y_tr == 1).sum() == 0:
        return None
    thr = float(np.quantile(s_tr[y_tr == 1], frac))
    pred_bad = s_te < thr
    return {"threshold": thr,
            "caught_wrong": int(np.sum(pred_bad & (y_te == 0))),
            "false_cancels": int(np.sum(pred_bad & (y_te == 1))),
            "wrong": int((y_te == 0).sum()),
            "correct": int((y_te == 1).sum())}


def gru_eval(H, lengths, y, splits, horizon, *, npoints=8, hidden=32, pca=32,
             epochs=300, seeds=(0, 1, 2)):
    idx = np.where(lengths > horizon)[0]
    pos = np.stack([np.linspace(0, min(int(lengths[i]) - 1, horizon),
                                npoints).astype(int) for i in idx])
    Xs = H[idx[:, None], pos]                      # [n, T, n_embd]
    yy = y[idx]
    tr = np.array([j for j in range(len(idx)) if splits[idx[j]] == "train"])
    te = np.array([j for j in range(len(idx)) if splits[idx[j]] == "test"])
    flat = Xs[tr].reshape(-1, Xs.shape[2])
    mu = flat.mean(0, keepdims=True)
    sd = flat.std(0, keepdims=True) + 1e-6
    Z = (Xs - mu) / sd
    Str = Z[tr].reshape(-1, Z.shape[2])
    Str = Str - Str.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Str, full_matrices=False)
    V = Vt[:pca].T
    P = Z @ V
    fmu = P[tr].reshape(-1, pca).mean(0, keepdims=True)
    fsd = P[tr].reshape(-1, pca).std(0, keepdims=True) + 1e-6
    P = (P - fmu) / fsd
    Xt = torch.tensor(P, dtype=torch.float32)
    yt = torch.tensor(yy, dtype=torch.float32)
    out_aucs = []
    for seed in seeds:
        torch.manual_seed(seed)
        perm = np.random.RandomState(seed).permutation(len(tr))
        nval = max(1, int(0.2 * len(tr)))
        vi, ti = perm[:nval], perm[nval:]
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
            a = auc(list(sv), list(yy[vi]))
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
            o, _ = gru(Xt[te])
            s = head(o[:, -1]).squeeze(1).numpy()
        out_aucs.append(auc(list(s), list(yy[te])))
    return {"n": int(len(idx)), "n_test": int(len(te)),
            "aucs": [round(a, 4) for a in out_aucs],
            "auc_mean": float(np.mean(out_aucs))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=Path,
                    default=CASCADE / "streams" / "scion-bench-monitor")
    ap.add_argument("--out", type=Path, default=R / "monitor-nonlinear.json")
    args = ap.parse_args()

    t0 = time.time()
    H, lengths, y, splits = load_aligned(args.windows)
    print(f"loaded {H.shape} (numeric order), {time.time()-t0:.0f}s", flush=True)

    specs = {}
    for h in (64, 128, 256, 512):
        specs[f"start{h}"] = (np.where(lengths > h)[0],
                              lambda i, h=h: H[i, h])
    specs["last8"] = (np.where(lengths > 32)[0],
                      lambda i: H[i, lengths[i]-8:lengths[i]].mean(0))
    specs["last32"] = (np.where(lengths > 32)[0],
                       lambda i: H[i, max(0, lengths[i]-32):lengths[i]].mean(0))
    specs["mid"] = (np.where(lengths > 16)[0],
                    lambda i: H[i, max(0, lengths[i]//2-4):
                                lengths[i]//2+4].mean(0))

    res = {"alignment": "numeric win_%02d order", "positions": {}}
    for name, (idx, feat) in specs.items():
        try:
            X = np.stack([feat(i) for i in idx]).astype(np.float64)
            yy = y[idx]
            tr = [j for j in range(len(idx)) if splits[idx[j]] == "train"]
            te = [j for j in range(len(idx)) if splits[idx[j]] == "test"]
            Xtr, Xte = X[tr], X[te]
            ytr, yte = yy[tr], yy[te]
            Ztr, Zte = standardize(Xtr, Xte)
            m = ridge_fit(Xtr, ytr)
            row = {"n": int(len(idx)), "n_test": len(te),
                   "ridge": {"auc": auc(list(ridge_score(Xte, m)), list(yte))}}
            Ptr, Pte = pca_reduce(Ztr, Zte, k=128)
            Ptr_s, Pte_s = standardize(Ptr, Pte)
            mlp_te, mlp_vs, mlp_vy = fit_mlp(Ptr_s, ytr, Pte_s)
            row["mlp_pca128"] = {
                "auc": auc(list(mlp_te), list(yte)),
                "operational_val_thr": operational(mlp_vs, mlp_vy, mlp_te, yte)}
            rff_te, rff_tr = rff_ridge(Ztr, ytr, Zte)
            row["rff_ridge"] = {
                "auc": auc(list(rff_te), list(yte)),
                "operational_train_thr": operational(rff_tr, ytr, rff_te, yte)}
            res["positions"][name] = row
            print(f"{name:>8}: n={row['n']:>3} te={len(te):>3} | "
                  f"ridge {row['ridge']['auc']:.3f} | "
                  f"mlp {row['mlp_pca128']['auc']:.3f} | "
                  f"rff {row['rff_ridge']['auc']:.3f}", flush=True)
        except Exception as exc:  # keep going; record the failure
            res["positions"][name] = {"error": repr(exc)}
            print(f"{name:>8}: FAILED {exc!r}", flush=True)

    for horizon in (64, 256):
        try:
            res[f"gru_start{horizon}"] = gru_eval(H, lengths, y, splits, horizon)
            r = res[f"gru_start{horizon}"]
            print(f"gru_start{horizon}: n={r['n']} te={r['n_test']} "
                  f"aucs={r['aucs']} mean={r['auc_mean']:.3f}", flush=True)
        except Exception as exc:
            res[f"gru_start{horizon}"] = {"error": repr(exc)}
            print(f"gru_start{horizon}: FAILED {exc!r}", flush=True)

    args.out.write_text(json.dumps(res, indent=1))
    print("wrote", args.out, f"({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
