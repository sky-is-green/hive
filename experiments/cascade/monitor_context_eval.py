#!/usr/bin/env python
"""Trace-relative monitor evaluation on generation-context windows.

Same readouts as monitor_nonlinear.py (static ridge + GRU over 8 trajectory
snapshots), but positions are addressed relative to the generation prefix
(prompt_len) so the monitor only sees the trace region, in the generation
context rather than the read-back context.

Emits e.g. monitor-context-scion.json / monitor-context-qwen.json.
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

torch.set_num_threads(4)


def load_ctx(windows: Path):
    meta = json.loads((windows / "meta.json").read_text())
    order = meta["order"]
    splits = np.array([meta["tasks"][o.rsplit(":", 1)[0]] for o in order])
    files = sorted((windows / "dump").glob("win_*_h.bin"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if len(files) != len(order):
        raise SystemExit(f"{len(files)} dumps for {len(order)} streams")
    H = np.stack([np.fromfile(p, dtype=np.float32).reshape(meta["seq"], -1)
                  for p in files])
    return (H, np.asarray(meta["lengths"]), np.asarray(meta["labels"], float),
            splits, np.asarray(meta["prompt_len"]),
            np.asarray(meta.get("truncated", [False] * len(order))))


def gru_eval(H, idx, pos, y, splits, *, hidden=32, pca=32, epochs=300,
             seeds=(0, 1, 2)):
    Xs = H[idx[:, None], pos]                      # [n, T, n_embd]
    yy = y[idx]
    tr = np.array([j for j in range(len(idx)) if splits[idx[j]] == "train"])
    te = np.array([j for j in range(len(idx)) if splits[idx[j]] == "test"])
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
    aucs, ops = [], []
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
            o, _ = gru(Xt[vi])
            sv = head(o[:, -1]).squeeze(1).numpy()
            o, _ = gru(Xt[te])
            st = head(o[:, -1]).squeeze(1).numpy()
        aucs.append(auc(list(st), list(yy[te])))
        if (yy[vi] == 1).sum():
            thr = float(np.quantile(sv[yy[vi] == 1], 0.10))
            pb = st < thr
            ops.append({"caught_wrong": int(np.sum(pb & (yy[te] == 0))),
                        "false_cancels": int(np.sum(pb & (yy[te] == 1))),
                        "wrong": int((yy[te] == 0).sum()),
                        "correct": int((yy[te] == 1).sum())})
    return {"n": int(len(idx)), "n_test": int(len(te)),
            "aucs": [round(a, 4) for a in aucs],
            "auc_mean": float(np.mean(aucs)),
            "ops_val10": ops}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()
    H, lengths, y, splits, plen, trunc = load_ctx(args.windows)
    print(f"loaded {H.shape}, prompt_len max {int(plen.max())}, "
          f"{time.time()-t0:.0f}s", flush=True)

    specs = {}
    for h in (64, 128, 256):
        idx = np.where(lengths - plen > h)[0]
        pos = (plen[idx][:, None]
               + np.linspace(0, h, 8).astype(int)[None, :])
        specs[f"start{h}"] = (idx, pos, plen[idx] + h)
    idx = np.where((~trunc) & (lengths > 8))[0]
    specs["last8"] = (idx,
                      np.stack([np.arange(lengths[i] - 8, lengths[i])
                                for i in idx]), None)

    res = {"windows": str(args.windows), "positions": "trace-relative",
           "horizons": {}}
    for name, (idx, pos, static_abs) in specs.items():
        if len(idx) < 30:
            print(f"{name:>8}: skipped ({len(idx)} streams)", flush=True)
            continue
        try:
            row: dict = {"n": int(len(idx))}
            if static_abs is not None:
                X = H[idx, static_abs].astype(np.float64)
                yy = y[idx]
                tr = [j for j in range(len(idx))
                      if splits[idx[j]] == "train"]
                te = [j for j in range(len(idx))
                      if splits[idx[j]] == "test"]
                m = ridge_fit(X[tr], yy[tr])
                row["ridge"] = {
                    "auc": auc(list(ridge_score(X[te], m)), list(yy[te])),
                    "n_test": len(te)}
            g = gru_eval(H, idx, pos, y, splits)
            row["gru"] = g
            res["horizons"][name] = row
            r_a = row.get("ridge", {}).get("auc", float("nan"))
            print(f"{name:>8}: n={row['n']} te={g['n_test']} | "
                  f"ridge {r_a:.3f} | gru aucs={g['aucs']} "
                  f"mean={g['auc_mean']:.3f}", flush=True)
        except Exception as exc:
            res["horizons"][name] = {"error": repr(exc)}
            print(f"{name:>8}: FAILED {exc!r}", flush=True)

    args.out.write_text(json.dumps(res, indent=1))
    print("wrote", args.out, f"({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
