#!/usr/bin/env python
"""Monitor diagnostic: where in the stream does final correctness become readable?

Fits the same ridge readout at positions measured both from the stream start
(prefix horizons) and from the stream end (answer region).  Answers: is the
early-abort idea dead, or is the signal only present near the answer (which
would bound token savings)?

Usage:
    monitor_diag.py --windows experiments/cascade/streams/scion-bench-monitor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from probe_judge import ridge_fit, ridge_score  # noqa: E402
from readout_judge import auc  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"


def eval_pos(H, lengths, y, splits, pos_fn, pool=1):
    idx = [i for i in range(len(lengths)) if pos_fn(int(lengths[i])) is not None]
    if len(idx) < 30:
        return None
    rows = []
    for i in idx:
        t = pos_fn(int(lengths[i]))
        lo = max(0, t - pool + 1)
        rows.append(H[i, lo:t + 1].mean(axis=0))
    X = np.stack(rows).astype(np.float64)
    y2 = y[idx]
    sp = [splits[i] for i in idx]
    tr = [j for j, s in enumerate(sp) if s == "train"]
    te = [j for j, s in enumerate(sp) if s == "test"]
    if len(tr) < 10 or not te:
        return None
    m = ridge_fit(X[tr], y2[tr])
    s = ridge_score(X, m)
    return {"n": len(idx), "n_test": len(te),
            "auc": auc(list(s[te]), list(y2[te]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=Path,
                    default=CASCADE / "streams" / "scion-bench-monitor")
    ap.add_argument("--out", type=Path, default=R / "monitor-diag.json")
    args = ap.parse_args()

    meta = json.loads((args.windows / "meta.json").read_text())
    order, lengths, labels = meta["order"], meta["lengths"], meta["labels"]
    splits = [meta["tasks"][o.rsplit(":", 1)[0]] for o in order]
    dump_paths = sorted((args.windows / "dump").glob("win_*_h.bin"),
                        key=lambda p: int(p.stem.split("_")[1]))
    first = np.fromfile(dump_paths[0], dtype=np.float32)
    dim = first.size // meta["seq"]
    H = np.stack([np.fromfile(p, dtype=np.float32).reshape(meta["seq"], dim)
                  for p in dump_paths])
    y = np.asarray(labels, dtype=np.float64)
    out = {"from_start": {}, "from_end": {}}
    for t in (64, 128, 192, 256, 384, 512):
        r = eval_pos(H, lengths, y, splits, lambda L, t=t: t if L > t else None)
        if r:
            out["from_start"][str(t)] = r
            print(f"start+{t:>4}: AUC {r['auc']:.3f} (n={r['n']})", flush=True)
    for k in (0, 8, 16, 32, 64, 128, 256):
        r = eval_pos(H, lengths, y, splits, lambda L, k=k: L - 1 if L > k else None,
                     pool=min(k + 1, 33))
        if r:
            out["from_end"][f"last{max(k,1)}"] = r
            print(f"end-{k:>4}: AUC {r['auc']:.3f} (n={r['n']}, "
                  f"pool {min(k+1,33)})", flush=True)
    # midpoint
    r = eval_pos(H, lengths, y, splits, lambda L: L // 2 if L > 8 else None)
    if r:
        out["midpoint"] = r
        print(f"midpoint: AUC {r['auc']:.3f} (n={r['n']})", flush=True)
    args.out.write_text(json.dumps(out, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
