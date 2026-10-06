#!/usr/bin/env python
"""Monitor diagnostic, token level: argmax stability / switch rate per stream.

The per-position argmax dumps from `test-mtp-probe` (int32 token ids) are a
free trajectory feature set: how repetitive the tail is (stability) and how
often the prediction changes (switch rate).  Fits a small ridge on these
features and reports held-out AUC for final correctness.

Usage:
    monitor_token_diag.py --windows experiments/cascade/streams/scion-bench-monitor
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=Path,
                    default=CASCADE / "streams" / "scion-bench-monitor")
    ap.add_argument("--out", type=Path, default=R / "monitor-token-diag.json")
    args = ap.parse_args()

    meta = json.loads((args.windows / "meta.json").read_text())
    order, lengths, labels = meta["order"], meta["lengths"], meta["labels"]
    splits = [meta["tasks"][o.rsplit(":", 1)[0]] for o in order]
    y = np.asarray(labels, dtype=np.float64)
    A = np.stack([np.fromfile(p, dtype=np.int32)[:meta["seq"]] for p in
                  sorted((args.windows / "dump").glob("win_*_argmax.bin"))])

    def features(W: int):
        st, sw = [], []
        for i, length in enumerate(lengths):
            seg = A[i, max(0, length - W):length]
            vals, counts = np.unique(seg, return_counts=True)
            st.append(counts.max() / len(seg))
            sw.append(float((np.diff(seg) != 0).mean()))
        return np.asarray(st), np.asarray(sw)

    tr = [i for i, s in enumerate(splits) if s == "train"]
    te = [i for i, s in enumerate(splits) if s == "test"]
    out = {}
    for W in (32, 64, 128, 256):
        st, sw = features(W)
        out[f"W{W}"] = {
            "auc_stability": auc(list(st[te]), list(y[te])),
            "auc_switch": auc(list(sw[te]), list(y[te])),
        }
    st, sw = features(64)
    X = np.stack([st, sw, st * sw], axis=1)
    model = ridge_fit(X[tr], y[tr])
    s = ridge_score(X, model)
    out["combined_W64"] = {"auc": auc(list(s[te]), list(y[te])),
                           "features": ["stability", "switch", "product"]}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
