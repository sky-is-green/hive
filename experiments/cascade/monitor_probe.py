#!/usr/bin/env python
"""Prefix monitor heads on per-position hidden states (L3).

Loads the `test-mtp-probe` dump (``win_XX_h.bin`` float32 [seq, n_embd]) and
the matching meta (order/lengths/labels), fits a ridge at each horizon on
train-task streams, and evaluates on held-out tasks: AUC for final
correctness, plus the catch/false-cancel trade-off.  This is the "knows it's
failing before it finishes" signal that gates early abort.

Runtime prediction = score below a threshold (low predicted correctness).
The threshold is set on train at a fixed false-cancel budget (flag at most
10% of correct streams) and applied unchanged to the test split.

Usage:
    monitor_probe.py --windows experiments/cascade/streams/scion-bench-monitor
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
HORIZONS = (64, 128, 192, 256, 384, 512)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=Path,
                    default=CASCADE / "streams" / "scion-bench-monitor")
    ap.add_argument("--out", type=Path, default=R / "monitor-probe.json")
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    args = ap.parse_args()

    meta = json.loads((args.windows / "meta.json").read_text())
    order, lengths, labels = meta["order"], meta["lengths"], meta["labels"]
    splits = [meta["tasks"][o.rsplit(":", 1)[0]] for o in order]
    dumps = sorted((args.windows / "dump").glob("win_*_h.bin"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if len(dumps) != len(order):
        raise SystemExit(f"{len(dumps)} dump windows for {len(order)} streams")
    H = np.stack([np.fromfile(p, dtype=np.float32).reshape(meta["seq"], -1)
                  for p in dumps])
    y = np.asarray(labels, dtype=np.float64)
    lengths = np.asarray(lengths)
    print(f"states {H.shape} labels {int(y.sum())}/{len(y)}", flush=True)

    out = {"n": len(order), "horizons": {}}
    for horizon in [int(x) for x in args.horizons.split(",")]:
        mask = lengths > horizon
        if mask.sum() < 30:
            continue
        sp = [splits[i] for i in range(len(order)) if mask[i]]
        tr = [j for j, s in enumerate(sp) if s == "train"]
        te = [j for j, s in enumerate(sp) if s == "test"]
        if len(tr) < 10 or not te:
            continue
        X = H[:, horizon, :][mask].astype(np.float64)
        yy = y[mask]
        model = ridge_fit(X[tr], yy[tr])
        s = ridge_score(X, model)
        s_te, y_te = s[te], yy[te]
        # threshold: flag at most 10% of train-correct streams
        s_tr, y_tr = s[tr], yy[tr]
        thr = float(np.quantile(s_tr[y_tr == 1], 0.10)) if (y_tr == 1).any() else 0.0
        pred_bad = s_te < thr
        caught = int(np.sum(pred_bad & (y_te == 0)))
        fc = int(np.sum(pred_bad & (y_te == 1)))
        out["horizons"][str(horizon)] = {
            "n_eval": int(mask.sum()),
            "n_test": len(te),
            "test_auc": auc(list(s_te), list(y_te)),
            "threshold": thr,
            "caught_wrong": caught,
            "false_cancels": fc,
            "wrong": int((y_te == 0).sum()),
            "correct": int((y_te == 1).sum()),
        }
        h = out["horizons"][str(horizon)]
        print(f"horizon {horizon:>4}: test AUC {h['test_auc']:.3f}  "
              f"catches {caught}/{h['wrong']} wrong, {fc} false cancels "
              f"of {h['correct']} correct", flush=True)
    args.out.write_text(json.dumps(out, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
