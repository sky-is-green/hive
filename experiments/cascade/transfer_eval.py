#!/usr/bin/env python
"""Cross-body probe transfer: fit on A's train states, score B's test states.

Answers "does the decision signal survive ternarization?" at the final layer:
a head fitted on f16 states applied frozen to ternary states, and vice versa.
States come from `probe_scion.py` (same prompts, same readout position).

Usage:
    transfer_eval.py --a probe-states-clef-f16.npz --b probe-states-clef-qat.npz
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
from readout_judge import auc, best_ba  # noqa: E402

R = REPO / "experiments/cascade/results/jev-readout-20261006"


def load(path: Path):
    z = np.load(path, allow_pickle=True)
    return {
        "ids": [str(x) for x in z["item_ids"]],
        "labels": np.asarray([int(x) for x in z["labels"]]),
        "splits": [str(x) for x in z["splits"]],
        "X": z["L1"].astype(np.float64),
    }


def fit_eval(src: dict, dst: dict) -> dict:
    tr = [i for i, s in enumerate(src["splits"]) if s == "train"]
    model = ridge_fit(src["X"][tr], src["labels"][tr])
    # dst test items (same ordering as src: ids are checked in main)
    dst_test = [i for i, s in enumerate(dst["splits"]) if s == "test"]
    s = ridge_score(dst["X"][dst_test], model)
    y = dst["labels"][dst_test]
    return {"n": int(len(y)), "auc": auc(list(s), list(y)),
            "best_ba": best_ba(list(s), list(y))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", type=Path, default=R / "transfer.json")
    args = ap.parse_args()

    A = load(args.out.parent / args.a if not Path(args.a).is_absolute() else Path(args.a))
    B = load(args.out.parent / args.b if not Path(args.b).is_absolute() else Path(args.b))
    if A["ids"] != B["ids"]:
        raise SystemExit("state files are over different item sets")
    out = {
        "a": Path(args.a).stem, "b": Path(args.b).stem,
        "fit_a_eval_a": fit_eval(A, A),
        "fit_a_eval_b": fit_eval(A, B),
        "fit_b_eval_b": fit_eval(B, B),
        "fit_b_eval_a": fit_eval(B, A),
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
