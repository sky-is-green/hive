#!/usr/bin/env python
"""Unified comparison of every J-Bench result: readouts, probes, judges.

Scans the results directory for `readout-*.json`, `probe-*.json`, and
`judge-*.json`, re-scores each on the frozen **test** split with the same
metrics, and writes `COMPARISON.md`.

Usage:
    python experiments/cascade/compare_readouts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from readout_judge import metrics  # noqa: E402

R = REPO / "experiments/cascade/results/jev-readout-20261006"


def load_items(path: Path) -> list[dict] | None:
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    items = data.get("items")
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        return None
    if "p_correct" not in items[0]:
        return None
    return items


def row(name: str, kind: str, items: list[dict]) -> str:
    test = [i for i in items if i.get("split") == "test"]
    if not test:
        return ""
    m = metrics(test)
    nl = m.get("no_leak") or {}
    acc = m["at_0.5"]["correct"] / max(1, m["n"])
    return (f"| {name} | {kind} | {m['n']} | {m['auc']:.3f} | "
            f"{m['best_ba']['bal_acc']:.3f} | {acc:.3f} | "
            f"{m['at_0.5']['false_accepts']}/{m['at_0.5']['false_rejects']} | "
            f"{nl.get('threshold', float('nan')):.2f} | "
            f"{nl.get('correct', 0)/max(1, nl.get('n', 1)):.3f} | "
            f"{m['ece']:.3f} |")


def main() -> int:
    lines = [
        "# J-Bench comparison (frozen test split)",
        "",
        "| readout | kind | n | AUC | best BA | acc@0.5 | FA/FR@0.5 | no-leak thr | no-leak acc | ECE |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for path in sorted(R.glob("readout-*.json")):
        if "smoke" in path.stem:
            continue
        items = load_items(path)
        if items:
            lines.append(row(path.stem.replace("readout-", ""), "prompted", items))
    for path in sorted(R.glob("judge-*.json")):
        if "smoke" in path.stem:
            continue
        items = load_items(path)
        if items:
            lines.append(row(path.stem.replace("judge-", ""), "trained-judge", items))
    for path in sorted(R.glob("probe-*.json")):
        if "smoke" in path.stem:
            continue
        data = json.loads(path.read_text())
        states_path = R / data.get("state_file", "")
        if not states_path.exists():
            continue
        import numpy as np
        from probe_judge import ridge_fit, ridge_score, platt
        z = np.load(states_path, allow_pickle=True)
        ids = [str(x) for x in z["item_ids"]]
        labels = [int(x) for x in z["labels"]]
        splits = [str(x) for x in z["splits"]]
        base = path.stem.replace("probe-", "")
        for f in data.get("layers", {}):
            X = z[f"L{f}"].astype(np.float64)
            y = np.asarray(labels, dtype=np.float64)
            tr = [i for i, s in enumerate(splits) if s == "train"]
            te = [i for i, s in enumerate(splits) if s == "test"]
            model = ridge_fit(X[tr], y[tr])
            s_tr = ridge_score(X[tr], model)
            a, b = platt(s_tr, y[tr])
            s = ridge_score(X, model)
            import math
            items = []
            for i in te:
                p = 1.0 / (1.0 + math.exp(-(a * s[i] + b)))
                items.append({"split": "test", "label": labels[i],
                              "p_correct": p})
            lines.append(row(f"{base} depth {f}", "probe", items))
    out = "\n".join(l for l in lines if l) + "\n"
    (R / "COMPARISON.md").write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
