#!/usr/bin/env python
"""Fit a router threshold on train, report it on the held-out test split.

Consumes two ``router_eval.py`` artifacts (which carry per-task
``p_local_correct``, ``scion_ok`` and ``api_ok``) and applies one operating
rule: maximise policy quality, then minimise API share, then take the highest
threshold of the tie (the most escalation headroom).  The chosen threshold is
evaluated on the test artifact without re-fitting.

    .venv/bin/python experiments/cascade/bench_refit.py \
        --train-eval results/strata-bench-router-train.json \
        --test-eval results/strata-bench-router-test.json \
        --out results/strata-bench-refit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]

RULE = (
    "max train quality; tie -> min train api share; "
    "tie -> highest threshold (escalation headroom)"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sweep(rows: list[dict[str, Any]], step: int = 5) -> list[dict[str, Any]]:
    scorable = [r for r in rows if r.get("p_local_correct") is not None]
    out: list[dict[str, Any]] = []
    for percent in range(step, 100, step):
        threshold = percent / 100.0
        routed = [bool(r["p_local_correct"] >= threshold) for r in scorable]
        with_api = [
            (rl, r) for rl, r in zip(routed, scorable)
            if r.get("api_ok") is not None
        ]
        out.append(
            {
                "threshold": threshold,
                "quality": (
                    round(sum(
                        (r["scion_ok"] if rl else r["api_ok"]) for rl, r in with_api
                    ) / len(with_api), 4)
                    if with_api else None
                ),
                "api_share": round(sum(not rl for rl in routed) / len(routed), 4)
                if routed else None,
                "routing_accuracy": round(
                    sum(bool(rl) == bool(r["scion_ok"])
                        for rl, r in zip(routed, scorable)) / len(scorable), 4
                ) if scorable else None,
            }
        )
    return out


def choose(train_sweep: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    candidates = [row for row in train_sweep if row["quality"] is not None]
    if not candidates:
        return None
    best_quality = max(row["quality"] for row in candidates)
    quality = [row for row in candidates if row["quality"] == best_quality]
    best_share = min(row["api_share"] for row in quality)
    quality = [row for row in quality if row["api_share"] == best_share]
    return max(quality, key=lambda row: row["threshold"])


def evaluate(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scored = [r for r in rows if r.get("p_local_correct") is not None]
    routed = [bool(r["p_local_correct"] >= threshold) for r in scored]
    with_api = [(rl, r) for rl, r in zip(routed, scored) if r.get("api_ok") is not None]
    return {
        "threshold": threshold,
        "tasks": len(scored),
        "quality": (
            round(sum((r["scion_ok"] if rl else r["api_ok"]) for rl, r in with_api)
                  / len(with_api), 4)
            if with_api else None
        ),
        "api_share": round(sum(not rl for rl in routed) / len(routed), 4)
        if routed else None,
        "routing_accuracy": round(
            sum(bool(rl) == bool(r["scion_ok"]) for rl, r in zip(routed, scored))
            / len(scored), 4
        ) if scored else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-eval", required=True)
    parser.add_argument("--test-eval", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train_path, test_path = Path(args.train_eval), Path(args.test_eval)
    train = json.loads(train_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    train_sweep = sweep(train["rows"])
    test_sweep = sweep(test["rows"])
    chosen = choose(train_sweep)
    if chosen is None:
        raise SystemExit("no scorable rows in the train evaluation")

    artifact = {
        "rule": RULE,
        "inputs": {
            "train_eval": str(train_path),
            "train_sha256": sha256(train_path),
            "test_eval": str(test_path),
            "test_sha256": sha256(test_path),
        },
        "train": {
            "tasks": len(train["rows"]),
            "sources": train.get("summary", {}).get("sources"),
            "sweep": train_sweep,
            "chosen": chosen,
        },
        "test": {
            "tasks": len(test["rows"]),
            "sources": test.get("summary", {}).get("sources"),
            "sweep": test_sweep,
            "chosen": evaluate(test["rows"], chosen["threshold"]),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1), encoding="utf-8")

    print(f"rule: {RULE}")
    print(f"chosen on train: threshold {chosen['threshold']:.2f} "
          f"quality {chosen['quality']:.3f} api {chosen['api_share']:.3f}")
    t = artifact["test"]["chosen"]
    print(f"reported on test: quality {t['quality']:.3f} api {t['api_share']:.3f} "
          f"routing accuracy {t['routing_accuracy']:.3f} over {t['tasks']} tasks")
    print("artifact:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
