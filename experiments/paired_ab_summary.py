"""Summarise a paired-A/B run - finished or not.

The paired A/B writes a checkpoint as it goes, so a run that is still
going can be summarised.  That matters for the *common prefix* read: the
fixture conversations are ordered, so two arms that ran over the same
sample cover the same tasks in the same order, and the prefix they share
is a valid paired comparison even if one arm stopped early.

Usage::

    python -m experiments.paired_ab_summary <run-dir> [<run-dir> ...]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Optional


def load(run_dir: Path) -> list[dict]:
    """The rows of a run (the final file if it is there, else the checkpoint)."""
    for name in ("paired_ab.json", "paired_ab.ckpt.json"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows = data.get("rows")
        if rows:
            return rows
    return []


def summarize(rows: list[dict]) -> dict:
    """The paired A/B's own metrics over a set of rows."""
    if not rows:
        return {"n": 0}
    out: dict = {"n": len(rows)}
    measured = [r for r in rows if r.get("answer_hive_hit_ratio") is not None]
    out["measured"] = len(measured)
    for arm in ("hive", "fifo"):
        hits = [r.get(f"answer_{arm}_hit_ratio") for r in measured]
        hits = [h for h in hits if h is not None]
        out[f"{arm}_recall"] = round(100.0 * statistics.mean(hits), 2) if hits else None
        fid = [r.get(f"answer_{arm}_suff") for r in measured]
        fid = [f for f in fid if f is not None]
        out[f"{arm}_fidelity"] = round(statistics.mean(fid), 4) if fid else None
        tok = [r.get(f"{arm}_ctx_tokens") for r in rows]
        tok = [t for t in tok if t is not None]
        out[f"{arm}_ctx_tokens"] = round(statistics.mean(tok), 1) if tok else None
    # the paired comparison: the turns where one arm is strictly better
    hive_only = fifo_only = both = neither = 0
    for r in measured:
        h = r.get("answer_hive_hit_ratio")
        f = r.get("answer_fifo_hit_ratio")
        if h is None or f is None:
            continue
        if h > f:
            hive_only += 1
        elif f > h:
            fifo_only += 1
        elif h == f and h >= 0.5:
            both += 1
        else:
            neither += 1
    out["hive_only"] = hive_only
    out["fifo_only"] = fifo_only
    out["both"] = both
    out["neither"] = neither
    out["hive_ge_fifo"] = round(
        100.0 * (hive_only + both) / max(1, hive_only + fifo_only + both + neither), 2)
    # the fidelity comparison (the arm whose answer cited more of its own context)
    fav = [r for r in measured
           if r.get("answer_hive_suff") is not None
           and r.get("answer_fifo_suff") is not None]
    if fav:
        out["fidelity_hive_wins"] = sum(
            1 for r in fav if r["answer_hive_suff"] > r["answer_fifo_suff"])
        out["fidelity_turns"] = len(fav)
        out["fidelity_hive_share"] = round(
            100.0 * out["fidelity_hive_wins"] / len(fav), 2)
    # the tasks covered (the prefix)
    tasks = sorted({r.get("conversation_id") or r.get("conv_id") or "" for r in rows})
    out["tasks"] = len([t for t in tasks if t])
    return out


def main(argv: Optional[list[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print(__doc__)
        return 2
    summaries = {}
    for raw in args:
        run_dir = Path(raw)
        rows = load(run_dir)
        s = summarize(rows)
        summaries[run_dir.name] = s
        print(f"=== {run_dir.name}: {s.get('n')} rows, {s.get('tasks')} conversations")
        for key in ("measured", "hive_recall", "fifo_recall", "hive_fidelity",
                    "fifo_fidelity", "hive_ge_fifo", "hive_only", "fifo_only",
                    "both", "neither", "fidelity_hive_share", "fidelity_turns"):
            if key in s:
                print(f"  {key:22} {s[key]}")
    if len(summaries) >= 2:
        names = list(summaries)
        a, b = summaries[names[0]], summaries[names[1]]
        common = min(a.get("n", 0), b.get("n", 0))
        print(f"\ncommon prefix: {common} rows")
        print(f"  {names[0]}: hive {a.get('hive_recall')}% vs fifo {a.get('fifo_recall')}% "
              f"(hive>=fifo {a.get('hive_ge_fifo')}%)")
        print(f"  {names[1]}: hive {b.get('hive_recall')}% vs fifo {b.get('fifo_recall')}% "
              f"(hive>=fifo {b.get('hive_ge_fifo')}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
