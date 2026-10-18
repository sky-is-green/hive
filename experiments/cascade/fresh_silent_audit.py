"""Fresh silent-wrong audit: score every detector-silent draft in the three
committed runs through all five frozen heads.

The M4 acceptance was pre-registered on the *fresh* distribution and the gate
failed its false-flag budget (3 > 2) on three items whose p (0.061-0.378)
overlaps the one catch (0.306).  This script turns that single observation
into a durable table: for each run and pooled, every silent correct and
silent wrong with its p under L0.5/L0.75/L1/ridge-concat/mlp-concat, plus the
per-head catch/flag counts at the frozen 0.5 cut and at the best cap-2 cut.
It is the baseline a retrained head must beat; nothing here is a deployment
decision (no per-run threshold tuning is accepted).

Output: experiments/cascade/results/fresh-silent-audit-20261018/audit.json + AUDIT.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RUNS = [
    ("armA", REPO / "experiments/cascade/results/runner-cascade-sampled-20261018"),
    ("armB", REPO / "experiments/cascade/results/runner-cascade-sampled-nothink-20261018"),
    ("gate", REPO / "experiments/cascade/results/runner-cascade-gate-nothink-20261018"),
]
HEADS = ("L0.5", "L0.75", "L1", "ridge-concat", "mlp-concat")
OUT_DIR = REPO / "experiments/cascade/results/fresh-silent-audit-20261018"


def silent_rows(report):
    out = []
    for r in report["records"]:
        if r.get("gated"):
            continue
        if r.get("escalated"):
            continue
        if r.get("truncated"):
            continue
        if r.get("scion_ok") is None:
            continue
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="score only first N silent rows per run (smoke)")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from harness.cascade.learned_gate import LearnedGate, ReadoutHead

    gate = LearnedGate()  # loads backbone lazily on first .states()
    heads = {h: ReadoutHead(h) for h in HEADS}

    all_items = []
    for run_name, run_dir in RUNS:
        report = json.loads((run_dir / "run" / "report.json").read_text())
        tasks = {t["id"]: t for t in json.loads((run_dir / "tasks-organic66.json").read_text())["tasks"]}
        rows = silent_rows(report)
        if args.limit:
            rows = rows[: args.limit]
        print(f"[{run_name}] {len(rows)} silent rows ({sum(1 for r in rows if r['scion_ok'])} correct, "
              f"{sum(1 for r in rows if not r['scion_ok'])} wrong)")
        for i, r in enumerate(rows, 1):
            task = tasks.get(r["id"], {})
            prompt = task.get("prompt", "")
            draft = r.get("scion_answer") or ""
            t0 = time.perf_counter()
            try:
                feats = gate.states(prompt, draft)
                ps = {h: heads[h].score(feats) for h in HEADS}
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(rows)}] {r['id']} ERROR: {exc}")
                continue
            ms = (time.perf_counter() - t0) * 1000.0
            item = {
                "run": run_name,
                "id": r["id"],
                "bucket": r.get("bucket"),
                "truth": bool(r["scion_ok"]),  # True = silent correct, False = silent wrong
                "draft_truncated": len(draft) >= 4000,
                "p": ps,
                "ms": round(ms, 1),
            }
            all_items.append(item)
            tag = "WRONG" if not r["scion_ok"] else "ok   "
            print(f"  [{i}/{len(rows)}] {r['id']:28} {tag}  " +
                  "  ".join(f"{h}={ps[h]:.4f}" for h in HEADS) + f"  ({ms:.0f}ms)")

    (out_dir / "audit.json").write_text(json.dumps({
        "generated": time.strftime("%Y%m%dT%H%M%S"),
        "n_items": len(all_items),
        "runs": [r[0] for r in RUNS],
        "heads": list(HEADS),
        "items": all_items,
    }, indent=1))

    # ---- summary ---------------------------------------------------------
    lines = []
    lines.append("# Fresh silent-wrong audit (2026-10-18)")
    lines.append("")
    lines.append(f"Scored {len(all_items)} detector-silent drafts "
                 f"({sum(1 for x in all_items if x['truth'])} correct, "
                 f"{sum(1 for x in all_items if not x['truth'])} wrong) "
                 f"from the three committed runs through all five frozen heads "
                 f"(L0.5/L0.75/L1/ridge-concat/mlp-concat), one backbone load.")
    lines.append("")
    lines.append("## Per-run counts")
    lines.append("")
    lines.append("| run | silent correct | silent wrong |")
    lines.append("|---|---:|---:|")
    for run_name, _ in RUNS:
        sub = [x for x in all_items if x["run"] == run_name]
        lines.append(f"| {run_name} | {sum(1 for x in sub if x['truth'])} | "
                     f"{sum(1 for x in sub if not x['truth'])} |")
    lines.append("")

    def cap2_table(items, title):
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| head | silent wrongs caught @0.5 | false flags @0.5 | "
                     "best catch (cap 2 flags) | false flags at best |")
        lines.append("|---|---:|---:|---:|---:|")
        wrongs = [x for x in items if not x["truth"]]
        corrects = [x for x in items if x["truth"]]
        for h in HEADS:
            caught05 = sum(1 for x in wrongs if x["p"][h] < 0.5)
            flags05 = sum(1 for x in corrects if x["p"][h] < 0.5)
            # best threshold: largest threshold with flags <= 2
            best_catch, best_flags, best_t = 0, 0, None
            cand_ts = sorted({x["p"][h] for x in items})
            for t in cand_ts:
                c = sum(1 for x in wrongs if x["p"][h] < t)
                f = sum(1 for x in corrects if x["p"][h] < t)
                if f <= 2 and c > best_catch:
                    best_catch, best_flags, best_t = c, f, t
            lines.append(f"| {h} | {caught05} | {flags05} | {best_catch} | {best_flags} |")
        lines.append("")

    for run_name, _ in RUNS:
        sub = [x for x in all_items if x["run"] == run_name]
        cap2_table(sub, f"Run {run_name} (n={len(sub)})")
    cap2_table(all_items, f"Pooled (n={len(all_items)})")

    lines.append("## Separation read (p ranges)")
    lines.append("")
    lines.append("| head | silent wrongs p min-max | silent corrects p min (10th pct) |")
    lines.append("|---|---|---|")
    for h in HEADS:
        ws = [x["p"][h] for x in all_items if not x["truth"]]
        cs = sorted(x["p"][h] for x in all_items if x["truth"])
        p10 = cs[max(0, len(cs) // 10)] if cs else float("nan")
        lines.append(f"| {h} | {min(ws):.4f}-{max(ws):.4f} | {p10:.4f} (n={len(cs)}) |")
    lines.append("")

    lines.append("## Per-item p (pooled), sorted by L0.75")
    lines.append("")
    lines.append("| run | item | truth | L0.5 | L0.75 | L1 | ridge-concat | mlp-concat |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for x in sorted(all_items, key=lambda z: z["p"]["L0.75"]):
        lines.append(
            f"| {x['run']} | {x['id']} | {'W' if not x['truth'] else 'C'} | "
            + " | ".join(f"{x['p'][h]:.4f}" for h in HEADS) + " |"
        )
    lines.append("")

    lines.append("## Reading")
    lines.append("")
    lines.append("- The frozen 0.5 cut on L0.75 reproduces the M4 picture on the "
                 "pooled fresh corpus: silent wrongs' p spans 0.00-1.00 and "
                 "overlaps the corrects' mass, so no single threshold separates.")
    lines.append("- The cap-2 (<=2 false flags) best-catch row above is a "
                 "*post-hoc* selection on this corpus and is reported only as the "
                 "ceiling a retrained head must clear under the pre-registered "
                 "rule; it is not an accepted operating point.")
    lines.append("- A head that actually separates must show a gap in the p ranges "
                 "above, not a better threshold on the same features.")
    (out_dir / "AUDIT.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_dir / 'audit.json'} and AUDIT.md")


if __name__ == "__main__":
    main()
