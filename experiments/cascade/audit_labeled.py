"""Audit a labeled silent corpus: does any frozen head separate wrongs from
corrects on the *silent* distribution?

Input: <corpus_dir>/corpus.jsonl + corpus_states.npz produced by
sample_check.py.  Output: <corpus_dir>/AUDIT_LABELED.md + audit_labeled.json.

This is the gate's real acceptance test: a head that actually separates
must show a gap in the p ranges (wrongs' max < corrects' 10th pct), not a
better threshold on the same features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from harness.cascade.learned_gate import ReadoutHead

HEADS = ("L0.5", "L0.75", "L1", "ridge-concat", "mlp-concat")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="dir with corpus.jsonl + corpus_states.npz")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cdir = Path(args.corpus)
    meta = [json.loads(l) for l in (cdir / "corpus.jsonl").read_text().splitlines() if l.strip()]
    st = np.load(cdir / "corpus_states.npz")
    L05 = st["L0_5"]; L075 = st["L0_75"]; L1 = st["L1"]
    assert len(meta) == len(L05) == len(L075) == len(L1), (len(meta), len(L05))

    # Build per-row feats dict for each head.
    feats_list = []
    for i in range(len(meta)):
        feats_list.append({"L0.5": L05[i], "L0.75": L075[i], "L1": L1[i]})

    heads = {h: ReadoutHead(h) for h in HEADS}
    for m, f in zip(meta, feats_list):
        m["p"] = {h: heads[h].score(f) for h in HEADS}

    out_dir = Path(args.out) if args.out else cdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_labeled.json").write_text(json.dumps({
        "n": len(meta),
        "n_correct": sum(1 for m in meta if m["label"]),
        "n_wrong": sum(1 for m in meta if not m["label"]),
        "items": meta,
    }, indent=1))

    wrongs = [m for m in meta if not m["label"]]
    corrects = [m for m in meta if m["label"]]

    lines = []
    lines.append("# Labeled silent-corpus audit")
    lines.append("")
    lines.append(f"Corpus: {len(meta)} drafts ({len(corrects)} correct, {len(wrongs)} wrong) "
                 f"from sample_check.py (mined-hard tasks, k fresh drafts each, checker-labeled).")
    lines.append("")
    lines.append("## Per-head separation")
    lines.append("")
    lines.append("| head | wrongs p min-max | corrects p 10th pct (n) | caught@0.5 | flags@0.5 | cap-2 best catch/flags |")
    lines.append("|---|---|---|---:|---:|---:|")
    for h in HEADS:
        ws = [m["p"][h] for m in wrongs]
        cs = sorted(m["p"][h] for m in corrects)
        p10 = cs[max(0, len(cs) // 10)] if cs else float("nan")
        caught05 = sum(1 for m in wrongs if m["p"][h] < 0.5)
        flags05 = sum(1 for m in corrects if m["p"][h] < 0.5)
        # cap-2 best
        best_c, best_f = 0, 0
        for t in sorted({m["p"][h] for m in meta}):
            c = sum(1 for m in wrongs if m["p"][h] < t)
            f = sum(1 for m in corrects if m["p"][h] < t)
            if f <= 2 and c > best_c:
                best_c, best_f = c, f
        gap = "GAP" if (ws and cs and max(ws) < p10) else "overlap"
        lines.append(f"| {h} | {min(ws):.4f}-{max(ws):.4f} | {p10:.4f} (n={len(cs)}) | "
                     f"{caught05} | {flags05} | {best_c}/{best_f} |")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    sep_heads = [h for h in HEADS
                 if wrongs and corrects and max(m["p"][h] for m in wrongs) <
                 sorted(m["p"][h] for m in corrects)[max(0, len(corrects) // 10)]]
    if sep_heads:
        lines.append(f"**Separation found on: {', '.join(sep_heads)}.** "
                     "A retrain / restart decision can now be pre-registered on this evidence.")
    else:
        lines.append("**No head separates.** Wrongs' p ranges overlap the corrects' mass on the "
                     "silent distribution — the gate's problem is features/labels, not threshold. "
                     "Do not restart M5 or retrain yet; grow the corpus.")
    lines.append("")
    lines.append("## Per-item p (sorted by L0.75)")
    lines.append("")
    lines.append("| task | truth | L0.5 | L0.75 | L1 | ridge-concat | mlp-concat |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for m in sorted(meta, key=lambda x: x["p"]["L0.75"]):
        lines.append(f"| {m['task_id']} | {'W' if not m['label'] else 'C'} | " +
                     " | ".join(f"{m['p'][h]:.4f}" for h in HEADS) + " |")
    (out_dir / "AUDIT_LABELED.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}/audit_labeled.json + AUDIT_LABELED.md")


if __name__ == "__main__":
    main()
PY
echo "written"