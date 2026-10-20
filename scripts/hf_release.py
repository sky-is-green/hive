#!/usr/bin/env python
"""Assemble a Hugging Face release folder from the repository.

The release folders are build artifacts: the card is the paper, the
figures come from the memory system's repository, and the license and
citation come from here.  This script assembles them so a release is
reproducible instead of hand-copied.

Usage::

    python scripts/hf_release.py splinter-memory --out /tmp/release-splinter
    python scripts/hf_release.py hive-cascade --out /tmp/release-cascade
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIBLING = REPO.parent / "splinter-memory"

RELEASES = {
    "splinter-memory": {
        "card": REPO / "docs" / "SPLINTER-MEMORY.md",
        "figures": SIBLING / "figures",
        "figure_files": ["flow.png", "arch.png", "p4.png", "p9.png",
                         "pes.png", "budget.png", "b.png"],
        "citation": SIBLING / "CITATION.cff",
        "readme": "README.md",
        "files": [],
    },
    "hive-cascade": {
        "card": REPO / "docs" / "CASCADE-PAPER.md",
        "figures": REPO / "docs" / "figures",
        "figure_files": [],
        "citation": REPO / "CITATION.cff",
        "readme": "README.md",
        # the benchmark (with its train/test split) and the run reports
        "files": [
            (REPO / "experiments" / "cascade" / "tasks-bench.json", "tasks-bench.json"),
            (REPO / "experiments" / "cascade" / "tasks-community.json", "tasks-community.json"),
            (REPO / "experiments" / "cascade" / "results"
             / "runner-cascade-sampled-nothink-20261018" / "run" / "report.json",
             "results/runner-cascade-sampled-nothink-20261018-report.json"),
            (REPO / "experiments" / "cascade" / "results"
             / "runner-cascade-sampled-20261018" / "run" / "report.json",
             "results/runner-cascade-sampled-20261018-report.json"),
        ],
    },
    "splinter-encoder": {
        # the tuned drone encoder: the weights plus the operating points
        # they ship with (the thresholds are part of the release - they
        # are per-encoder, and the shipped ones were calibrated for a
        # different model)
        "card": REPO / "docs" / "ENCODER-CARD.md",
        "figures": REPO / "docs" / "figures",
        "figure_files": [],
        "citation": REPO / "CITATION.cff",
        "readme": "README.md",
        "files": [
            (REPO / "experiments" / "encoder-suite" / "results" / "20261020"
             / "l3-tuned" / "config_sentence_transformers.json",
             "config_sentence_transformers.json"),
            (REPO / "experiments" / "encoder-suite" / "results" / "20261020"
             / "l3-tuned" / "modules.json", "modules.json"),
            (REPO / "experiments" / "encoder-suite" / "results" / "20261020"
             / "l3-tuned" / "sentence_bert_config.json",
             "sentence_bert_config.json"),
            (REPO / "experiments" / "encoder-suite" / "results" / "20261020"
             / "l3-tuned" / "config.json", "config.json"),
            (REPO / "experiments" / "encoder-suite" / "results" / "20261020"
             / "l3-tuned" / "finetune_meta.json", "finetune_meta.json"),
        ],
    },
}


def build(name: str, out: Path) -> int:
    spec = RELEASES.get(name)
    if spec is None:
        print(f"unknown release: {name} (known: {', '.join(RELEASES)})",
              file=sys.stderr)
        return 2
    card = spec["card"]
    if not card.exists():
        print(f"the card is missing: {card}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(card, out / spec["readme"])

    figures = spec["figures"]
    copied = 0
    if figures.is_dir():
        target = out / "figures"
        target.mkdir(exist_ok=True)
        for filename in spec["figure_files"]:
            src = figures / filename
            if src.exists():
                shutil.copy2(src, target / filename)
                copied += 1
    for src, rel in spec.get("files", []):
        if src.exists():
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            print(f"  (missing: {src})", file=sys.stderr)
    license_src = REPO / "LICENSE"
    if license_src.exists():
        shutil.copy2(license_src, out / "LICENSE")
    citation = spec.get("citation")
    if citation and citation.exists():
        shutil.copy2(citation, out / "CITATION.cff")

    print(f"built {out}")
    for path in sorted(out.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(out)}  ({path.stat().st_size} bytes)")
    if copied < len(spec["figure_files"]):
        print(f"  (note: {len(spec['figure_files']) - copied} figure(s) not found in "
              f"{figures})", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("release", choices=sorted(RELEASES))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    return build(args.release, args.out.expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
