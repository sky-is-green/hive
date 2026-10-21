"""The encoder's held-out test: does the tuning generalise?

The tuned checkpoint was trained on the same traffic it was measured on, so
it is a *fit*, not a generalisation. This re-runs the whole thing with a
split: the conversations are divided in half (deterministically, by id),
the model is trained on one half, and both the stock encoder and the
tuned one are measured on the other half - the half neither the training
nor the tuning ever saw.

The measurement is the same one the suite makes: the two membrane
decisions (the dedup and the drift), with the lexical labels, plus the
cost. What changes is the corpus.

Usage::

    python -m experiments.encoder_holdout --out experiments/encoder-suite/results/holdout
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.encoder_suite import (build_dedup_pairs, build_drift_cases,
                                       load_conversations, make_encoder,
                                       measure_cost, measure_dedup,
                                       measure_drift)


def split_conversations(conversations: list[dict], seed: int = 20261020
                        ) -> tuple[list[dict], list[dict]]:
    """Half for training, half for the measurement - by conversation."""
    ordered = sorted(conversations, key=lambda c: c["id"])
    rng = random.Random(seed)
    rng.shuffle(ordered)
    cut = len(ordered) // 2
    return ordered[:cut], ordered[cut:]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", nargs="+",
                        default=[str(REPO_ROOT / "harness_state"),
                                 str(REPO_ROOT.parent / "splinter-memory" / "harness_state")])
    parser.add_argument("--student", default="sentence-transformers/paraphrase-MiniLM-L3-v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--tasks", default="dedup")
    parser.add_argument("--hinge-margin", type=float, default=0.0,
                        help="the dedup task's objective: a hinge at this "
                             "cosine margin (the shipped threshold is 0.92). "
                             "0 = the contrastive loss, which optimises a "
                             "ranking rather than the decision")
    parser.add_argument("--per-conv", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    conversations = load_conversations(args.corpus)
    if not conversations:
        raise SystemExit("no conversations found; pass --corpus")
    train_convs, holdout = split_conversations(conversations, seed=args.seed)
    print(f"corpus: {len(conversations)} conversations -> "
          f"{len(train_convs)} train / {len(holdout)} holdout")

    # --- the training, on the train half only -------------------------
    from experiments.encoder_finetune import build_examples, train

    examples = build_examples(train_convs, seed=args.seed)
    print(f"training examples (from the train half): {len(examples)}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = (args.out / "holdout-tuned") if args.out else REPO_ROOT / "models" / "encoder" / "holdout-tuned"

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        # build_examples already filtered to the train half; train on it
        sys.argv = ["encoder_finetune"]
        import experiments.encoder_finetune as ft

        original = ft.build_examples
        ft.build_examples = lambda convs, seed=0: examples
        try:
            ft.train(argparse.Namespace(
                student=args.student, teacher="", corpus=args.corpus,
                out=str(checkpoint), epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, temperature=0.05,
                alpha=0.5, hard_weight=0.5, seed=args.seed, log_every=40,
                hinge_margin=args.hinge_margin,
                tasks=args.tasks))
        finally:
            ft.build_examples = original

    # --- the measurement, on the holdout half only -------------------
    pairs = build_dedup_pairs(holdout, per_conv=args.per_conv, seed=args.seed)
    cases = build_drift_cases(holdout)
    print(f"\nholdout: {len(pairs)} duplicate pairs, {len(cases)} drift cases")

    results: dict = {"train_conversations": len(train_convs),
                     "holdout_conversations": len(holdout),
                     "checkpoint": str(checkpoint), "encoders": {}}
    for label, name in (("stock", args.student),
                        ("tuned", "checkpoint:" + str(checkpoint))):
        embed, info = make_encoder(name)
        dedup = measure_dedup(pairs, embed)
        drift = measure_drift(cases, embed)
        cost = measure_cost(embed, [c["text"] for c in holdout[0]["chunks"]])
        results["encoders"][label] = {"info": info, "dedup": dedup,
                                      "drift": drift, "cost": cost}
        print(f"{label}: dedup@0.92 f1 {dedup['at_0.92']['f1']} "
              f"(precision {dedup['at_0.92']['precision']}, "
              f"recall {dedup['at_0.92']['recall']}) | "
              f"dedup best {dedup['best']['f1']} | "
              f"drift auc {drift['auc']} | {cost.get('ms_per_chunk')} ms/chunk")

    stock = results["encoders"]["stock"]
    tuned = results["encoders"]["tuned"]
    verdict = {
        "dedup_recall_up": (tuned["dedup"]["at_0.92"]["recall"]
                            >= stock["dedup"]["at_0.92"]["recall"]),
        "dedup_precision_kept": (tuned["dedup"]["at_0.92"]["precision"]
                                 >= stock["dedup"]["at_0.92"]["precision"] - 0.02),
        "drift_auc_kept": tuned["drift"]["auc"] >= stock["drift"]["auc"] - 0.02,
        "cost_ok": (tuned["cost"].get("ms_per_chunk", 99)
                    <= 2.0 * stock["cost"].get("ms_per_chunk", 1)),
    }
    verdict["generalises"] = all(verdict.values())
    results["verdict"] = verdict
    print(f"\nverdict: {verdict}")

    if args.out:
        out = args.out / "holdout.json"
        out.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
