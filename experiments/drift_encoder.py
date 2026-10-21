"""The drift decision gets its own encoder.

**Why.** The membrane makes two decisions with one encoder, and the
campaign measured three ways that one encoder cannot serve both: the
dedup hinge moves the whole distribution and the drift AUC falls
0.957 -> 0.731; training the drift task beside the hinge makes it
0.420; our Qwen student, trained for dedup, inverts it (0.457).  The
Global tab already holds a different encoder per conversation, so the
drift decision can have its own model - the question is whether a model
trained *only* on the drift task is better at it than the stock one.

**The candidate.** The shipped `paraphrase-MiniLM-L3-v2` (so the cost,
2.7 ms/chunk, is unchanged) trained only on the drift task's triplets:
a recent window, the closest-in-topic older window as the positive, the
furthest as the negative.  The builder is the one the harness's own
training uses (`encoder_finetune.build_examples`), and it is trained on
the **train half** of the conversations only.

**The gate.** The hold-out half of the conversations, the shipped
threshold's own metric (`1 - cosine`, drift = above the threshold), and
the decision priced at a threshold calibrated on the **train half**
only.  The oracle `best` F1 is printed for reference and never used for
the gate: a threshold fitted on the hold-out is not a threshold the
harness could have.

    python -m experiments.drift_encoder --epochs 20
    python -m experiments.drift_encoder --measure-only <checkpoint>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "encoder-suite" / "results" / "drift-encoder"
SEED = 20261020


def drift_scores(cases: list[dict], embed) -> list[float]:
    """The harness's own signal: 1 - cosine(recent, older)."""
    from experiments.encoder_suite import _cosine, embed_many

    texts = [c["recent"] for c in cases] + [c["older"] for c in cases]
    vectors = embed_many(embed, texts)
    n = len(cases)
    return [1.0 - _cosine(vectors[i], vectors[n + i]) for i in range(n)]


def at_threshold(signals, labels, threshold: float) -> dict:
    tp = sum(1 for s, l in zip(signals, labels) if l and s > threshold)
    fp = sum(1 for s, l in zip(signals, labels) if not l and s > threshold)
    fn = sum(1 for s, l in zip(signals, labels) if l and s <= threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"the_threshold": round(threshold, 3),
            "the_precision": round(precision, 4),
            "the_recall": round(recall, 4),
            "the_f1": round(f1, 4),
            "the_fires": round((tp + fp) / len(labels), 3)}


def best_on(signals, labels) -> dict:
    """The oracle threshold - reported, never gated on."""
    best = {"the_f1": -1.0}
    for threshold in np.arange(0.30, 0.999, 0.002):
        row = at_threshold(signals, labels, float(threshold))
        if row["the_f1"] > best["the_f1"]:
            best = row
    return best


def train(args) -> Path:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    from experiments.encoder_finetune import build_examples, in_batch_loss
    from experiments.encoder_suite import load_conversations
    from experiments.qwen_encoder import split_conversations

    conversations = load_conversations(args.corpus)
    train_convs, holdout = split_conversations(conversations, seed=SEED)
    examples = [e for e in build_examples(train_convs, seed=SEED)
                if e["task"] == "drift"]
    print(f"the training conversations: {len(train_convs)}; "
          f"the held-out: {len(holdout)}; the drift triplets: {len(examples)}")
    if not examples:
        raise SystemExit("no drift triplets in the train split")

    student = SentenceTransformer(args.student, device="cpu")
    module = student[0].auto_model
    module.train()
    params = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    tokenizer = student.tokenizer

    def embed(texts: list[str]):
        batch = tokenizer(texts, padding=True, truncation=True,
                          max_length=256, return_tensors="pt")
        out = module(**batch)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        return summed / mask.sum(dim=1).clamp(min=1e-9)

    rng = random.Random(SEED)
    steps = (len(examples) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(examples)
        total = 0.0
        for start in range(0, len(examples) - 1, args.batch_size):
            batch = examples[start:start + args.batch_size]
            if len(batch) < 2:
                continue
            anchors = [e["anchor"] for e in batch]
            positives = [e["positive"] for e in batch]
            negatives = [e["negative"] for e in batch]
            a_vec = embed(anchors)
            p_vec = embed(positives)
            n_vec = embed(negatives)
            loss = in_batch_loss(a_vec, p_vec, temperature=args.temperature)
            a_norm = F.normalize(a_vec, dim=-1)
            n_norm = F.normalize(n_vec, dim=-1)
            neg_sim = (a_norm * n_norm).sum(dim=-1)
            loss = loss + args.hard_weight * F.relu(neg_sim - 0.8).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total += float(loss.item())
        print(f"  epoch {epoch}/{args.epochs}  the loss {total / max(steps, 1):.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    student.save(str(out))
    print(f"  the checkpoint: {out}")
    return out


def measure(args) -> dict:
    from experiments.encoder_suite import load_conversations, make_encoder
    from experiments.qwen_encoder import split_conversations

    conversations = load_conversations(args.corpus)
    train_convs, holdout = split_conversations(conversations, seed=SEED)
    from experiments.encoder_suite import build_drift_cases

    calibrate = build_drift_cases(train_convs)
    cases = build_drift_cases(holdout)
    labels = [bool(c["drift"]) for c in cases]
    print(f"  the calibration cases: {len(calibrate)} "
          f"({sum(1 for c in calibrate if c['drift'])} drift); "
          f"the hold-out cases: {len(cases)} "
          f"({sum(1 for c in labels if c)} drift)")

    rows = {}
    for name, encoder in (("the stock L3-v2", args.baseline),
                          ("the drift encoder", args.measure_only or args.out)):
        embed, info = make_encoder(
            encoder if encoder == args.baseline else "checkpoint:" + encoder)
        cal_signals = drift_scores(calibrate, embed)
        cal_labels = [bool(c["drift"]) for c in calibrate]
        cal_best = best_on(cal_signals, cal_labels)
        signals = drift_scores(cases, embed)
        from experiments.encoder_suite import _auc, measure_cost

        cost = measure_cost(embed, [c["text"] for c in conversations[0]["chunks"]])
        rows[name] = {
            "the_auc": round(_auc(signals, labels), 4),
            "the_auc_train_calibrated": at_threshold(
                signals, labels, cal_best["the_threshold"]),
            "the_oracle_best": best_on(signals, labels),
            "the_at_0_6": at_threshold(signals, labels, 0.6),
            "the_signal_p50": round(float(np.median(signals)), 4),
            "the_cost_ms": cost.get("ms_per_chunk"),
            "the_threshold_from": "the train split's best F1",
            "the_calibration_best": cal_best,
            "the_info": {k: v for k, v in info.items()
                         if k in ("the_params", "the_dim", "the_load_s")},
        }
        print(f"  {name:20s} AUC {rows[name]['the_auc']:.4f} | "
              f"F1 at the calibrated threshold "
              f"{rows[name]['the_auc_train_calibrated']['the_f1']:.4f} "
              f"({rows[name]['the_auc_train_calibrated']['the_threshold']}) | "
              f"oracle {rows[name]['the_oracle_best']['the_f1']:.4f} | "
              f"{rows[name]['the_cost_ms']} ms/chunk")

    verdict = {}
    stock = rows["the stock L3-v2"]
    cand = rows["the drift encoder"]
    verdict["the_auc_not_worse"] = cand["the_auc"] >= stock["the_auc"]
    verdict["the_f1_at_calibrated_not_worse"] = (
        cand["the_auc_train_calibrated"]["the_f1"]
        >= stock["the_auc_train_calibrated"]["the_f1"])
    verdict["the_cost_not_worse"] = (
        (cand["the_cost_ms"] or 99) <= (stock["the_cost_ms"] or 0) * 1.05)
    verdict["passes"] = all(verdict.values())
    out = {"the_seed": SEED, "the_rows": rows, "the_verdict": verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "drift-encoder.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(verdict, indent=1))
    print(f"  wrote {OUT / 'drift-encoder.json'}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="sentence-transformers/paraphrase-MiniLM-L3-v2")
    parser.add_argument("--baseline", default="l3")
    parser.add_argument("--corpus", nargs="+",
                        default=[str(REPO / "harness_state"),
                                 str(REPO.parent / "splinter-memory" / "harness_state")])
    parser.add_argument("--out", default=str(OUT / "l3-drift"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--measure-only", default="")
    args = parser.parse_args()
    if not args.measure_only:
        train(args)
    measure(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
