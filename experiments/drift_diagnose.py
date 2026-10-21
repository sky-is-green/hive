"""Why does the drift decision invert on our Qwen student?

The drift case is a (recent window, older half) pair from one
conversation, labelled *drift* by lexical overlap (Jaccard of the content
words below 0.05).  The signal the harness computes is
``1 - cosine(recent, older)``.  The stock MiniLM reaches AUC 0.957 on the
hold-out; our 4-layer student reaches 0.457 - below chance, and the 8-
and 28-layer students are worse still.  Capacity is not the story, so
the diagnostic separates the three candidates:

1. **truncation** - the drift texts average ~7,100 characters and the
   student reads 192 tokens (the stock reads 256);
2. **pooling** - the student takes the last real token, the stock mean-
   pools, and a last token at a truncated mid-sentence is a bad summary;
3. **distribution** - the student's cosine scale collapsed during the
   dedup hinge, so the signal is a small difference between large
   cosines.

It reports, for the stock and for the student under each (pooling,
max-length) setting: the AUC overall, the AUC *within* conversations
(the older text is constant inside a conversation, so this is the pure
test), the per-class cosine means, the spread, and the correlation of
the signal with the lexical label and with the two text lengths.

Runs on the GPU when one is given (the CPU pass at 512 tokens is ~half
an hour); the arithmetic is the same either way.

    HIP_VISIBLE_DEVICES=0 ../.venv-rocm/bin/python -m experiments.drift_diagnose \
        --device cuda --n-cases 600
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np

from experiments.encoder_suite import (_auc, build_drift_cases,
                                       load_conversations)
from experiments.qwen_encoder import split_conversations

SEED = 20261020
STUDENT = "experiments/encoder-suite/results/qwen/L4/qwen-bi-encoder"
STOCK = "sentence-transformers/paraphrase-MiniLM-L3-v2"


def embed_with(path: str, pooling: str, max_length: int,
               device: str = "cpu", batch: int = 16):
    """An embedder with the pooling and the length as dials."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, dtype=torch.float32).to(device)
    model.eval()

    def embed(texts: list[str]) -> np.ndarray:
        vecs = []
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"]
            if pooling == "last":
                idx = mask.sum(dim=1) - 1
                picked = hidden[torch.arange(hidden.size(0), device=device), idx]
            else:
                picked = ((hidden * mask.unsqueeze(-1)).sum(dim=1)
                          / mask.sum(dim=1, keepdim=True))
            vecs.append(F.normalize(picked.float(), dim=-1).cpu().numpy())
        return np.vstack(vecs)

    return embed


def corr(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def within_conversation_auc(cases, signals, labels) -> tuple[float, int]:
    """The AUC with the conversations kept apart - the older text is
    constant inside a conversation, so this is the pure ordering test."""
    by_conv: dict[str, list[int]] = {}
    for i, case in enumerate(cases):
        by_conv.setdefault(case["conversation_id"], []).append(i)
    aucs = []
    for idx in by_conv.values():
        y = [labels[i] for i in idx]
        if len(set(y)) < 2:
            continue
        aucs.append(_auc([signals[i] for i in idx], y))
    return (round(statistics.mean(aucs), 4) if aucs else float("nan"),
            len(aucs))


def describe(name: str, cases, embed) -> dict:
    texts = [c["recent"] for c in cases] + [c["older"] for c in cases]
    n = len(cases)
    vectors = embed(texts)
    cos = np.array([float(vectors[i] @ vectors[n + i]) for i in range(n)])
    signals = [1.0 - c for c in cos]
    labels = [bool(c["drift"]) for c in cases]
    jac = [c["jaccard"] for c in cases]
    len_recent = [len(c["recent"]) for c in cases]
    len_older = [len(c["older"]) for c in cases]

    drift_cos = [c for c, l in zip(cos, labels) if l]
    same_cos = [c for c, l in zip(cos, labels) if not l]
    within, n_conv = within_conversation_auc(cases, signals, labels)
    return {
        "the_arm": name,
        "the_auc": round(_auc(signals, labels), 4),
        "the_auc_within_conversation": within,
        "the_conversations": n_conv,
        "the_drift_cases": len(drift_cos),
        "the_cos_drift_mean": round(float(np.mean(drift_cos)), 4),
        "the_cos_same_mean": round(float(np.mean(same_cos)), 4),
        "the_cos_std": round(float(np.std(cos)), 4),
        "the_signal_p50": round(float(np.median(signals)), 4),
        "the_corr_signal_jaccard": round(corr(signals, jac), 3),
        "the_corr_signal_len_recent": round(corr(signals, len_recent), 3),
        "the_corr_signal_len_older": round(corr(signals, len_older), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=600)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/encoder-suite/results/drift-diagnose.json")
    args = ap.parse_args()

    convs = load_conversations(["harness_state",
                                "../splinter-memory/harness_state"])
    train, holdout = split_conversations(convs, seed=SEED)
    all_cases = build_drift_cases(holdout)
    random.Random(SEED).shuffle(all_cases)
    cases = all_cases[:args.n_cases]
    print(f"the holdout: {len(holdout)} conversations -> {len(cases)} of "
          f"{len(all_cases)} drift cases "
          f"({sum(1 for c in cases if c['drift'])} drift)", flush=True)

    keys = ["the_arm", "the_auc", "the_auc_within_conversation",
            "the_drift_cases", "the_cos_drift_mean", "the_cos_same_mean",
            "the_cos_std", "the_signal_p50", "the_corr_signal_jaccard",
            "the_corr_signal_len_recent", "the_corr_signal_len_older"]

    def show(row):
        print("  ".join(f"{k.replace('the_', '')}={row[k]}" for k in keys), flush=True)

    rows = []
    for path, pooling, length, name in (
            (STOCK, "mean", 256, "stock L3 (mean,256)"),
            (STUDENT, "last", 192, "student (last,192)"),
            (STUDENT, "mean", 192, "student (mean,192)"),
            (STUDENT, "last", 512, "student (last,512)"),
            (STUDENT, "mean", 512, "student (mean,512)")):
        row = describe(name, cases, embed_with(path, pooling, length, args.device))
        rows.append(row)
        show(row)

    Path(args.out).write_text(json.dumps(
        {"the_n": len(cases), "the_device": args.device, "the_rows": rows}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
