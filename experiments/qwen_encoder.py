#!/usr/bin/env python
"""Our own Qwen-family bi-encoder: a Qwen3 decoder trained as an encoder.

The Qwen-family encoder closest to "the generator's own space" is
``Qwen/Qwen3-Embedding-0.6B`` - a Qwen3 decoder trained as a bi-encoder.
It is also 22x the drone's cost (19.6 ms a chunk against the stock
MiniLM's 0.9 ms), so it is a quality ceiling and not a shippable drone.

This is Option B of ``docs/ENCODER-PLAN.md``: **our own** Qwen-family
bi-encoder, trained on the harness's own traffic, with the Qwen teacher
supplying the judgements.

Three things make it ours and not a copy of the teacher:

1. **The data is ours** - the curation's own pairs, from the harness's
   conversations.
2. **The objective is the decision, not the ranking.**  The measurements
   are unambiguous: the threshold that maximises F1 is not the threshold
   that preserves the facts (a false merge *deletes* a fact; a missed
   merge only costs budget).  So the dedup objective is a **hinge at the
   shipped operating point** (0.92): the duplicates' cosine above it, the
   distinct pairs' below it.
3. **The teacher's relation structure is the soft target**, not the
   teacher's own labels: the student reproduces the teacher's *judgements*
   over the pairs, so it inherits the generator-family semantics rather
   than the teacher's quirks.

The gate is the held-out split: train on half the conversations, measure
on the other half, and require that no job regresses and that the cost
holds.

**The three stages, because the two venvs have different packages.** The
CPU venv has sentence-transformers (the teacher and every candidate in
the suite) and a CPU torch; the ROCm venv has the GPU torch but no
sentence-transformers.  So:

    python -m experiments.qwen_encoder --stage teacher   # the CPU venv
    ../.venv-rocm/bin/python -m experiments.qwen_encoder --stage train
    python -m experiments.qwen_encoder --stage measure   # the CPU venv
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TEACHER = "Qwen/Qwen3-Embedding-0.6B"
STUDENT = "Qwen/Qwen3-0.6B"
#: the shipped dedup threshold: the operating point the objective is about
OPERATING_POINT = 0.92
#: The Qwen3-Embedding recipe puts an instruction on the query side only.
#: The harness has no notion of a side - one embedding function serves both
#: - so the student is trained and measured *symmetrically*, which is how
#: the suite measures every asymmetric model (the ``qwen3-0.6b`` arm, the
#: one without the prefix).  The teacher's judgements below are computed
#: the same way the teacher is measured, so the two are on one footing.


def split_conversations(conversations: list[dict], seed: int = 20261020
                        ) -> tuple[list[dict], list[dict]]:
    """Half for training, half for the measurement - by conversation."""
    ordered = sorted(conversations, key=lambda c: c["id"])
    random.Random(seed).shuffle(ordered)
    cut = len(ordered) // 2
    return ordered[:cut], ordered[cut:]


def _same_text(a: str, b: str) -> bool:
    """Whether two chunks are the same text (whitespace aside)."""
    return " ".join(a.split()).strip().lower() == " ".join(b.split()).strip().lower()


def drop_degenerate(pairs: list[dict]) -> tuple[list[dict], int]:
    """Drop the pairs whose "distinct" negative is textually the same as the query.

    The fixtures repeat some prompts across conversations, so the random
    negative - a chunk from *another* conversation - is sometimes the very
    same text as the query.  Those pairs are labelled "distinct" and are
    not: the teacher scores them at 1.0, and a hinge at the operating
    point would ask the student to push a pair of identical strings
    below 0.92.  That is an impossible target and it would corrupt the
    training, so the pair goes.
    """
    kept, dropped = [], 0
    for p in pairs:
        if _same_text(p["negative"], p["a"]) or _same_text(p["negative"], p["b"]):
            dropped += 1
            continue
        kept.append(p)
    return kept, dropped


def build_pairs(conversations: list[dict], per_conv: int, seed: int) -> list[dict]:
    """The pairs the drone has to get right, from our own conversations."""
    from experiments.encoder_suite import build_dedup_pairs

    rng = random.Random(seed)
    pairs = []
    for pair in build_dedup_pairs(conversations, per_conv=per_conv, seed=seed):
        other = rng.choice(conversations)
        if not other.get("chunks"):
            continue
        negative = rng.choice(other["chunks"])["text"]
        pairs.append({"a": pair["a"], "b": pair["b"], "negative": negative,
                      "duplicate": bool(pair["duplicate"])})
    pairs, dropped = drop_degenerate(pairs)
    if dropped:
        print(f"  dropped {dropped} degenerate pairs (the negative was the query)",
              flush=True)
    return pairs


def student_encoder(path: str):
    """Our student's embedding function, the way the suite measures one.

    A Qwen3 decoder with the Qwen3-Embedding pooling (the last real
    token), unit length, on the CPU - the same contract every other
    candidate in the suite has, so the comparison is exact.
    """
    import time

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, dtype=torch.float32)
    model.eval()

    def embed(texts: list[str]):
        """The student's vectors.

        An empty chunk is a real thing in a store (a turn with no content),
        and the Qwen tokenizer raises on one - so the empty texts are
        replaced by a single space and their vectors are zeroed.  The
        suite's other encoders tolerate it, so ours has to as well.
        """
        import numpy as np

        empty = [i for i, t in enumerate(texts) if not (t or "").strip()]
        safe = [(t if (t or "").strip() else " ") for t in texts]
        batch = tokenizer(safe, padding=True, truncation=True, max_length=192,
                          return_tensors="pt")
        with torch.no_grad():
            out = model(**batch).last_hidden_state
            lengths = batch["attention_mask"].sum(dim=1) - 1
            picked = out[torch.arange(out.size(0)), lengths]
            vec = F.normalize(picked.float(), dim=-1).numpy()
        for i in empty:
            vec[i] = 0.0
        return vec

    info = {"the_model": path, "the_pooling": "last token", "the_normalised": True,
            "the_device": "cpu", "the_load_s": round(time.time() - t0, 1),
            "the_params": sum(p.numel() for p in model.parameters())}
    return embed, info


# ---------------------------------------------------------------------------
# stage 1: the teacher's judgements (the CPU venv)
# ---------------------------------------------------------------------------
def stage_teacher(args) -> int:
    from experiments.encoder_suite import load_conversations, make_encoder

    conversations = load_conversations(args.corpus)
    if not conversations:
        raise SystemExit("no conversations found; pass --corpus")
    train_convs, holdout = split_conversations(conversations, seed=args.seed)
    print(f"corpus: {len(conversations)} conversations -> {len(train_convs)} train "
          f"/ {len(holdout)} holdout", flush=True)
    pairs = build_pairs(train_convs, args.per_conv, args.seed)
    print(f"the training pairs: {len(pairs)} "
          f"({sum(1 for p in pairs if p['duplicate'])} duplicates)", flush=True)

    embed, info = make_encoder(TEACHER)
    t0 = time.time()
    judgements = []
    for i, p in enumerate(pairs):
        q = embed([p["a"]])[0]
        pos = embed([p["b"]])[0]
        neg = embed([p["negative"]])[0]

        def cos(a, b):
            return float(a @ b / (math.sqrt(float(a @ a) * float(b @ b)) + 1e-9))

        judgements.append({"cos_pos": cos(q, pos), "cos_neg": cos(q, neg)})
        if (i + 1) % 100 == 0:
            print(f"  the teacher's judgements: {i + 1}/{len(pairs)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    out = {"the_teacher": TEACHER, "the_teacher_info": info,
           "the_seed": args.seed, "the_per_conv": args.per_conv,
           "the_train_conversations": len(train_convs),
           "the_holdout_conversations": len(holdout),
           "the_pairs": pairs, "the_judgements": judgements}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "teacher-judgements.json").write_text(json.dumps(out))
    print(f"wrote {args.out / 'teacher-judgements.json'}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# stage 2: the training (the GPU venv)
# ---------------------------------------------------------------------------
def stage_train(args) -> int:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    data = json.loads((args.out / "teacher-judgements.json").read_text())
    pairs = data["the_pairs"]
    judgements = data["the_judgements"]
    # the same filter the pair builder applies (the cached pairs may predate
    # it): the degenerate ones are dropped from both lists together
    keep = [i for i, p in enumerate(pairs)
            if not (_same_text(p["negative"], p["a"]) or _same_text(p["negative"], p["b"]))]
    if len(keep) != len(pairs):
        print(f"  dropped {len(pairs) - len(keep)} degenerate pairs", flush=True)
    pairs = [pairs[i] for i in keep]
    judgements = [judgements[i] for i in keep]
    print(f"the training pairs: {len(pairs)} (the teacher's judgements cached)",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    model = AutoModel.from_pretrained(args.student, dtype=torch.bfloat16).to(args.device)
    if args.layers and args.layers < len(model.layers):
        # A depth-pruned Qwen.  The cost of a Qwen3 decoder is its layers
        # (the embedding table is a lookup, not a matmul), so the depth is
        # the knob that decides whether the drone can afford it: 28
        # layers at 1024 hidden measured 223 ms a chunk on the CPU, and
        # the stock MiniLM (3 layers at 384) measures 2.8 ms.  The
        # tokenizer, the pretraining and the pooling stay Qwen-family; the
        # depth is ours to choose, and the teacher's judgements decide
        # what the truncated model still knows.
        model.layers = model.layers[:args.layers]
        model.config.num_hidden_layers = len(model.layers)
        # the config carries a per-layer type list; it has to match the
        # depth or the checkpoint refuses to save (the validation is
        # strict: num_hidden_layers must equal len(layer_types))
        if getattr(model.config, "layer_types", None):
            model.config.layer_types = list(model.config.layer_types)[:args.layers]
        print(f"the student is pruned to {len(model.layers)} layers", flush=True)
    model.gradient_checkpointing_enable()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = max(1, len(pairs) // args.batch_size) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.1,
        anneal_strategy="linear")

    def embed(texts: list[str]) -> torch.Tensor:
        """The student's vector: last-token pooling, then unit length."""
        batch = tokenizer(texts, padding=True, truncation=True,
                          max_length=args.max_length, return_tensors="pt")
        batch = {k: v.to(args.device) for k, v in batch.items()}
        out = model(**batch).last_hidden_state
        lengths = batch["attention_mask"].sum(dim=1) - 1
        picked = out[torch.arange(out.size(0), device=out.device), lengths]
        return F.normalize(picked.float(), dim=-1)

    order = list(range(len(pairs)))
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(order)
        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            idx = order[i:i + args.batch_size]
            batch_pairs = [pairs[j] for j in idx]
            batch_teacher = [judgements[j] for j in idx]
            a_vec = embed([p["a"] for p in batch_pairs])
            p_vec = embed([p["b"] for p in batch_pairs])
            n_vec = embed([p["negative"] for p in batch_pairs])
            cos_pos = (a_vec * p_vec).sum(dim=-1)
            cos_neg = (a_vec * n_vec).sum(dim=-1)
            # 1. the hard labels, as a hinge at the operating point
            dup = torch.tensor([1.0 if p["duplicate"] else 0.0 for p in batch_pairs],
                               device=args.device)
            hinge = (dup * F.relu(OPERATING_POINT - cos_pos)
                     + (1 - dup) * F.relu(cos_neg - OPERATING_POINT)).mean()
            # 2. the teacher's relation structure, as the soft target
            t_pos = torch.tensor([t["cos_pos"] for t in batch_teacher], device=args.device)
            t_neg = torch.tensor([t["cos_neg"] for t in batch_teacher], device=args.device)
            kd = F.mse_loss(cos_pos, t_pos) + F.mse_loss(cos_neg, t_neg)
            loss = hinge + args.kd_weight * kd
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if step < steps - 1:
                scheduler.step()
            step += 1
            if step % 10 == 0:
                print(f"  step {step}/{steps} the hinge {hinge.item():.4f} "
                      f"the kd {kd.item():.4f} ({time.time() - t0:.0f}s)", flush=True)

    checkpoint = args.out / "qwen-bi-encoder"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    print(f"saved {checkpoint}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# stage 3: the gate (the CPU venv)
# ---------------------------------------------------------------------------
def stage_measure(args) -> int:
    from experiments.encoder_suite import (build_drift_cases, load_conversations,
                                           make_encoder, measure_cost,
                                           measure_dedup, measure_drift)

    conversations = load_conversations(args.corpus)
    train_convs, holdout = split_conversations(conversations, seed=args.seed)
    pairs = build_pairs(holdout, args.per_conv, args.seed)
    drift_cases = build_drift_cases(holdout)
    print(f"holdout: {len(pairs)} pairs, {len(drift_cases)} drift cases", flush=True)
    checkpoint = args.out / "qwen-bi-encoder"
    results: dict[str, Any] = {"the_holdout_conversations": len(holdout),
                               "the_checkpoint": str(checkpoint),
                               "the_operating_point": OPERATING_POINT,
                               "the_encoders": {}}
    for label, name in (("the stock MiniLM", "sentence-transformers/paraphrase-MiniLM-L3-v2"),
                        ("the Qwen teacher", TEACHER),
                        ("our student", "qwen-student:" + str(checkpoint))):
        embed_fn, info = make_encoder(name)
        dedup = measure_dedup(pairs, embed_fn)
        drift = measure_drift(drift_cases, embed_fn)
        cost = measure_cost(embed_fn, [c["text"] for c in holdout[0]["chunks"]])
        results["the_encoders"][label] = {"info": info, "dedup": dedup,
                                          "drift": drift, "cost": cost}
        print(f"{label}: dedup@0.92 f1 {dedup['at_0.92']['f1']} "
              f"(precision {dedup['at_0.92']['precision']}, "
              f"recall {dedup['at_0.92']['recall']}) | "
              f"dedup best {dedup['best']['f1']} | "
              f"drift auc {drift['auc']} | {cost.get('ms_per_chunk')} ms/chunk", flush=True)
    (args.out / "qwen-encoder.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out / 'qwen-encoder.json'}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("teacher", "train", "measure"), required=True)
    ap.add_argument("--corpus", nargs="+",
                    default=[str(REPO / "harness_state"),
                             str(REPO.parent / "splinter-memory" / "harness_state")])
    ap.add_argument("--student", default=STUDENT)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kd-weight", type=float, default=1.0,
                    help="the teacher's relation structure's share of the loss")
    ap.add_argument("--per-conv", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--layers", type=int, default=0,
                    help="the student's depth: 0 = the base model's own, "
                         "N = the first N layers (the cost knob: the "
                         "embedding table is a lookup, the layers are the "
                         "matmul)")
    ap.add_argument("--out", type=Path,
                    default=REPO / "experiments/encoder-suite/results/qwen")
    args = ap.parse_args(argv)
    return {"teacher": stage_teacher, "train": stage_train,
            "measure": stage_measure}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
