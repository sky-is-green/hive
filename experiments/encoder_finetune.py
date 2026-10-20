"""Tune the drone's encoder on the harness's own traffic.

The scion recipe's shape, in the encoder's idiom: **a teacher, a student, a
curriculum, a gate, an export**.  What changes is the loss - a ternary
quantisation recipe distils next-token logits; an encoder distils *which
texts should be close and which should be far*.

The student is the drone's model (``paraphrase-MiniLM-L3-v2``, 17M params):
the harness's budget is a CPU encoder that costs about five milliseconds a
turn, and that number is not negotiable.  The teacher is a bigger encoder
whose judgement is distilled into the student.

Three tasks, one loss.  The harness gives its encoder three jobs, and each
one is a statement about a cosine:

``rank``   a query and the chunk that answers it should be close
``dedup``  a chunk and the chunk that says the same thing should be close
``drift``  a recent window and an older window of the same topic should be
           close; a recent window and an older window of another topic
           should be far

All three become in-batch-negative contrastive examples: the anchor and
the positive are one pair, the other pairs in the batch are the negatives.
Nothing about the drone's architecture changes - it still embeds a text
once and compares cosines - so the tuned model is a drop-in replacement
for the one the Studio loads.

The data is the harness's own traffic (the conversations the Studio and
the sibling actually served), labelled lexically so the label does not
come from the encoder being trained:

  - the duplicate pairs: the same content words
  - the drift triples: the same content words (same topic) and none
    (different topic)

The teacher is optional.  With one, the teacher's similarity matrix is the
soft target (the distillation); without one, the labels are the target.

Usage::

    # the student, the labels only
    python -m experiments.encoder_finetune --student l3 --out models/encoder/l3-tuned

    # the student distilled from the teacher
    python -m experiments.encoder_finetune --student l3 --teacher bge-m3 \\
        --out models/encoder/l3-bge3-tuned --epochs 3

    # the gate: measure the checkpoint against the baseline
    python -m experiments.encoder_finetune --evaluate models/encoder/l3-tuned
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]

#: the three tasks, with how much of the batch each one takes
TASK_WEIGHTS = {"rank": 0.0, "dedup": 0.6, "drift": 0.4}


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------
def build_examples(conversations: list[dict], seed: int = 20261020
                   ) -> list[dict]:
    """The training examples, from the harness's own conversations.

    Each example is ``{"anchor": str, "positive": str, "negative": str,
    "task": str}`` - the in-batch negatives are the other examples'
    positives, so an example only needs one hard negative of its own.
    """
    from experiments.encoder_suite import (build_dedup_pairs, content_terms,
                                           jaccard, load_conversations)

    rng = random.Random(seed)
    examples: list[dict] = []

    # --- dedup: the same fact, said twice -------------------------------
    for pair in build_dedup_pairs(conversations, per_conv=20, seed=seed):
        if not pair["duplicate"]:
            continue
        # the positive is the pair itself; the negative is a chunk from
        # another conversation (a different fact)
        other = rng.choice(conversations)
        if not other["chunks"]:
            continue
        negative = rng.choice(other["chunks"])["text"]
        examples.append({"anchor": pair["a"], "positive": pair["b"],
                         "negative": negative, "task": "dedup"})

    # --- drift: same topic close, other topic far -----------------------
    for conv in conversations:
        chunks = conv["chunks"]
        if len(chunks) < 8:
            continue
        terms = [content_terms(c["text"]) for c in chunks]
        for cut in range(4, len(chunks) - 2, 2):
            recent = chunks[cut:cut + 3]
            recent_terms = set().union(*terms[cut:cut + 3])
            older = list(range(0, max(1, cut - 1)))
            if not older:
                continue
            # the most similar older chunk is the same topic; the least
            # similar is another topic.  An example where the two are
            # too close together is ambiguous and is dropped.
            scored = sorted(((jaccard(recent_terms, terms[i]), i)
                             for i in older), reverse=True)
            best_jac, best = scored[0]
            worst_jac, worst = scored[-1]
            if best_jac < 0.08 or best_jac - worst_jac < 0.08:
                continue
            examples.append({
                "anchor": " ".join(c["text"] for c in recent),
                "positive": chunks[best]["text"],
                "negative": chunks[worst]["text"],
                "task": "drift",
            })
    rng.shuffle(examples)
    return examples


def build_harvest_examples(records: list[dict]) -> list[dict]:
    """The harvested turns: the query and the chunk the curation chose.

    The label here is the *harness's own decision*, so this task
    distils the current system rather than improving on it.  It is kept
    separate for that reason: the weight is zero by default, and the
    teacher supplies the ranking target when one is configured.
    """
    examples: list[dict] = []
    for record in records:
        query = (record.get("query") or "").strip()
        selected = record.get("selected") or []
        candidates = record.get("candidates") or {}
        if not query or not selected:
            continue
        positives = [cid for cid in selected]
        negatives = [cid for cid in candidates if cid not in set(positives)]
        if not negatives:
            continue
        # the chunk texts are not in the record; the store has them, so
        # the caller fills them in - here we only count what is usable
        examples.append({"query": query, "positives": positives,
                         "negatives": negatives})
    return examples


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------
def in_batch_loss(anchor: torch.Tensor, positive: torch.Tensor,
                  temperature: float = 0.05) -> torch.Tensor:
    """MultipleNegativesRankingLoss, by hand (no ``datasets`` needed)."""
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    sim = a @ p.t() / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


def kd_loss(student: torch.Tensor, teacher: torch.Tensor,
            labels: torch.Tensor, temperature: float = 0.05,
            alpha: float = 0.5) -> torch.Tensor:
    """The teacher's similarity matrix as the soft target.

    ``alpha`` is the teacher's share: at 0.5 the student is pulled half
    toward the hard labels and half toward the teacher's distribution.
    """
    t = F.normalize(teacher, dim=-1) @ F.normalize(teacher, dim=-1).t() / temperature
    s = F.normalize(student, dim=-1) @ F.normalize(student, dim=-1).t() / temperature
    soft = F.kl_div(F.log_softmax(s, dim=-1),
                    F.softmax(t.detach(), dim=-1),
                    reduction="batchmean")
    hard = F.cross_entropy(s, labels)
    return alpha * soft + (1.0 - alpha) * hard


# ---------------------------------------------------------------------------
# The training
# ---------------------------------------------------------------------------
def train(args) -> dict:
    from sentence_transformers import SentenceTransformer

    from experiments.encoder_suite import load_conversations

    conversations = load_conversations(args.corpus)
    if not conversations:
        raise SystemExit("no conversations found; pass --corpus")
    examples = build_examples(conversations, seed=args.seed)
    print(f"examples: {len(examples)} "
          f"({sum(1 for e in examples if e['task'] == 'dedup')} dedup, "
          f"{sum(1 for e in examples if e['task'] == 'drift')} drift)")

    student = SentenceTransformer(args.student, device="cpu")
    teacher = None
    if args.teacher:
        print(f"teacher: {args.teacher}")
        teacher = SentenceTransformer(args.teacher, device="cpu")

    # the model, as a torch module, so the loop can backpropagate
    module = student[0].auto_model
    module.train()
    params = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = max(1, len(examples) // args.batch_size) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.1,
        anneal_strategy="linear")

    def embed(texts: list[str], model, tokenizer) -> torch.Tensor:
        batch = tokenizer(texts, padding=True, truncation=True, max_length=256,
                          return_tensors="pt")
        out = model(**batch)
        # mean pooling over the tokens (the sentence-transformers default)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    student_tokenizer = student.tokenizer
    teacher_tokenizer = teacher.tokenizer if teacher is not None else None

    history: list[dict] = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        random.shuffle(examples)
        for i in range(0, len(examples) - args.batch_size + 1, args.batch_size):
            batch = examples[i:i + args.batch_size]
            anchors = [e["anchor"] for e in batch]
            positives = [e["positive"] for e in batch]
            a_vec = embed(anchors, module, student_tokenizer)
            p_vec = embed(positives, module, student_tokenizer)
            if teacher is not None:
                with torch.no_grad():
                    t_vec = embed(positives, teacher[0].auto_model, teacher_tokenizer)
                loss = kd_loss(p_vec, t_vec,
                               torch.arange(p_vec.size(0)), temperature=args.temperature,
                               alpha=args.alpha)
            else:
                loss = in_batch_loss(a_vec, p_vec, temperature=args.temperature)
            # the hard negative: the anchor should be far from it
            negatives = [e["negative"] for e in batch]
            n_vec = embed(negatives, module, student_tokenizer)
            neg_sim = (F.normalize(a_vec, dim=-1)
                       * F.normalize(n_vec, dim=-1)).sum(dim=-1)
            hard = F.relu(neg_sim - 0.8).mean()
            loss = loss + args.hard_weight * hard

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if step < steps - 1:
                scheduler.step()
            step += 1
            if step % args.log_every == 0:
                print(f"  step {step}/{steps} loss {loss.item():.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        history.append({"epoch": epoch, "steps": step})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    student.save(str(out))
    meta = {
        "student": args.student,
        "teacher": args.teacher,
        "examples": len(examples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "hard_weight": args.hard_weight,
        "seed": args.seed,
        "seconds": round(time.time() - t0, 1),
        "history": history,
    }
    (out / "finetune_meta.json").write_text(json.dumps(meta, indent=1),
                                             encoding="utf-8")
    print(f"saved {out}")
    return meta


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def evaluate(args) -> dict:
    """Measure a checkpoint against the baseline, with the harness's tasks.

    The gate: the tuned model must not lose on any of the three jobs, and
    it must stay inside the CPU budget.  A checkpoint that wins one job
    and loses another is not a release.
    """
    from experiments.encoder_suite import (build_dedup_pairs, build_drift_cases,
                                           load_conversations, make_encoder)

    conversations = load_conversations(args.corpus)
    pairs = build_dedup_pairs(conversations, per_conv=args.per_conv)
    cases = build_drift_cases(conversations)
    out: dict = {"checkpoint": args.checkpoint, "baseline": args.baseline}
    for label, name in (("checkpoint", args.checkpoint), ("baseline", args.baseline)):
        if not name:
            continue
        if name.startswith("checkpoint:"):
            embed, info = make_encoder("checkpoint:" + name.split(":", 1)[1])
        else:
            embed, info = make_encoder(name)
        from experiments.encoder_suite import measure_cost, measure_dedup, measure_drift

        dedup = measure_dedup(pairs, embed)
        drift = measure_drift(cases, embed)
        cost = measure_cost(embed, [c["text"] for c in conversations[0]["chunks"]])
        out[label] = {"info": info, "dedup": dedup, "drift": drift, "cost": cost}
        print(f"{label}: dedup f1@0.92 {dedup['at_0.92']['f1']} "
              f"best {dedup['best']['f1']} | drift auc {drift['auc']} | "
              f"{cost.get('ms_per_chunk')} ms/chunk")
    if "checkpoint" in out and "baseline" in out:
        c, b = out["checkpoint"], out["baseline"]
        out["verdict"] = {
            "dedup_f1": c["dedup"]["at_0.92"]["f1"] >= b["dedup"]["at_0.92"]["f1"],
            "dedup_best_f1": c["dedup"]["best"]["f1"] >= b["dedup"]["best"]["f1"],
            "drift_auc": c["drift"]["auc"] >= b["drift"]["auc"],
            "cost_ok": (c["cost"].get("ms_per_chunk", 99)
                        <= 2.0 * b["cost"].get("ms_per_chunk", 1)),
        }
        out["verdict"]["pass"] = all(out["verdict"].values())
        print("verdict:", out["verdict"])
    print(json.dumps(out, indent=1)[:4000])
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", default="sentence-transformers/paraphrase-MiniLM-L3-v2")
    parser.add_argument("--teacher", default="")
    parser.add_argument("--corpus", nargs="+",
                        default=[str(REPO_ROOT / "harness_state"),
                                 str(REPO_ROOT.parent / "splinter-memory" / "harness_state")])
    parser.add_argument("--out", default="models/encoder/tuned")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="the teacher's share of the loss")
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--evaluate", default="",
                        help="measure a checkpoint instead of training")
    parser.add_argument("--baseline", default="l3",
                        help="the encoder to compare the checkpoint against")
    parser.add_argument("--per-conv", type=int, default=20)
    args = parser.parse_args(argv)
    if args.evaluate:
        evaluate(args)
        return 0
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
