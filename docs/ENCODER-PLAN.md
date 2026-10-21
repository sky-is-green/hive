# The encoder, part two: match the generator, or train our own

**The question.** The curation selects chunks in the *drone's* embedding
space, and the generator reads them as *text* in its own space. The
generators are Qwen-family (the 35B-A3B is a Qwen3.5 MoE, the 176B a
Qwen4 exp); the drone has always been a BERT-family bi-encoder (MiniLM,
bge, e5, gte) or a Mistral-family one (nomic). The two spaces are
unrelated.

**Why that might be the whole story.** On the 27B the curated context
won 115 turns outright to 26; on the 35B it is 13 to 17 - the arms tie.
The hypothesis: the selection is made in a space the generator does not
share, so a stronger generator gains nothing from it. The measurement in
§2A.2 of the results log tests it directly (the same tasks, the same
generator, the curation's encoder swapped for `Qwen/Qwen3-Embedding-0.6B`).

---

## Option A: adopt a Qwen-family bi-encoder

**What it is.** `Qwen/Qwen3-Embedding-0.6B` is a Qwen3 decoder trained as
a bi-encoder: the same tokenizer, the same pretraining, the same
dimensions (1024) as the generator's family, and a contrastive objective
on top. It is the closest thing to "the generator's own space" that
exists off the shelf.

**What it costs.** 600M parameters against the drone's 17M - 35× the
memory, and on the CPU roughly 80 ms a chunk against 1.7 ms. The drone
scores *every stored chunk* on *every turn*, so the curation's cost - which
the stability run already showed is linear in the store - would go from
157 ms a turn at 1,600 chunks to hours. It is a quality ceiling, not a
shippable drone.

**What it is good for.** The **teacher** for a distilled student: it
knows the generator's semantics ✓, and its judgements are what the
student should reproduce ✓.

## Option B: train our own Qwen-family bi-encoder

**The student.** A small Qwen3 decoder (0.6B - the same family, the same
tokenizer) trained as a bi-encoder: mean pooling over the last hidden
state, a contrastive objective, and the harness's own data.

**The teacher.** `Qwen/Qwen3-Embedding-0.6B` (bidirectional ✓ - a causal
model cannot be the teacher for a bi-encoder ✗ - the attention mask is
the whole difference ✓).

**The objective.** Two terms, and the second is the one the measurements
say matters:

1. **The hard labels** - the lexical ones the suite already computes (the
   same content words = a duplicate; the same topic = not a drift) ✓.
2. **A relation distillation** - the teacher's similarity *structure*
   over the (query, chunk) pairs as the soft target ✓. The teacher's
   judgements are what the student reproduces, so the student inherits the
   generator-family semantics rather than the teacher's own quirks.

**The operating point, not the F1.** The measurements are unambiguous
here: the threshold that maximises the decision's F1 is *not* the
threshold that preserves the facts (§7.1 of the encoder notes), because a
false merge deletes a fact while a missed merge only costs budget. So the
objective must be a **hinge at the operating point** - the true
duplicates' cosine above 0.92, the distinct pairs' below it - and the
gate must measure the *decision at the shipped threshold*, not the
threshold-free ranking.

**The gate.** The held-out split (`experiments/encoder_holdout.py`): train
on half the conversations, measure on the other half, and require that no
job regresses and that the cost holds. The current checkpoint passes the
*separation* test and fails the *cash-in* test, and the reason is the
threshold, not the separation.

**The data.** The harvest (the real turns, the query, the reply and the
curation that was used - off by default, the user's call), the corpus
(6,975 stored chunks from 81 real conversations), and the teacher's
labels over both.

**What the scion recipe contributes here.** Its *shape*, not its
machinery: a teacher, a student, a curriculum, a gate, an export, and
the discipline of a pre-registered decision rule. Its ternary STE/KD
machinery is built for causal decoders with an LM head and does not
transfer to a pooled bi-encoder - but the *ternary* idea does: a drone
that has been quantised is a drone that can be bigger. That is a
separate experiment, after the quality question is settled.

---

## The order

1. **Measure** the Qwen-family bi-encoder on the three jobs and in the
   paired A/B. *Done - results log 2B, 2A.2, 2C.*
2. **If the architecture matters**, train the student with the
   relation-KD objective on the harness's own data, with the hinge at the
   operating point. *Done - the 4-layer student, results log 2C-ter.*
3. **Gate it** on the held-out split, and measure the *decision at the
   shipped threshold* - the thing the first fine-tune got wrong.
   *Done - and then the decision was priced on the curation itself
   (results log 2C-quater): moving it deletes facts.*
4. **Only then** look at quantising it, and only if the quality is there
   to keep. *Not reached: nothing wins both jobs at the shipped cost, so
   there is no quality to keep.*

## What the measurement decided

The family-match is real: the teacher is the same family as the
generator, and a student trained on our data with the decision as the
objective beats it at that decision by 9.4x. But the decision is not the
product. The replay closed it: **no measured candidate improves the
fact-per-token trade-off at a calibrated point**, and the reason is that
a false merge deletes a fact. A better ranker moves along that frontier,
not off it.

So Option A stays a teacher, Option B stays a ceiling, and the only
shippable move in this path is the one the replay leaves open: give the
**drift** decision its own encoder, because its threshold is free. The
shipped drone does not change on the strength of what has been measured.
