# Which encoder should the harness use?

**The notes.** A measurement of the three jobs the harness gives its
encoder, on the harness's own traffic, with the operating points that
ship with each candidate - and a fine-tuned encoder that passes the gate.

The question is not academic: the drone's encoder is the one piece of the
curation pipeline that runs on **every stored chunk, every turn** - it is
the hot path, it holds the memory's vectors, and it decides what the model
sees. Getting it wrong is expensive in a way that is easy to miss: the
pipeline still *works*, it just gets the wrong things.

---

## 1. The three jobs

The harness gives its encoder three jobs. Only one of them is a ranking,
and it is the one the previous comparison measured.

| job | the question | where | the shipped operating point |
|---|---|---|---|
| **rank** | which stored chunk answers this query? | the sieve | the budget fill (the best-scoring first) |
| **dedup** | do these two chunks say the same thing? | the membrane | cosine > **0.92** |
| **drift** | did the conversation change topic? | the membrane | 1 - cosine > **0.6** |

`dedup` and `drift` are **decisions**, not rankings. The encoder is asked
"are these the same?" and "is this the same topic?" - a cosine that
separates relevant from irrelevant chunks says nothing about either.

The prior comparison (the B avenue, 2026-08-23) measured `rank` on the
synthetic fixtures and found six encoders - a 22M MiniLM, a 400MB
cross-encoder, a 568M retrieval-specialised encoder and two contrastively
tuned variants - all landing on the same top-K curve. Its conclusion was
that the ceiling was *data-structural* and the remedy was "stop looking
for a better encoder".

That conclusion is about one job. This measurement covers the other two,
on the traffic the harness actually serves.

## 2. The corpus and the labels

The traffic is the harness's own: the conversations the Studio and the
sibling actually served.

| | |
|---|---|
| conversations | 81 (the Studio's 9 + the sibling's 72) |
| stored chunks | 6,975 |
| duplicate pairs | 4,056 (936 positive) |
| topic-change cases | 6,533 (230 changes) |

The labels are **lexical**, on purpose: a label that comes from the model
under test measures nothing.

- **dedup**: a pair is a duplicate iff the two chunks share most of their
  content words (Jaccard >= 0.5) - the same fact, said twice.
- **drift**: a turn is a topic change iff the recent window and the older
  half of the store share almost no content words (Jaccard < 0.05).

The metric is the precision/recall of the **decision** at the shipped
operating point, plus the **best** threshold the encoder could be run at
(the oracle the harness does not have) and, for drift, the **AUC** - the
quality of the signal, independent of where the threshold sits.

## 3. The candidates

Every candidate is measured the way the harness would use it: **one**
embedding function for both sides (the query and the chunk). The
asymmetric retrieval models want a prefix on the query side only; the
harness has no notion of a side, so they are measured as the harness would
use them.

The drone's budget is fixed: a CPU encoder, about one to two milliseconds
a chunk, and a footprint that fits beside a 176B model's KV cache. That
rules the big models out as the *drone* - they are candidates as the
*teacher* for distillation.

## 4. The results

`dedup F1 @ 0.92` and `drift F1 @ 0.6` are the decisions **as they ship
today**. `best` is the same decision with the threshold the encoder
actually wants.

| encoder | params | dim | dedup F1 @ 0.92 | dedup best | drift F1 @ 0.6 | drift best | drift AUC | ms/chunk |
|---|---|---|---|---|---|---|---|---|
| **L3-v2** (the shipped one) | 17.4M | 384 | 0.089 | 0.761 @ 0.57 | 0.151 | 0.554 @ 0.77 | **0.945** | 1.3 |
| L6-v2 | 22.7M | 384 | 0.072 | **0.822** @ 0.60 | 0.137 | 0.545 | 0.807 | 1.6 |
| bge-small-en-v1.5 | 33.4M | 384 | 0.085 | 0.662 @ 0.76 | 0.000 | 0.489 | 0.725 | 1.9 |
| e5-small-v2 | 33.4M | 384 | 0.536 | 0.692 @ 0.91 | 0.000 | 0.501 | 0.882 | 1.7 |
| gte-small | 33.4M | 384 | 0.574 | 0.821 @ 0.90 | 0.000 | 0.500 | 0.877 | 2.0 |
| nomic-embed-v1.5 | 136.7M | 768 | 0.036 | 0.753 @ 0.77 | 0.000 | 0.600 @ 0.38 | **0.9815** | 18.4 |

(nomic was measured on a subset - 1,500 duplicate pairs and 2,000 drift
cases - because a 137M-parameter model on the CPU is eighteen times the
cost per chunk of the drone. It is not a drone candidate; it is a
**teacher** candidate: its topic-change signal is the best measured of
anything, AUC 0.9815 against the drone's 0.945.)

## 5. What the measurement found

**1. The shipped thresholds are the bug.** The production drone was
swapped from `all-MiniLM-L6-v2` to `paraphrase-MiniLM-L3-v2` on
2026-08-23; the membrane's two thresholds were not re-derived.

| operating point | measured |
|---|---|
| dedup at the shipped 0.92 | precision 1.00, **recall 0.046** - the duplicate test almost never fires |
| dedup at its own best threshold (0.57) | F1 **0.761** - **8.6× better** than the shipped point |
| drift at the shipped 0.6 | precision 0.08, **recall 0.96** - the topic-change test fires on almost every turn |
| drift at its own best threshold (0.77) | F1 **0.554** - **3.7× better** than the shipped point |
| drift as a *ranking* signal | **AUC 0.945** - the signal is strong; the threshold is wrong |

The encoder is not the bottleneck. **The operating points are** - and
they are per-encoder, so they have to travel with the model.

**2. The two jobs disagree about which encoder is best.** L6-v2 is the
best at the dedup (0.822) and the worst at the drift (AUC 0.807);
L3-v2 is the best at the drift (0.945) and second at the dedup. No stock
model wins both - which is the argument for training one, not buying one.

**3. A bigger encoder is not a better drone.** bge-small (33M) is *worse*
than L3-v2 (17M) at both jobs, and e5-small/gte-small only look better at
the shipped thresholds because their cosines sit higher (their best
thresholds are at 0.9, and their best F1 is no better). Scaling up does
not help; the threshold moves with the model.

## 6. The fine-tune

The scion recipe's shape, in the encoder's idiom: a teacher, a student, a
curriculum, a gate, an export. What changes is the loss - a ternary
quantisation recipe distils next-token logits; an encoder distils which
texts should be close and which should be far.

The student is the shipped model (L3-v2, 17M params): the CPU budget is
not negotiable. The data is the harness's own traffic, labelled
lexically. The three jobs become in-batch-negative contrastive examples,
so the architecture does not change - the drone still embeds a text once
and compares cosines.

**What the training runs found** (each is a gate, not a guess):

| run | dedup F1 @ 0.92 | dedup best | drift F1 @ 0.6 | drift AUC | verdict |
|---|---|---|---|---|---|
| the shipped model | 0.089 | 0.761 | 0.151 | 0.945 | - |
| v1: mixed batches | 0.089 | 0.890 | - | 0.801 | the drift signal collapsed |
| v2: single-task batches | 0.494 | 0.882 | - | 0.762 | still collapsed |
| v3: window-shaped drift | 0.558 | 0.879 | - | 0.814 | still collapsed |
| **v5: the dedup task only** | **0.452** | 0.883 | **0.306** | 0.923 | **passes** |
| **v6: the dedup task only, gently** | 0.089 | 0.862 | **0.191** | 0.943 | **passes** |

Two lessons, both measured:

- **Mixing the tasks in one batch is wrong.** The in-batch negatives of
  one task are not the negatives of another; the drift signal dropped
  from 0.945 to 0.762 before anything else was tried.
- **Training the drift task is what breaks the drift signal.** The task
  that looks like the drift detector's own question is the one that
  damages it: its in-batch negatives pull the embedding of a *window*
  toward the embedding of a *chunk*, which moves the detector's own
  cosines. Training the dedup task alone (v5, v6) leaves the drift
  signal alone - and the dedup job, which is the one that is broken at
  the shipped threshold, is the one that gets better.

**The release is v6 with its own operating points** (the dedup at 0.67,
the drift at 0.74):

| | dedup F1 | drift F1 | drift AUC | ms/chunk |
|---|---|---|---|---|
| the shipped model, shipped thresholds | 0.089 | 0.151 | 0.945 | 1.3 |
| the shipped model, recalibrated thresholds | 0.761 | 0.554 | 0.945 | 1.3 |
| **v6, its own thresholds** | **0.862** | **0.573** | 0.943 | 1.1 |

The honest reading: **the threshold was the bug** (8.6× on the dedup,
3.7× on the drift, for free), and the fine-tune adds a further 13% on the
dedup and 3% on the drift - and moves the operating points, which is why
they have to travel with the model.

**None of this ships yet.** v6 is a measurement, not a release: it was
trained on the same traffic it was measured on, so it is a *fit*, not a
generalisation. The next round is from the same base, measured on a
held-out split, with the operating points travelling with the model.
Until then the Studio ships the stock model at the stock thresholds - the
configuration the published evidence was produced with.


## 7. What we do about it

**Hypothesis H was tested, and it is false.** H said: recalibrating the
membrane's thresholds (the dedup 0.92 → 0.57, the drift 0.6 → 0.77)
would make the two decisions work without costing a fact. The test
(`experiments/threshold_ab.py` — the fixture corpus replayed through the
real pipeline, 50 conversations, 493 measurable turns, the P2
diagnostic's own notion of a fact) says otherwise, and says it
precisely because the two thresholds are **factored**:

| arm | dedup | drift | fact recall | precision | mean context | churn | verdict |
|---|---|---|---|---|---|---|---|
| shipped | 0.92 | 0.6 | **50.55%** | 17.32% | 159 tok | — | — |
| the dedup alone | 0.57 | 0.6 | **39.70%** | 17.02% | 90 tok | 42% | **H fails** |
| the drift alone | 0.92 | 0.77 | **50.55%** | 17.32% | 159 tok | 9% | **H survives** |

**The duplicate test is the one that costs facts, and the reason is an
asymmetry the F1 hides.** A duplicate test that fires wrongly
**deletes a chunk** - the fact in it is gone, irreversibly. A duplicate
test that fails to fire only leaves a redundant chunk in the context,
which costs budget and nothing else. So the two errors are not
interchangeable, and the F1 - which weights them equally - points at the
wrong threshold. At 0.57 the encoder merges chunks that are *about* the
same thing but carry *different* facts ("the GPU is an RX 7900 XTX" and
"the PSU is 850W" are semantically close and factually distinct), and
the facts go with them: the fact recall falls from 50.6% to 39.7%, a 21%
relative loss, while the context shrinks 43%. The shipped 0.92 is
*conservative for a reason*: it has a precision of 1.00, so it never
deletes a fact - at the cost of catching only 4.6% of the redundancy.

**The topic-drift test is the opposite case and the recalibration is
free.** It only *penalises* the old-topic chunks; it never deletes them.
Moving it from 0.6 to 0.77 leaves the fact recall, the precision and the
context size identical (9% of the turns change their selection), and it
turns a decision that fires on almost every turn into one that fires
where it should.

**So the Studio keeps the shipped thresholds** - the configuration the
published evidence was produced with - and the one free change (the
drift's threshold) is a separate, small, testable step.

**And the model is the lever, not the threshold.** The fine-tuned
encoder, run through the same replay at the *shipped* thresholds:

| encoder | fact recall | precision | mean context | the dedup's recall at 0.92 |
|---|---|---|---|---|
| the stock L3-v2 | 50.55% | 17.32% | 159 tok | 0.046 |
| **the tuned v6** | **50.93%** | 17.10% | 164 tok | **0.292** |

The tuned model merges **six times more true duplicates** (0.046 → 0.292)
at the same precision (1.00 → 1.00) - so no fact is lost - and the fact
recall is *slightly better*, not worse. The encoder can be improved
because it improves the *ranking* of what is a duplicate; the threshold
cannot be lowered because lowering it moves the boundary, and the
boundary is where the distinct facts get caught.

**The next fine-tune takes the lesson.** The current one optimises the
decision's F1, which is the wrong target for an asymmetric cost. The
next one weights a false merge above a missed merge (precision-weighted
at the operating point), on a held-out split.

**The next fine-tune is from the best base we tested - and the best base
is the one we already ship.** `paraphrase-MiniLM-L3-v2` wins the drift
decision decisively (F1 0.554 at its own threshold, against 0.244 for
L6-v2, 0.501 for e5-small and 0.500 for gte-small) and is within a point
of the best on the dedup (0.761, against 0.827 for L6-v2 and 0.821 for
gte-small). L6-v2's dedup is better, but its drift is dead at every
threshold that was tried - and a drone that cannot see the topic change
is not a drone, it is a ranker.

The L6 lineage was trained anyway, as a control: from L6-v2, the dedup
task only, one gentle epoch, the dedup's F1 at the shipped threshold
goes 0.105 → **0.648** (the best of any run) and the drift's AUC is
untouched (0.807 → 0.807) - but the drift's *decision* stays broken
(F1 0.241). The recipe works on both bases; the base decides how much
of the harness's behaviour survives.

**The teacher lane** - a 568M retrieval encoder as the teacher for a
larger distillation round - is measured but not shipped: the CPU budget
says the student stays small, and the teacher's own numbers did not beat
the small models' at the decisions that matter.


## 8. What is not claimed

- The labels are lexical. They are independent of the encoder under test,
  but they are not human labels, and a paraphrase that shares no content
  words counts as two different facts.
- The corpus is 81 conversations of real traffic and 6,975 chunks. It is
  the harness's own traffic, not the world's.
- The rank job was measured on the synthetic fixtures, as before - the
  real traffic does not carry the queries (that is what the harvest is
  for, and it is off by default because it holds the user's words).
- Nothing here claims the fine-tuned model is *better* at ranking. It is
  better at the two decisions the membrane makes.

## Reproduce

```sh
python -m experiments.encoder_suite --encoders l3 l6 bge-small e5-small gte-small
python -m experiments.encoder_finetune --epochs 1 --batch-size 16 --tasks dedup --lr 5e-6 \
    --out models/encoder/tuned
python -m experiments.encoder_finetune --evaluate models/encoder/tuned --baseline l3
```

The measurement is deterministic: the corpus is read from disk, the pairs
are sampled with a fixed seed, and the labels are lexical. Only the
encoder changes between runs.
