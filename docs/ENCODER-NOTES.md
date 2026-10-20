# Which encoder should the harness use?

**The notes.** A measurement of the three jobs the harness gives its
encoder, on the harness's own traffic, with the operating points that
ship with each candidate.

The question is not academic: the drone's encoder is the one piece of the
curation pipeline that runs on **every stored chunk, every turn** - it is
the hot path, it holds the KV of the memory, and it decides what the
model sees. Getting it wrong is expensive in a way that is easy to miss:
the pipeline still *works*, it just gets the wrong things.

---

## 1. The three jobs

The harness gives its encoder three jobs. Only one of them is a ranking,
and it is the one the previous comparison measured.

| job | the question | where | the operating point |
|---|---|---|---|
| **rank** | which stored chunk answers this query? | the sieve | the budget fill (top-scoring first) |
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

That conclusion is about one job. This measurement covers the other two.

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

The metric is the precision/recall of the decision at the harness's
operating point, plus the **best** threshold the encoder could be run at
(the oracle the harness does not have) and, for drift, the **AUC** - the
quality of the signal, independent of where the threshold sits.

## 3. The candidates

Every candidate is measured the way the harness would use it: **one**
embedding function for both sides (the query and the chunk). The
asymmetric retrieval models want a prefix on the query side only; the
harness has no notion of a side, so they are measured as the harness would
use them, and (where it matters) with the prefix as well.

The drone's budget is fixed: a CPU encoder, about five milliseconds a
turn, and a footprint that fits beside a 176B model's KV cache. That
rules the big models out as the *drone* - they are candidates as the
*teacher* for distillation.

## 4. The results

<!-- the table is written by the sweep; see experiments/encoder-suite/results/ -->

## 5. What the measurement found

<!-- the findings, once the sweep is in -->

## 6. What we do about it

<!-- the decision -->

## Reproduce

```sh
python -m experiments.encoder_suite --encoders l3 l6 bge-small e5-small gte-small nomic
python -m experiments.encoder_finetune --epochs 2 --out models/encoder/tuned
python -m experiments.encoder_finetune --evaluate models/encoder/tuned
```

The measurement is deterministic: the corpus is read from disk, the pairs
are sampled with a fixed seed, and the labels are lexical. Only the
encoder changes between runs.
