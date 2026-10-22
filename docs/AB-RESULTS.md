# The A/B campaign: the results so far

The running log of the campaign scoped in [`AB-PLAN.md`](AB-PLAN.md). Every
number here is a measurement, not an estimate; the run that produced it is
named, and the raw artifacts are in `experiments/cascade/results/`.

---

## 1A. The two Scions, head to head

**The setup.** Both models served at once, on the same card, answering the
same 28 tasks (the organic set the published numbers came from - 28 tasks
× 3 samples = the 66 items), one draw each, T=0.7, 640 tokens.

- **the 35B**: `Scion-35B-A3B-PQ2_0-corr.gguf` (11.3 GB, ternary), served
  by the Scion llama.cpp build on `:8082`
- **the 176B**: `Scion-FlashNext-176B-A6B-Q6_K` (the running Strata serve
  on `:8081`)
- the run: `experiments/cascade/results/dual-scion-20261020/full28/`
  (62 minutes wall, both models concurrently, 0 errors)

**The result.**

| | the 35B-A3B | the 176B-A6B |
|---|---|---|
| correct (the task's own checker) | **21/28** | 19/28 |
| latency, median (under co-hosting) | 84.9 s | 93.9 s |
| latency, p90 | 189.5 s | 213.5 s |
| the weight | 11.3 GB | ~132 GB |
| answered without error | 28/28 | 28/28 |

| the overlap | items |
|---|---|
| both correct | 13 |
| **only the 35B** | **8** |
| **only the 176B** | **6** |
| neither | 1 |
| the two answers agree | 12/28 |

**What it means, stated carefully.**

1. **The two models are equivalent on this class, within noise.** 21 vs 19
   on 28 items is not a difference (the split on the disagreements, 8 vs 6,
   is a coin flip). The 176B is **twelve times the weight** for no
   accuracy gain here. That is the tiering story's strongest form: the
   small tier can take the class outright.
2. **They disagree on 16 of 28** - so neither dominates, and the
   disagreement is the raw material for an arbiter (the tool's own
   framing). A router that can tell *which* model to trust per item is
   worth more than either model alone.
3. **The latencies are measured under co-hosting** - the two engines share
   one card, so these are the numbers that matter for the tiered
   deployment, not the isolated ones (the 8-task smoke, where the
   contention pattern differed, measured the 35B's median at 40.1 s
   against the 176B's 83.6 s). Two engines on one card cost the small one
   roughly 2× - which is the price of the tiering, and it is small next to
   the 12× weight difference.

**What it does not say.** One draw per task, 28 tasks, one class
(arithmetic, code, multiple choice). The 66-item published number (62/66
served-correct) is a different measurement - the cascade's, not the
model's - and the next arm (1B) runs the cascade over each model on the
same set.

---

## 2B. The membrane's operating points (hypothesis H)

**Falsified.** The deterministic replay of the corpus through the real
pipeline (`experiments/threshold_ab.py`, 50 conversations, 493 measurable
turns, the P2 diagnostic's own notion of a fact):

| arm | dedup / drift | fact recall | context | churn | verdict |
|---|---|---|---|---|---|
| shipped | 0.92 / 0.6 | **50.55%** | 159 tok | - | - |
| the dedup alone | 0.57 / 0.6 | **39.70%** | 90 tok | 42% | **falsified** |
| the drift alone | 0.92 / 0.77 | **50.55%** | 159 tok | 9% | **survives** |

The duplicate test's two errors are not interchangeable - a false merge
*deletes* a fact, a missed merge only costs budget - so the F1-optimal
threshold is the wrong target. The topic-drift test only penalises, so its
recalibration is free.

**The cascade over the 35B** (the same 28 tasks, the deterministic judge and
the repair leg on):

| | the 35B (28 tasks) | the 35B (10 tasks) |
|---|---|---|
| the model's own accuracy | 0.821 (23/28) | 0.800 (8/10) |
| **the cascade's served-correct** | **0.857 (24/28)** | 0.800 (8/10) |
| the repair leg | 1 attempted, **1 cleared** | 0 (nothing catchable) |
| **false rejects** | **0** | **0** |
| false accepts | 4 (0.143) | 2 (0.200) |
| **the mean per-item latency** | **6.9 s** | **5.4 s** |

The loop does its job on the small tier: one wrong answer caught and
repaired, nothing false-rejected, and the served answer is better than
the model's own.

**And the comparable pair - the same 10 tasks, the same cascade, the
two models:**

| | the 35B-A3B | the 176B-A6B |
|---|---|---|
| the model's own accuracy | 0.800 | 0.800 |
| **the cascade's served-correct** | **0.800** | **0.800** |
| the false rejects | **0** | **0** |
| the accepted-and-wrong | 2 | 2 |
| the repairs | 0 | 0 |
| **the mean per-item latency** | **5.36 s** | **229.67 s** |
| the mean tokens | 329 | 745 |

**Identical outcomes, 42.9x the wall clock.** Same accuracy, same false
rejects, same false accepts, same repairs - and the 176B spends 2.3x the
tokens thinking and 43x the seconds getting there. On this class the
small tier is not a compromise: it is the same answer, faster and
twelve times lighter.

**The latency is the tiering story.** The 176B answers at 2-5 minutes an
item (the expert weights stream from storage: 43-55 s to read the prompt,
then 3 tok/s of generation), against the 35B's 5.4 s.

---

## 2C / 5A. The encoder, on the same replay

At the **shipped** thresholds, the tuned encoder against the stock:

| encoder | fact recall | the dedup's recall at 0.92 | the dedup's precision |
|---|---|---|---|
| the stock L3-v2 | 50.55% | 0.046 | 1.00 |
| the tuned v6 | **50.93%** | **0.292** | 1.00 |

Six times more true duplicates merged, the same precision, the facts
slightly better. The model is the lever; the threshold is not.

**The held-out test** (the conversations split in half, trained on one
half, measured on the other - `experiments/encoder_holdout.py`):

| | the stock L3-v2 | the tuned one |
|---|---|---|
| the dedup's best F1 on the holdout | 0.698 | **0.827** |
| the dedup's recall at the shipped 0.92 | 0.047 | 0.047 |
| the drift's AUC on the holdout | 0.957 | 0.956 |

**The tuning generalises** (18% better separation on unseen
conversations) **and it still cannot be cashed in**: at the shipped
threshold the two encoders are a wash, and moving the tuned one to its own
best threshold costs facts (the replay: 46.3% against the stock's 50.6%).

**The negative result, stated plainly:** as it stands, the tuning buys a
better ranking and nothing else, because the only way to use a better
ranking is to move the threshold - and moving the threshold is what costs
the facts. The next objective is therefore a **hinge at the operating
point** (the duplicates' cosine above 0.92, the distinct pairs' below),
not an F1.

### 2C-bis. The hinge at the operating point: it moves the point, and it breaks the other decision

> **Correction (2C-septies): the drift figures in this section are invalid.** The corpus was 79% ingested documents, so the drift cases were short queries against an attachment; the dedup figures are affected by the same corpus. The replay consequences stand.

The hinge was the log's own next step, so it was run
(`--hinge-margin 0.92`, on the same held-out split, 12 epochs at
lr 5e-5 - the contrastive arm's 42 steps were far too few to say
anything). The objective replaces the dedup task's contrastive loss with
the decision itself: the duplicates' cosine above 0.92, the distinct
pairs' below it, the two errors weighted alike because the decision
weights them alike.

| on the held-out half | the stock L3-v2 | the contrastive tune | the hinge, dedup only | the hinge, dedup + drift |
|---|---|---|---|---|
| the dedup's recall at 0.92 | 0.047 | 0.047 | **0.986** | **0.916** |
| the dedup's precision at 0.92 | 1.000 | 1.000 | 0.722 | 0.833 |
| the dedup's best F1 | 0.698 | 0.824 | 0.930 | 0.908 |
| **the drift's AUC** | **0.957** | 0.956 | **0.731** | **0.420** |
| the held-out gate | - | passes | **fails** | **fails** |

**The premise is confirmed.** The contrastive objective could not move a
fixed threshold - the recall at 0.92 was 0.047 either way. The hinge
moves it to **0.986**: the decision at the operating point is
*reachable*, and the only way to reach it is to optimise the decision.
That is the campaign's claim, now measured rather than argued.

**And the candidate still fails the gate, on the drift task.** The
dedup hinge moves the whole distribution - the positives' median cosine
goes 0.678 (stock) to 0.826, and the negatives' with it - so the drift
pairs' cosines move under the change too, and the drift AUC falls from
0.957 to 0.731.

**And training the drift task alongside does not fix it - it makes it
worse** (0.420, below chance). The drift task's own contrastive loss
and the dedup hinge want opposite things from one encoder: the hinge
wants every positive high, the drift signal wants the *topical*
distance between two chunks to be sharp, and a shared representation
that satisfies the first does not give the second. The two membrane
decisions are not two readouts of one geometry.

**So the shipped defaults stay where they are** - the candidate loses a
job, and the gate says a candidate that wins one job and loses another
is not a release. The finding is the negative one, and it is a real
one: **the membrane's two decisions cannot both be served by the same
encoder at the shipped thresholds**, and the fix is two encoders (or a
different one), not a better training of this one.

*What the next experiment would be:* the dedup decision and the drift
decision are different questions, so they should not share a model. The
hivebench's Global tab already switches the encoder per conversation
(a per-choice drone cache), so a *drift-specific* encoder is a
configuration change, not a new mechanism - and the drift decision is
the cheaper one to move, because its threshold is free (hypothesis H
falsified it: the drift threshold buys nothing and the fact recall pays).

### 2C-ter. Our own Qwen-family bi-encoder

> **Correction (2C-septies): the drift figures in this section are invalid.** The corpus was 79% ingested documents, so the drift cases were short queries against an attachment; the dedup figures are affected by the same corpus. The replay consequences stand.

The hinge above was run on the stock MiniLM, and the campaign's open item
was the *Qwen family*: "make our own qwen-family bi-encoder, we'll need
this regardless of whichever direction we make."

**What it is.** A Qwen3 decoder (`Qwen/Qwen3-0.6B`) with the
Qwen3-Embedding pooling (the last real token, unit length), trained on
the harness's own conversations with two terms: the **hard labels as a
hinge at the shipped operating point** (the duplicates' cosine above
0.92, the distinct pairs' below it) and the **Qwen teacher's relation
structure** as the soft target (`Qwen/Qwen3-Embedding-0.6B`'s cosines on
the same pairs - the student reproduces the teacher's *judgements*, so it
inherits the generator-family semantics rather than the teacher's own
quirks). `experiments/qwen_encoder.py`; the gate is the held-out split,
and the depth is a free parameter the measurements choose.

**The held-out numbers** (the train half / the holdout half of the
conversations; the dedup and the drift jobs, the shipped thresholds):

| the encoder | the dedup f1 at 0.92 | the precision | the recall | the best F1 | **the drift's AUC** | the cost a chunk |
|---|---|---|---|---|---|---|
| the stock MiniLM-L3-v2 | 0.090 | 1.000 | 0.047 | 0.713 | **0.957** | **2.7 ms** |
| the Qwen teacher (0.6B) | **0.090** | 1.000 | 0.047 | 0.855 | 0.952* | 19.6 ms |
| **our 4-layer student** | **0.842** | 0.903 | **0.789** | 0.842 | 0.457 | 29.4 ms |
| our 8-layer student | 0.700 | 0.952 | 0.554 | 0.809 | 0.226 | 59.3 ms |
| our 28-layer student | 0.620 | 0.833 | 0.493 | **0.879** | 0.444 | 206.1 ms |

\* the teacher's drift AUC is the decision-head study's own number on its
frozen J-Bench split (0.9524), not a re-measurement on this holdout: the
teacher's own arm over these 3,575 cases is ~35 minutes on the CPU (the
windows average 7,103 characters), and a 600-case subsample run was
killed at its 25-minute wall before it wrote a number. The cell is not
load-bearing - the teacher is not a shipping candidate.

**The teacher and the shipped encoder are *identical* at the operating
point** - both 0.0897, both a recall of 0.047 - and they differ only in
the ranking (the best F1 0.855 against 0.713). **So the family-match
buys a better ranker and nothing else**, which is the campaign's
central claim about the threshold, now measured on the teacher itself
and not just argued.

**And our student beats its own teacher at the teacher's own task, by
9.4x** (0.842 against 0.090 at the shipped threshold). The teacher was
never trained for the decision; the student was. The 28-layer student
even beats the teacher's *ranking* (a best F1 of 0.879 against 0.855).
**The teacher's quality does not transfer as a decision - the training
objective does.** That is the sharpest form of the campaign's "the model
is the lever, not the threshold" finding: you can beat the teacher at
the teacher's task by training for the decision, and the family-match is
what gives you a teacher worth beating.

**And the depth is a free parameter the measurements chose.** The
4-layer student beats the 28-layer one *at the operating point* (0.842
against 0.620) at **one seventh the cost**. The shallow model is the
better encoder here, and it is also the cheap one.


**But the drift decision is destroyed, and it is not a length artefact.**
Every student's drift AUC collapses (0.957 to 0.457, 0.226, 0.444). The
obvious excuse is that the drift cases are long windows (a mean of
**7,103 characters**) and the student was trained on single chunks. The
measurement says otherwise: on the *same* cases truncated to 200
characters, the 4-layer student's AUC is **0.198** - worse, not better
(the stock's is 0.964 on the same truncation, so it was never
length-sensitive). The student's geometry has **no topical signal at
all**: the hinge taught it "these specific pairs are duplicates" and the
pairwise distillation never constrained the window geometry. The two
membrane decisions want opposite things from one encoder, now confirmed
three independent ways (the MiniLM hinge 0.731, the MiniLM hinge with
the drift task 0.420, and the Qwen students 0.226-0.457).

**And the cost is 11x the drone's budget.** The 4-layer student is
**29.4 ms a chunk** against the stock's 2.7 ms, so at the store size the
stability run measured (1,612 chunks) the curation's cost goes from
157 ms a turn to **1.8 s**. The depth is not the wall - the 4-layer
student is 218M parameters and the *embedding table* is 155M of them: the
cost floor is the Qwen vocabulary and the 1024-wide residual stream, not
the layers. **Narrowing the hidden size is the remaining lever**, and it
is a surgery on the residual stream rather than a training run.

**What this settles, and what it does not.** It settles that the
family-match is not a myth: a Qwen-family encoder, trained on our data
with the decision as the objective, is **nine times better at the
dedup decision** than the shipped one. It does not settle the drone:
the shipped thresholds stay (the student loses the drift job), and the
11x cost keeps it a ceiling rather than a citizen. The two decisions
need two encoders - which the Global tab can already do - and the
shipped drone stays the MiniLM until something wins *both* jobs at the
shipped cost.

### 2C-quater. The replay prices the decision: a true duplicate merged is a fact deleted

2C-ter stopped at the *job*: the 4-layer student makes the dedup decision
nine times better at the shipped threshold. The job is not the product.
The product is the curation - the context the next turn is written from -
so every candidate was replayed through the real pipeline (the same
instrument as hypothesis H: the real router, the real assembly, the real
dedup, the real drift detector, the real store) on the fixture corpus,
50 conversations and 1,187 measurable turns:

| the encoder | at the shipped 0.92 / 0.6 | at its own tuned point | at the recalibrated 0.57 / 0.77 |
|---|---|---|---|
| the stock L3-v2 | **60.51% / 374 tok** | 57.31% / 235 tok | 47.18% / 167 tok |
| the hinged L3-v2 | **32.66% / 139 tok** | 13.67% / 51 tok | 13.67% / 51 tok |
| the Qwen teacher | 64.47% / 615 tok | 48.87% / 244 tok | 33.81% / 142 tok |

(the fact recall / the mean assembled context; `experiments/threshold_ab.py`,
the artifacts under `experiments/encoder-suite/results/replay-20261021/`.)

**Two things are now measured, and both are negative.**

*The decision and the curation pull opposite ways.* The hinge makes the
dedup decision 0.047 -> 0.986 at the shipped threshold - and it nearly
halves the facts (60.51% -> 32.66%) while shrinking the context 2.7x.
The reason is structural: a true duplicate merged is not a scoring error,
it is **a fact deleted from the context**. So improving the decision is
improving the deletion. The shipped 0.92 is not a miscalibration; it is
where the fact-preservation constraint puts the curve.

*A better ranker does not cross the frontier, it moves along it.* The
teacher beats the stock on the dedup ranking (the best F1 0.855 against
0.713), and on the replay it **loses at the two comparable operating
points**: at its own best threshold it keeps 48.87% of the facts in 244
tokens where the stock keeps 57.31% in 235 - worse curation, more
context; at the recalibrated point it is 33.81% against 47.18%. Its one
win is at the shipped threshold: 64.47% against 60.51%, for **615 tokens
against 374**. The campaign's answer-level noise is ~3 points, so that
+4 is not distinguishable from zero - and it is paid in every turn's
prompt. (The stock at a *looser* threshold than 0.92 - the direction that
buys facts by buying tokens - was not measured; the frontier above 0.92
is open, though it is not the direction the shipped point moves in.)

**So the encoder path's dedup decision is closed, and it closes in the
shipped configuration's favour.** No candidate measured improves the
fact-per-token trade-off at a calibrated point: the hinge halves the
facts, the teacher buys 4 points with two thirds more context. The honest
reading is not "the threshold was calibrated for the wrong encoder"; it
is that the fact-preservation constraint dominates *both* dials, the
threshold and the model. The one decision that remains free is the
**drift** decision: its threshold costs no facts (the drift-only arm
keeps 60.51% exactly), and its encoder can be swapped independently. But
*free* is not *better*: no candidate measured beats the stock's drift
AUC of 0.957 - the teacher ties it (0.952 on its own split), the hinge
falls to 0.731, our student collapses to 0.457 - so the door is open and
currently leads nowhere. What the replay cannot see is the drift
*threshold*'s answer-level price (it is neutral on facts); that needs the
expensive half of the A/B, not another encoder.

One caveat on the numbers: this corpus is `tests/fixtures/generated`, not
the `harness_state` corpus behind 2C's 50.55%; compare *within* the
table, not across the two sections.

### 2C-quinquies. Why the student inverts on the drift decision

> **Correction (2C-septies): the drift figures in this section are invalid.** The corpus was 79% ingested documents, so the drift cases were short queries against an attachment; the dedup figures are affected by the same corpus. The replay consequences stand.

The 4-layer student's drift AUC is 0.457 - below chance - while the stock
L3-v2 is 0.957. Capacity is not the story: the 8-layer student is 0.226,
the 28-layer 0.444. The diagnostic separates pooling, truncation and the
cosine scale, on 600 of the 3,575 hold-out cases (27 drift, the class
rate is 4.5%):

| the arm | AUC | within-conversation AUC | cos (drift) | cos (same) | signal p50 | corr(signal, length) |
|---|---|---|---|---|---|---|
| the stock (mean, 256) | 0.934 | 0.919 | 0.351 | 0.573 | 0.429 | -0.414 |
| the student (last, 192) | 0.421 | 0.338 | 0.425 | 0.406 | 0.605 | +0.036 |
| the student (mean, 192) | 0.360 | 0.645 | 0.930 | 0.926 | 0.069 | -0.121 |
| the student (last, 512) | 0.408 | 0.623 | 0.411 | 0.392 | 0.626 | +0.025 |
| the student (mean, 512) | 0.575 | **0.853** | 0.952 | 0.961 | 0.034 | -0.484 |

**Last-token pooling on a truncated document is the primary cause.** At
the same 192 tokens, mean pooling lifts the within-conversation AUC
0.338 -> 0.645. The student's last token is at position 192 of a
7,000-character window - mid-sentence - and its state is a poor summary
of the window.

**Truncation is the second.** Mean pooling at 512 tokens lifts it
0.645 -> 0.853. The student was trained at 192; the drift windows are
~1,800 tokens, so it reads a tenth of the evidence.

**The cosine scale is the third, and it breaks the threshold even for
the stock.** Mean pooling puts every cosine at ~0.95 (signal p50 0.034);
last-token puts them at ~0.41 (p50 0.605). The shipped 0.6 is a point on
neither scale, so the same constant fires on half the turns for one
model and almost never for the other. **A threshold is a property of an
encoder's signal distribution, not of the decision.**

**And the metric has a length shortcut that the stock uses.** The stock's
signal correlates -0.414 with the recent window's length: the shorter the
window, the lower its Jaccard against the long older half, and the more
likely the lexical label says drift. That is a property of the label
construction, and the stock exploits it. The student does not (-0.04 to
+0.04), which is part of why its pooled AUC is lower than its
within-conversation ordering.

**What this settles.** The student is not "bad at drift"; it is a model
trained to summarise a short chunk, asked to summarise a long window with
the wrong pooling. Mean pooling at 512 recovers 0.853 within
conversations - still behind the stock's 0.919, at 11x the cost. The
obvious shippable path was a drift-specific encoder at the stock's cost;
2C-sexies trains it, and the stock beats it (0.957 against 0.277).

`experiments/drift_diagnose.py` (GPU, `--device cuda`); the artifact is
`experiments/encoder-suite/results/drift-diagnose.json`.

### 2C-sexies. The drift encoder: we trained it, and the stock is still the ceiling

The drift decision's threshold is free (H), so a model that ranks better
should be shippable. The candidate was the shipped L3-v2 (same cost)
fine-tuned *only* on the drift triplets from the train half - the
topic-close older window as the positive, the topic-far one as the
negative - 25 epochs, 1,289 triplets.

The hold-out (3,575 cases, 172 drift, 4.8%):

| the encoder | AUC | F1 at the shipped 0.6 | F1 at the train-calibrated threshold | oracle F1 | fires at 0.6 |
|---|---|---|---|---|---|
| the stock L3-v2 | **0.957** | 0.205 | 0.376 (t 0.824) | **0.629** (t 0.734) | 41.1% |
| the drift-fine-tuned | **0.277** | 0.081 | 0.049 (t 0.998) | 0.093 | 92.4% |

The fine-tune does not merely fail to improve it: it **destroys the
topical geometry** (0.277, below even the student's 0.457). Its signal's
median jumps 0.581 -> 0.851: the contrastive objective spread the whole
space apart, and the ordering went with it.

**This is the third fine-tune to lose to the stock on this job** - the
multi-task variant 0.420, the dedup hinge 0.731, the drift-only 0.277 -
and it settles the question the two-encoder angle opened. Our pair sets
are too small and too narrow to improve on a pre-trained topical space;
the stock's 0.957 is not a lucky number, it is the ceiling of what our
data can teach. The two-encoder design is still *sound* (a second
instance cannot hurt the dedup), but there is no better encoder to put
in it.

**The table also shows where the value actually is: the threshold.** At
the shipped 0.6 the stock keeps 97.7% of the drifts and fires on 41.1%
of all turns - a topic reset on two turns in five. At its own 0.734 it
keeps 73.8%, fires on 6.5%, and the F1 goes 0.205 -> 0.629. The replay
already showed that moving the drift threshold costs no facts (60.51%
either way). So the drift decision's problem was never the encoder, and
it is not the model: it is the constant.

**Two cautions on the calibration.** The train split's prevalence is
2.0% (58/2,958) against the hold-out's 4.8% (172/3,575), so a threshold
fitted on train is too strict for the hold-out (F1 0.376 against the
oracle's 0.629): a threshold must be re-derived for the deployment's
prevalence, not carried across a split. And 0.734 is an oracle; the
deployable number is the F1 at a prior-corrected threshold on a
held-out period, which has not been measured.

`experiments/drift_encoder.py`; the artifact is
`experiments/encoder-suite/results/drift-encoder/drift-encoder.json`.

### 2C-septies. **Correction:** the drift corpus was measuring the attachment

**The drift numbers in 2C through 2C-sexies are invalid.** This section
says why, and what replaces them. Two corpus bugs, both in
`load_conversations`:

1. **The ingested attachments were loaded as conversations.** 17 of the
   82 `conv-*.json` files are a knowledge-base attachment chunked into
   `[block n/328]` pieces - 5,531 of the 6,977 stored chunks. The drift
   builder compares a "recent window" against the first half of the
   file, so its cases are short user *queries* against a 5 MB
   *document*. The label (Jaccard < 0.05) then says "drift" whenever the
   recent window is a query: **45.7% of the shortest length decile, 0%
   of the seven longest.** The stock's 0.957 was detecting that; the
   head's 0.994 was learning the window length; the student's 0.457 was
   failing to reproduce the artifact.
2. **The same conversation id was loaded twice.** Both state
   directories hold `conv-*.json` with the same ids (a 29-chunk copy and
   a 244-chunk copy) and the loader did not dedupe - so one conversation
   could sit in both the train and the hold-out half.

After the fix (the longer copy wins; files that are mostly `[block
n/m]` are dropped), the corpus is **64 conversations and 1,417
chunks** - and it cannot support a drift measurement: the split gives
**2 drift cases in train** and **176 in the hold-out, 144 of them from
one conversation**. The clean pooled AUC is 0.109 (below chance) and
the head's cross-validation is undefined: there is nothing to train on.

**So the drift decision is unmeasured.** Not 0.957, not 0.457, not
0.277, not 0.994. The dedup column of the same tables is affected too -
the attachment's repeated headers are template duplicates - though the
*replay* (the consequence, on `tests/fixtures/generated`) is a separate
corpus and stands. The cascade results (1B through 1B-sexies) are
unaffected: they never used this corpus.

**What replaces it.** A drift benchmark with (a) **size-matched
windows**, so Jaccard is not a length proxy, (b) a
**conversation-disjoint** split, and (c) enough conversations that one
cannot dominate. J-Bench is the closest labelled artifact on disk (the
decision-head study scored 0.9524 on it); a purpose-built set from the
fixtures is the alternative. The `drift_head.py` instrument (a head on
frozen stock pair embeddings) is ready for whichever lands first.

---

## 3A-3E. The cascade's own arms, measured on the 35B

The same 28 organic tasks, the 35B, T=0.7, **three draws each** (84 items
per arm - the multi-sample regime, because a single draw is a direction and
not a measurement). Only the arm changes.

| arm | served-correct | the misses | the verifier rejected | of which false | the repair leg | the gate | the mean |
|---|---|---|---|---|---|---|---|
| **no verifier, no repair** | 64/84 (76.2%) | 20 | - | - | - | - | 7.6 s |
| **the deterministic verifier, no repair** | **63/84 (75.0%)** | 21 | 6 | 1 | - | - | 6.5 s |
| **the verifier + the repair leg** | **67/84 (79.8%)** | 17 | 9 | 1 | **8/9 cleared** | - | 6.6 s |
| **+ the learned gate** | **70/84 (83.3%)** | 14 | 12 | 3 | 8/11 | fired 7 (4 true, **3 false**) | 8.1 s |

**The cascade's economics, with effect sizes.**

- **A verifier on its own is a tax: -1.2 points.** It caught 5 wrong
  answers and, with no repair leg, they were simply *not served*. A
  verifier that drops what it catches and fixes nothing makes the system
  worse.
- **The repair leg is what makes verification pay: +4.8 points** over the
  verifier alone, and **+3.6 over no verification at all** (79.8% vs
  76.2%) - 8 of the 9 catches cleared.
- **The learned gate adds +3.5 points** (83.3% vs 79.8%) - and **3 false
  flags against the pre-registered budget of 2, so by its own rule the
  gate stays OFF**.

**The last row is why the regime matters.** The single-draw gate arm
looked like a clean win (+1 served item, 0 false rejects, 2 flags in 56
items). At 84 items the same arm shows **3 false flags** - over the
budget - and the verdict is the one the rule was written for: **off**. A
single draw would have shipped it.

**A correction to the earlier claim.** The "0 false rejects" in the
published 66-item table is the *escalation's* bookkeeping, not the
verifier's: the runner's `false_reject` field only exists once something
escalates, and nothing escalated in these runs. The verifier's own false
rejects - a rejected answer that was right - are **1, 1 and 3** across
the three arms, computed from the verdict and the local answer's
correctness. They are not zero, and the trade is still worth making: the
verifier catches 8 wrong answers for every 1 correct answer it drops.

### 3C. The learned gate, on the 35B

The gate's verdict is in the table above (the 84-item regime): **+3.5
points, 4 true catches, 3 false flags - over the pre-registered budget of
2, so it stays off by its own rule**, exactly as the 176B's measurement
did (3 false flags, 2026-10-18). The single-draw arm that appeared to
flip the verdict was the sampling noise, and the multi-sample regime put
it back.

What is consistent across both measurements: the gate **catches real
silent wrongs** (4 here, 1 there) and its false flags are the thing the
budget exists to bound. The mechanism works; the head does not separate
well enough on this distribution to pay for itself.

### 2A. The curation against the FIFO window (the memory's headline claim)

The paired A/B: the same model, the same turns, two arms - the context the
curation assembles, and the last-4k-token window of the same history. The
35B as the generator, 25 conversations x 20 turns (**180 measurable turns**).

| | the curated arm | the FIFO arm |
|---|---|---|
| the answer's fact recall | 81.1% | 81.7% |
| the average fact hit ratio | 0.793 | 0.810 |
| the context fidelity (the answer's terms sourced from its own context) | **0.241** | 0.208 |
| the turns where the arm's fidelity is higher | **61.7%** (the pre-registered bar: 63.9%) | 38.3% |
| the turns where the answer is at least as good | **88.4%** (the bar: 80%) | - |
| the strictly-better turns | 13 | 17 |
| the both / neither buckets | 116 / 34 | |
| the context sufficiency (P3-style) | **97.5%** >= FIFO | - |

**Two metrics, two verdicts.** The answer-level "the curated answer is at
least as good" passes (88.4% against the 80% bar) and the context-level
sufficiency passes (97.5%); the *fidelity* metric misses its bar (61.7%
against 63.9%). On the answers the two arms are a **tie** (81.1% against
81.7%), and the curated arm never wins a turn strictly more often than the
raw window (13 against 17).

**And this is the published P3 result, on a different model, telling a
different story.** The published measurement (the 27B) had the curated arm
winning **115 turns outright to 26** - a 4.4:1 margin - on 175
fact-retrievable turns. Here, on the 35B-A3B, it is **13 to 17**. The
"neither" bucket also collapses (145 of 175 published, 34 of 180 here):
the 35B answers from *whatever* context it is given, so the arms tie.

**The scoping result, stated plainly: the better the generator is at long
context, the less the curation has to add.** The FIFO window is the most
recent turns verbatim, and a model that uses them well does not need a
curation to find them. The architecture's value is a function of the
generator - which is the first thing the paper's own Postulate 3 (the
separation) implies, and the first thing the measurements had not tested.

**What this does not say.** It does not say the curation is useless: the
curated context is still sufficient on 97.5% of the turns, the context
fidelity is ahead (0.241 against 0.208), and the *curation* is what keeps
the context bounded - the FIFO arm's window is 4k tokens of recency, and
the arms are equal *at that budget*. It says the curation's *accuracy*
advantage is generator-dependent, and that the honest claim is the bounded
context, not the better answers.

**The paired arm, re-run to the end** (15 conversations x 16 turns, the
same generator, the engines sequential so the read is unconfounded),
because §2A.2's family-match arm needs a complete arm to be paired
against. 66 comparable turns, 11 conversations:

| | the curated arm | the FIFO arm |
|---|---|---|
| the answer's fact recall | 85.15% | 84.67% |
| the context fidelity | 0.803 | 0.803 |
| the turns where the curated arm is at least as good | 84.85% | - |
| the strictly-better turns | 4 | 2 |
| the curated context's mean size | **355 tokens** | 544 tokens |

The same verdict as the 180-turn run: a tie at the answer level, and the
curation's real output is a **35% smaller context with the same facts**.


### 2A.2 The hypothesis this suggests: the encoder and the generator are
### different architectures

**H (architecture match).** The curation's selection happens in the
*drone's* embedding space - a MiniLM, a BERT-family bi-encoder - while
the generator reads the curated chunks as *text* in a Qwen-family causal
model's space. The two spaces are unrelated, so the drone's notion of
"relevant" is not the generator's. If the drone were a bi-encoder from
the *generator's own family* (a Qwen3 decoder trained as a bi-encoder),
the selection would happen in the generator's own semantic space, and the
curation's advantage would return.

**Why this is worth testing.** The sweep in §4 never included a
Qwen-family encoder: every candidate was a BERT-family bi-encoder
(MiniLM, bge, e5, gte) or a Mistral-family one (nomic). The generators
are Qwen-family (the 35B-A3B is a Qwen3.5 MoE, the 176B a Qwen4 exp).
So the one family that could share the generator's space was never
measured - the gap the hypothesis names.

**The test.** The same paired A/B (the same tasks, the same generator,
the same arms) with the curation's encoder swapped for
`Qwen/Qwen3-Embedding-0.6B` - a Qwen3 decoder trained as a bi-encoder,
1024 dims - against the stock MiniLM on the identical sample.

**Both arms are now run to the end, on the same 66 comparable turns**
(15 conversations x 16 turns, the 35B as the generator, the engines
sequential so the read is unconfounded; `experiments/paired_ab_summary`
reads the common prefix).

**The answer level, which carries the sampling noise.** Both arms tie
the FIFO window:

| | the curated arm's recall | the FIFO arm's recall | the curated >= FIFO |
|---|---|---|---|
| the stock MiniLM | 85.15% | 84.67% | 84.85% |
| the Qwen3-Embedding | 81.59% | 84.87% | 77.27% |

The two arms are separate runs with separate draws, so their answers are
not comparable to each other - and the 3.3-point gap the table shows is
that noise, not a result.

**The selection level, which does not.** The store replays the fixture
answers (the fair-selection default), so the *curated context* is a
deterministic function of the conversation and the encoder: the only
sampled thing in the run is the reply. The measure is therefore exact,
and it is the mechanism the hypothesis is about - did the curation's
*selection* put the facts in the context?

| on the same 66 turns | the curated context carried the facts | the FIFO window | the curated only | the FIFO only |
|---|---|---|---|---|
| the stock MiniLM | 59/66 (89.4%) | 62/66 (93.9%) | **0** | **3** |
| the Qwen3-Embedding | **62/66 (93.9%)** | 62/66 (93.9%) | **0** | **0** |

**The mechanism is confirmed, and it is a real effect.** The
Qwen-family encoder's curation carries the facts on **every** turn the
FIFO window does - **0 discordant pairs in either direction**. The
stock MiniLM's curation **loses 3 turns** to the raw window and never
wins one. So the hypothesis's mechanism is right: a bi-encoder from the
generator's own family selects better than a BERT-family one, and the
whole of the stock encoder's deficit is those 3 turns.

**But it does not change the deployment, for two measured reasons.**

1. **The answer level is unchanged** - the 35B answers from whatever
   context it is given (§2A's scoping result), so a better selection
   has no answer-level effect to amplify. The curated context is 401
   tokens against the FIFO window's 544, so the Qwen arm does deliver
   the same facts in 26% fewer tokens; that is the bounded context, the
   architecture's actual claim, and it is now delivered by the
   generator's own family.
2. **The cost is 22x.** The curation scores *every stored chunk* on
   *every turn*, so the encoder's per-chunk cost is the whole budget:
   **0.9 ms a chunk** for the stock MiniLM against **19.6 ms** for the
   Qwen3-Embedding (measured, 64 chunks, warm). At the store sizes the
   stability run measured (1,612 chunks) that is 1.5 s a turn against
   31 s a turn - the drone's cost goes from the measured 157 ms a turn
   to over a minute, and the curation's cost is linear in the store.

**So the hypothesis is confirmed at the mechanism and rejected as a
deployment.** The family-match encoder is the *right* encoder for the
selection - it is the only candidate measured that never loses a turn -
and it is 22x too expensive to be the drone. That is Option A's own
verdict in `../splinter/docs/ENCODER-PLAN.md` ("a quality ceiling, not a shippable
drone"), now measured rather than argued, and it points at the same
place the plan already pointed: **the Qwen encoder is a teacher, and the
drone must be a student of it.**

---

### 3D. The router (the local Tiny-Jev)

The router's "api" route and the escalation tier both need the frontier
API, and **the configured key has expired** (HTTP 401,
`api_key_expired`), so both were re-pointed at the local 176B - a
legitimate "large tier" stand-in.

| | |
|---|---|
| **the router's agreement with the large model** | **0.70** |
| the served-correct (the measure mode does not change the answers) | 0.800 (8/10) |
| the false rejects | 0 |
| the mean per-item latency | 37.2 s |

The local 1.7B router agrees with the 176B on **70% of the items** - the
cascade's "the router is a floor" claim, measured. (The 37.2 s is the
*measurement's* cost: the measure mode fetches the large model's
comparison answer for every item, which is not a serving cost.)

### 3E. The escalation tier

With the repair leg on, the escalation **never fired** (0.0): the two
wrong answers the verifier caught were both cleared by the repair leg
(2/2), so nothing was left for the expensive tier.

That is the ladder working as designed - the cheap leg comes before the
expensive one - but it means the arm measures the ladder, not the
escalation. The arm that measures the escalation is the one with the
repair leg **off**, so the verifier's catches fall through to the large
tier.

**That arm came out inconclusive, and why it matters.** With the repair
leg off, the escalation still did not fire - because in *that draw* the
deterministic verifier caught **nothing** (0 rejections, 21/28 served
against the other repair-off draw's 18/28).

**The verifier's catch rate is wildly draw-dependent: 0 to 4 catches out
of 28, on the same tasks and the same model.** Every single-draw arm is
therefore noisy, and the published 66-item number - which is the *3-sample*
regime - is the stable one. The arms in this section are useful as
*directions* (the repair leg is worth +6, the gate is worth +1 on this
set) and should not be read as precise effect sizes until they are re-run
in the multi-sample regime.

**The escalation tier's own value is therefore still unmeasured here**:
with the repair leg on it never fires (the cheap leg gets there first),
and with it off the verifier's catch rate is too noisy to put a number
on. What is established: the ladder is ordered correctly, and the
escalation is a **backstop**, not a routine path.

---

## The encoder sweep (all the candidates)

| encoder | params | dim | dedup F1 @ 0.92 | dedup best | drift F1 @ 0.6 | drift best | drift AUC | ms/chunk |
|---|---|---|---|---|---|---|---|---|
| **L3-v2** (the shipped one) | 17.4M | 384 | 0.089 | 0.761 @ 0.57 | 0.151 | 0.554 @ 0.77 | 0.945 | 1.3 |
| L6-v2 | 22.7M | 384 | 0.072 | **0.827** @ 0.60 | 0.137 | 0.545 | 0.807 | 1.6 |
| bge-small-en-v1.5 | 33.4M | 384 | 0.085 | 0.662 @ 0.76 | 0.000 | 0.489 | 0.725 | 1.9 |
| e5-small-v2 | 33.4M | 384 | 0.536 | 0.692 @ 0.91 | 0.000 | 0.501 | 0.882 | 1.7 |
| gte-small | 33.4M | 384 | 0.574 | 0.821 @ 0.90 | 0.000 | 0.500 | 0.877 | 2.0 |
| nomic-embed-v1.5 | 136.7M | 768 | 0.036 | 0.753 @ 0.77 | 0.000 | 0.600 @ 0.38 | **0.9815** | 18.4 |

nomic was measured on a subset (1,500 pairs, 2,000 cases - a 137M model on
the CPU is eighteen times the cost per chunk). It is not a drone
candidate; it is the **teacher** candidate: the best topic-change signal
measured, AUC 0.9815 against the drone's 0.945.

---

---

## 4A. The whole protocol, on the 35B

The P1-P10 protocol, live, against the 35B-A3B (a 12-conversation fixture
subset - the protocol's own runner, `experiments.run_p1_p10 --live`).

| | verdict | the evidence |
|---|---|---|
| P1 constant throughput | **PASS** | the throughput does not decay over the run |
| P2 retrieval precision | **FAIL** | the precision clause (the known encoder ceiling) |
| P3 context sufficiency | **PASS** | the curated context is sufficient against an equal-size FIFO window |
| P4 domain-dependent decay | **PASS** | code m90 1.8 (91.0% at the best) vs prose m90 1.2 (15.3%), a gap of 0.6 - the domains separate beyond the 0.2 band |
| P5 targeted masking | SKIP | a training experiment |
| P6 confidence escalation | **FAIL** | the escalated recall is *worse* (0.174 against 0.565) |
| P7 auditor-human agreement | SKIP | it needs human raters (90.25% measured 2026-08-23) |
| P8 routing accuracy | **PASS** | 200/200 |
| P9 densest-duplicate retention | **PASS** | the densest chunk wins 100% of the turns against recency's 0% |
| P10 drift reset | **FAIL** | the reset changes the recall by 0.0 points (the detector fires 571/628 when forced) |
| P11 comb resurrection | **PASS** | the pressure regime: 100% topic-return recall against 20% without the comb |

**PASS 6/11, 2 skipped - the published verdict set, exactly.** The
published run (2026-08-23/24, on a 27B, on the full corpus) reported
P1/P3/P4/P8/P9/P11 PASS, P2/P6/P10 FAIL, P5/P7 SKIP. The same set
reproduces here on a *different model* and a *smaller corpus*, which says
the protocol is measuring the architecture and not the generator.

**P10 is the one worth a second look**, and the campaign has an answer
for it: the drift detector's threshold is calibrated for a different
encoder (the measurement in §7), so "the reset has no effect" may be
measuring a detector that fires on almost everything (571/628) rather
than a reset that does not work. The P10 verdict stands as measured; the
mechanism is now in question, and that is a different claim.

---

---

## 4B. The 500-turn stability - and a correction to the latency claim

The memory pipeline, 500+ turns of a *growing* conversation (every turn
states a new fact; every fifth asks about an old one), the real drone, no
model. `experiments/stability_500.py`.

| turns | the store | the assembled context | the curation's cost (p50) | early | late | the peak RSS |
|---|---|---|---|---|---|---|
| 250 | 212 chunks | 322 tok | 22.8 ms | 10.8 | 35.0 | 519 MB |
| 500 | 412 chunks | 320 tok | 37.3 ms | 14.5 | 81.3 | 564 MB |
| 1,000 | 812 chunks | 335 tok | 89.7 ms | 21.2 | 143.5 | 594 MB |
| 2,000 | 1,612 chunks | 332 tok | 156.8 ms | 31.9 | 263.5 | 667 MB |

**The context stays bounded and the cost does not.** The assembled
context is flat across the whole range (322 → 332 tokens ✓ - the
curation's promise ✓✓), the store grows ✓, and nothing is OOM-killed ✓
over 2,000 turns ✓.

**But the curation's per-turn cost is linear in the store's size**: 22.8 ms
at 212 chunks, 156.8 ms at 1,612 - about **7x for 7.6x the chunks** ✓✓.
The published "≈18 ms added per turn" is the *small-store* number ✓; the
honest statement is that the curation's cost grows with the conversation
it is curating ✓ (every chunk is scored, every turn ✓).

**So the memory's trade is explicit:** the *prompt* stays bounded ✓ -
which is the expensive part for the model ✗✗ - while the *curation* gets
more expensive as the conversation grows ✗. On a long conversation the
two costs trade against each other, and the crossover (where the
curation's ms exceed the tokens it saves) is a measurement the system
does not yet make ✗.

---

## What ran, and what is still owed

| id | what | state |
|---|---|---|
| 1A | the two Scions, head to head | **done** (28 tasks; equivalent, 12x the weight apart) |
| 1B | the cascade over each model | **done** (identical outcomes, 42.9x the latency) |
| 1C | the MTP drafter on/off | **owed** - the Hub sidecar is in the Strata format, not llama.cpp's `-md` |
| 2A | curated vs FIFO | **done** (180 turns; a tie on the 35B - the generator scopes it) |
| 2B | the membrane's thresholds | **done** (H falsified; the asymmetry is the reason) |
| 2C / 5A | the encoder on a held-out split | **done** (the tuning generalises, and still cannot be cashed in) |
| 3A-3E | the cascade's arms | **done** except the escalation's own value (the verifier's catch rate is too draw-dependent to measure it) |
| 4A | the P1-P10 protocol | **done** (PASS 6/11, 2 skipped - the published verdict set, reproduced) |
| 4B | the 500-turn stability | **done** (2,000 turns: the context is flat, nothing OOMs, and the curation's cost is linear in the store) |
| 4C | the PES per configuration | rides along with 4A |

**The rule the campaign learned:** a single draw is a direction, not a
measurement. The verifier's catch rate on the same tasks and the same
model ranges from 0 to 4 out of 28 across draws, so every number in this
log that moved a headline verdict was re-measured with three draws (84
items) before it was believed. The gate's verdict changed when it was.


---

## 1B. The cascade over each model, same tasks

The same 10 organic tasks, the same cascade, the same verifiers, the two
models:

| | the 35B-A3B | the 176B-A6B |
|---|---|---|
| the served-correct | **0.800** | **0.800** |
| the false rejects | **0** | **0** |
| the wrong-accepted | 2 | 2 |
| the repairs | 0 needed | 0 needed |
| **the mean per item** | **5.4 s** | **229.7 s** |

**Identical quality, 42.5x the latency.** The 176B spends 2.3x the
tokens thinking and 42.5x the seconds getting there. On this class the
small tier is not a compromise - it is the same answer, faster and twelve
times lighter.

---

## 1B-bis. Where the 176B's "minutes of work" actually goes

The 229.7 s per item above is real, and it is **not** a property of the
model. The engine's own log says the 176B decodes at **67.7 tok/s** over
the 4,424 generations it has served, with 4,095 of them above 20 tok/s.
The 3 tok/s in the campaign was a *state*, and the state has two names.

**The bimodality, measured on the same engine, the same request:**

| | the request | wall | the rate |
|---|---|---|---|
| the first (cold) | 103 tokens | 29.8 s | 3.5 tok/s |
| the second | 103 tokens | 3.8 s | 27.9 tok/s |
| the third | 103 tokens | 2.1 s | 51.1 tok/s |
| **8 in a row, steady state** | 256 tokens each | **4.0 s each** | **60-73 tok/s** |

The engine's own line for each request separates the two phases:

```
prompt 70 tokens =  0 reused + 70 read in 14029 ms (5.0 tok/s)   <- cold
prompt 70 tokens = 65 reused +  5 read in    71 ms (70.6 tok/s)   <- warm
```

A **fresh prompt read costs 5 tok/s; a reused prefix costs 71 tok/s.**
The 13x is the whole difference. The cold phase is the OS pulling the
model's expert weights off the NVMe: the engine reports *"the OS read
96842 MB from storage"* on one cold request and *"7633 MB"* on the next.
That is the "minutes of work" - not the arithmetic.

**The two levers, and one falsified hypothesis.**

- **Keep the engine warm.** An unload/reload costs the whole cold phase
  again. This is the single largest factor in the campaign's 229.7 s.
- **Do not co-site.** The 35B on the same cards halves the 176B's expert
  cache (8.55 GiB -> 4.11 GiB) and evicts its page cache. The engine's
  own readout: *"decode expert cache hit rate: 96.5%"* alone, *"89.8%"*
  with the second engine resident. Measured end to end, co-siting costs
  **2.6x**.
- **Falsified: `--ple-io mmap`.** The engine's own source flags the
  default as the suspect: *"`--ple-io direct` (the default) does
  unbuffered random reads of the n-gram table; on some drives they run at
  34 MB/s or 4-8 tok/s and a prompt reads layer 1 for minutes"*
  (`strata/src/program/generate.cpp:1775`). The symptom matched - our
  cold prompt read was 3.5-5.0 tok/s, right in the documented band. But
  switching the PLE table to `mmap` did **not** help: the cold prompt
  read was 21.3 s under `mmap` against 17.2 s under `direct`, and the
  dominant I/O is the **96.9 GB of expert weights**, not the 2.1 GB
  n-gram table. The source's warning was about a *different* drive's
  unbuffered I/O; on this box the PLE table is not the bottleneck. The
  hypothesis is dead and the engine keeps `direct`.

**Why it cannot be fully cached.** The 176B's Q6_K file is 60.2 GB; the
box has 30 GB of RAM, of which ~13 GB is available for the page cache.
The kernel's own answer (`mincore`, through
`harness/models.py::page_cache_share`): the 176B's file is **20.8%
cached**, the 35B's is **0%**. A cold request must therefore stream from
the NVMe, and no scheduling trick changes that - only a smaller model, a
bigger box, or not asking the 176B in the first place.

**Falsified: a whole-file page-cache prefetch.** The obvious fix - ask
the kernel to read the 60 GB into the page cache before the engine
starts - does not work, and the reason is arithmetic rather than
subtle. `posix_fadvise(WILLNEED)` over the whole file, then polling
`mincore` for 80 s: the file stayed at **0.0% cached** the entire time.
A 13 GB page cache cannot hold a 60 GB file, and the kernel's readahead
has nothing to keep. The controlled A/B agrees: with the prefetch the
first request was 36.9 s, without it 33.6 s - no difference, both at
~7 tok/s. The launch path's prefetch (for the 35B, which *does* fit) is
untouched; the Strata path never gets one.

**The lever that does work: the engine's resident tier.** Strata can
keep the experts its VRAM cache cannot hold **in ordinary RAM**
(`--resident-budget-gib N`), so they are never re-read from the file.
With an 8 GiB budget the engine's own tier readout changes character:

| | the stock engine | the resident tier |
|---|---|---|
| the blobs served from RAM | 0 | **19,178** |
| the blobs served from the file | 3,246 | 3,164 (and falling) |
| the storage read, the first request | 67,301 MB | 71,026 MB (it has to fill once) |
| the storage read, the next | 1,415 MB | **204 MB** |
| the storage read, the third | 462 MB | **93 MB** |

The engine's own line: *"resident RAM: 8.00 GiB of experts in RAM,
2112 exchanged with the VRAM tier"*.

**The controlled A/B**, both arms from the same cold state (the file's
pages evicted with `fadvise(DONTNEED)`, then the engine restarted, the
same 256-token probe - `experiments/coldstart_prefetch.py`):

| | the first (cold) request | the warm rate |
|---|---|---|
| the stock engine | 33.8 s (7.6 tok/s) | 63-74 tok/s |
| **the resident tier, 8 GiB** | **17.3 s (14.8 tok/s)** | 74 tok/s |

**1.9x on the cold request, and the warm rate is unchanged.** The
resident tier cannot be free: it holds 8 GiB of the box's 30 GB, and it
takes a first request to fill (the engine's 71 GB first read is the
arena filling itself). What it buys is that every request *after* that
one reads 100-200 MB from storage instead of 1.4 GB.

**And it is now a control, not a command line.** The Studio's Global tab
carries it as the *resident experts* field (the runtime block, the launch
baseline), and the launch panel passes it to the engine as
`--resident-budget-gib`. The same launch, measured both ways on this
box, the 176B alone:

| through the Studio's own launch | the resident budget off | the resident budget 8 GiB |
|---|---|---|
| the first (cold) request | 14.2 s (18.0 tok/s) | **7.5 s (34.1 tok/s)** |
| the warm rate | 84.7-94.3 tok/s | **93.3-101.9 tok/s** |
| the storage read, the next request | 54-261 MB (11-54 faults) | **0.6-2.4 MB (0 faults)** |

The engine's own line with the budget on: *"resident RAM mode: 8.00 GiB
of experts in RAM (page-locked), 11707 in the GPU cache; adaptive swaps
exchange them with the GPU cache (no file reads)"* and *"layer split:
90% of the experts resident"*. With it, the file tier is **never
touched** - 0 major faults a request.

(The cold numbers here are "the engine restarted, the page cache still
warm"; the *true* cold - the file's pages evicted with
`fadvise(DONTNEED)` - is the 33.8 s against 17.3 s above. The ratio is
the same in both, which is the point.)


This is the DeepInfer idea, already implemented in our engine:
DeepInfer's **EPLB** (Expert Parallel Load Balancer) re-places experts
by their measured routing counts, and Strata's **adaptive tier** does
the same for its own cache - `--expert-profile-save` writes out what
the engine learned, and `--expert-profile` reads it back at the next
start. The profile the engine already loads (ranked by routing
frequency, 8,538 pairs pre-filled at 971 MB/s) is that ranking.


---

## 1B-ter. The 176B's own head: a verdict costs 2.4 s, a generation 27 s

The escalation tier asks the 176B to *answer* again. But the 176B has
a **decision head** - and reading it needs only the model's *last
prompt token's hidden state*, which is a **prefill**, not a generation.
The head is in our own published dataset
(`SkyIsNotGreen/decision-heads`, `scion-176b-a6b-decision-head.json`),
and it was fitted on this engine, on this body, through the engine's
`emb=` readout.

**The head, re-verified here.** The stored states the head was fitted
on (`probe-states-main176b.npz`, 378 items) reproduce its published
numbers exactly:

| | the head's own | recomputed from the stored states |
|---|---|---|
| the AUC (the 116-item test split) | 0.9941 | **0.9941** |
| the mean p on a correct answer | - | 0.9945 |
| the mean p on a wrong answer | - | 0.1370 |

**The live engine agrees with the stored states.** Running the real
`POST /v1/verdict` on the frozen judge bench's own items, the live
p_correct tracks the stored one to within **0.027** (0.9969 vs 0.9977 on
a correct answer; 0.1517 vs 0.1760 on a wrong one), and the separation
is total: every correct answer above 0.99, every wrong one below 0.16.

**What it costs.** A verdict is a judge prompt (170-184 tokens) and
nothing else:

| | the time |
|---|---|
| **a verdict (the prefill)** | **2.4 s** |
| a generation (the warm engine) | 27 s |
| a generation (the campaign's co-sited engine) | 230 s |

**What it buys, on the campaign's own 28 tasks and the 35B's own
answers to them** (`experiments/verdict_tier.py`):

| | |
|---|---|
| the head's AUC on the 35B's answers | **0.871** |
| the mean p where the 35B was right | 0.988 |
| the mean p where the 35B was wrong | 0.750 |
| the errors caught at p < 0.5 | **2 of 7** |
| the correct answers kept at p >= 0.5 | **21 of 21** |
| the mean verdict | **2.69 s** against the 35B's 6.68 s |

So the 176B's head is a *cheap* judge: it never rejects a correct
answer, it catches about a third of the errors, and it costs 2.7 s
where a generation costs 27 s and the campaign's engine cost 230 s.

**And the thing it is competing with is worth less than it looks.**
Comparing the two models on the same 28 tasks, the 176B's *generation*
fixed 6 of the 35B's errors and **broke 8** - net zero, at 40x the
latency. (Re-measured later, warm and alone, on the same tasks: it
fixes 2 and breaks 2 - still net zero.) A 2.4 s verdict that catches a
third of the errors and breaks none of the correct answers is a better
use of the 176B than a 27 s generation that breaks more than it fixes.

**The honest limit.** The head's AUC on the 35B's answers (0.871) is
well below its AUC on its own model's answers (0.994) - it is reading a
*different* model's output. It is a **filter, not a replacement**: a
gate that lets a correct answer through untouched and flags a third of
the wrong ones, so the expensive generation is spent on a third of the
cases instead of all of them. That is a 3x cut in the escalation tier's
cost, not a 10x. The dataset's own community head reaches 0.852
per-sample (0.965 within-pool) and says the same thing: *"the in-domain
refit beats the frozen J-Bench head ... validate per model"*.

---

## 1B-quater. The escalation tier, measured both ways

The two arms are **alternatives**, so they are measured **separately** -
interleaving them makes each inherit the other's expert cache (a
generation's decode evicts the judge's experts, and the verdict after it
costs 23 s instead of 4.6 s). `experiments/escalation_arms.py`, the
same 28 organic tasks, the same 35B answers, the same 176B:

| | the generate arm | the judge arm |
|---|---|---|
| what the 176B does | answers again | reads the 35B's answer |
| the cost for 28 items | **579.1 s** | **129.8 s** |
| the mean per item | 20.68 s | **4.64 s** |
| the median per item | 22.72 s | **3.82 s** |
| the errors it fixes | 2 | - |
| the correct answers it breaks | 2 | 0 |
| the net | **+0** | - |

**The generate arm is worth nothing here.** Scored with the tasks' own
checkers, the 176B's fresh answer fixes 2 of the 35B's 7 errors and
breaks 2 of the 21 correct ones: **net zero, at 20.68 s an item.** (The
earlier co-sited run said 6 fixes against 8 breaks on the same 28 tasks
- also net zero, and noisier.)

**The judge arm finds 2 of the 7 errors and breaks none of the 21
correct answers**, at 4.64 s an item. Both of its escalations are real
errors (`gsm8k-0033` at p=0.427, `truthfulqa-0015` at p=0.397) and none
of its 21 correct answers went up.

**So the escalation tier's honest options are:**

| | the cost | the net effect |
|---|---|---|
| ask the 176B to answer | 20.68 s an item | **+0** on this class |
| ask the 176B to judge | 4.64 s an item | 2 errors found, 0 broken |
| ask the 35B to answer and stop | 6.68 s an item | the 35B's own accuracy |

**4.5x cheaper, and the only one of the three with a positive expected
value on the errors.** The tier the campaign shipped - a 176B
generation - was paying 20.68 s (and 230 s co-sited) for nothing.

**What this does not claim.** The judge arm is a *filter*: it found 2 of
7 errors, not 7. The remaining 5 are the escalation the tier still has
to pay for. The right shape is the two together - judge first, generate
only for what the judge sends up - which on this set is 2 generations
out of 28.

**And the dataset's own caveat, checked.** The dataset's port report
says the *in-domain* community head beats the frozen J-Bench head on
this body ("their own 'validate per model' caveat, in action"). That was
on the community task set. On the cascade's organic set, with both heads
applied to the *same* live readouts
(`experiments/heads_on_the_cascade.py`), it does not hold:

| | the J-Bench head | the community head |
|---|---|---|
| the AUC on the 35B's answers | **0.864** | 0.782 |
| the mean p on the correct | 0.988 | 0.885 |
| the mean p on the wrong | 0.777 | 0.720 |
| at 0.50: caught / kept | 2/7 and **21/21** | 2/7 and 19/21 |
| at 0.90: caught / kept | 3/7 and 20/21 | **4/7** and 16/21 |

The community head catches one more error at 0.90 and keeps five fewer
correct answers doing it. At 0.50 the frozen head is strictly better:
the same errors caught, four more correct answers kept. So the head to
ship for this class is the **frozen J-Bench one** - the opposite of the
dataset's own recommendation, on a different task set, and the reason
the dataset says to validate per model.

**Two things had to be right for a head to read anything at all**, and
both were wrong at first:

- **The prompt must be the model's own template, with thinking off.** The
  template injects *"Reasoning effort is set to xhigh..."* into the
  system turn and a `<think>` block before the answer; the head was
  fitted on a clean prompt. With the default template the head reads
  **nothing**: p was 0.0238 for every item, correct and wrong alike
  (an AUC of 0.497, a coin). `enable_thinking: false` is what the
  engine's own `verdict()` uses.
- **The vector must be raw.** `/v1/embeddings` L2-normalises by
  default (the OpenAI shape); the head was fitted on the raw float32
  vector. `/embedding` with `embd_normalize: -1` is the raw one.

Both mistakes produced a *constant* p rather than a wrong one, which is
what made them look like "the head does not work here" instead of "the
readout is the wrong vector". The check that catches it: a head with no
signal gives the same p for every item.

**The cost of the corrected readout** is 3.19 s a mean (the same judge
prompt the `/v1/verdict` route builds), so the tier's price stands.


**What the tier was hiding.** The campaign's escalation arms never fired
with the repair leg on (0 escalations) and the 176B's own generation
was net-zero on the class, so the tier's *value* was never measured -
only its cost. It is now: **the 176B's generation buys nothing on this
class, and its judge buys the errors it flags for a tenth of the price.**

---

## 1B-quinque. The judge tier: is a JEV model better than the 176B's own head?

The architecture under test: **the 35B generates, a judge says whether
the answer is right, and the 176B answers only when the judge sends it
up.** The judge is therefore the most valuable thing in the cascade, and
the question is which judge to put there.

Every candidate is judged on the **same 28 organic tasks and the same
35B answers**, so the arms are paired
(`experiments/judge_gate.py`; the artefact is
`experiments/cascade/results/judge-gate-20261021/`).

| the judge | the AUC | the cost a verdict | the errors caught that the deterministic stack misses | the correct answers it breaks |
|---|---|---|---|---|
| **the 176B's own head** (a readout) | **0.864** | 2.52 s | **3 of 5** | **0** |
| JEV-9B (prompted) | 0.837 | 0.21 s | **4 of 5** | 3 |
| Intern-Decision-4B (prompted) | 0.738 | 0.16 s | 2 of 5 | 3 |
| Tiny-Jev-1.7B (prompted) | 0.646 | 0.05 s | 0 of 5 | 0 |
| Intern-Decision-0.8B (prompted) | 0.527 | 0.08 s | 0 of 5 | 0 |

**A JEV model does work as a gate.** JEV-9B is the best *prompted*
judge (AUC 0.837) and it catches **4 of the 5** errors the cascade's
deterministic verifiers (the arithmetic consistency check and
`code_exec`) miss - one more than the 176B's own head.

**And the readout beats the prompted judge at every scale.** The order
is monotone in the model's size and in the mechanism: the 176B's own
readout (0.864) > the prompted JEV-9B (0.837) > the prompted
Intern-Decision-4B (0.738) > the prompted Tiny-Jev-1.7B (0.646) > the
prompted Intern-Decision-0.8B (0.527, a coin). **The two small JEV
models are near chance on another model's answers and are useless as
gates** - which is the same finding the decision-head study recorded
("the readout class generalises: trained-low-bit bodies keep it"), now
on a *third* model's answers.

**The net is the same either way, and that is the finding.** The 176B's
generation fixes 2 of 7 errors and breaks 2 of 21 correct answers, so a
routed error is worth 2/7 and a routed correct answer costs 2/21:

| the gate | the routed | the broken | the net | the time for 28 items |
|---|---|---|---|---|
| the 176B's own head (at 0.65) | 3 | 0 | **+0.86 answers (+3.1 points)** | 132.6 s |
| JEV-9B (at 0.80) | 4 | 3 | **+0.86 answers (+3.1 points)** | 88.6 s |
| the OR of both | 4 | 3 | +0.86 answers | 178.2 s |

JEV-9B's extra catch is **exactly cancelled** by the three correct
answers it routes to a model that breaks 2 of 21. The head's version
never touches a correct answer; JEV-9B's is 12x cheaper a verdict
(0.21 s against 2.52 s - and the head's 2.52 s is *with* the resident
tier on, 3.20 s without it), which is worth 44 s over 28 items.

**The constraint that decides it is the box, not the quality.** A 9B
judge in bf16 needs ~18 GB of host RAM to load and does not fit beside
the 176B on this machine (a load attempt took the box to 4 GB of
`MemAvailable`, and the server with it). The two judges that *do* fit
(0.8B, 1.7B) are near chance. So the choice is between:

- **the 176B's own head** - the best gate (AUC 0.864, 3 unique catches,
  0 false rejects), no extra model, no extra VRAM, 2.52 s a verdict;
- **JEV-9B** - the same net, 12x cheaper a verdict, 3 false rejects,
  and a 9B model that cannot share the box with the 176B.

**The head is the one to ship, and the reason is the false rejects.**
A judge that routes a correct answer to a model with a 2-in-21 break
rate is buying its extra catch with a coin-flip on the answers that were
already right. The head's 0 false rejects is not a small difference -
it is the whole difference between a gate that can be trusted and one
that needs a second opinion.

**And the honest ceiling.** Even the best gate buys **+3.1 points** on
this class (0.821 -> 0.852), because the 176B's generation is net-zero
here. The gate is worth what the escalation is worth, and the escalation
is worth nothing on this class. The judge's value will be on the
classes where the big model is actually better - which is the
measurement the tier has still not made.



**The honest summary of the levers**, in the order they pay:

1. **Ask the 35B.** Identical quality (0.800 vs 0.800), 42.5x faster.
   The 176B is for the escalation set, not the bulk.
2. **Keep the 176B resident** between the tasks that need it, so the
   cold phase is paid once per session rather than once per request.
3. **Run it alone.** Co-siting costs 2.6x, and it is the one lever that
   also *hurts* the small tier.
4. **Let the resident tier hold what the VRAM cache cannot** - 1.9x on
   the cold request, 10x less storage traffic after it, at the price of
   8 GiB of the box's 30 GB.
5. **Shrink the working set** - the Q4X160 build (57.6 GB) is the only
   smaller native on disk, and it is 4% smaller, which the cache
   arithmetic makes worth little.

**What the levers are worth, measured warm** (the same 28 organic tasks,
the 176B alone, the engine warm):

| | the campaign (co-sited) | warm and alone |
|---|---|---|
| the mean per item | 229.7 s | **26.9 s** |
| the decode rate | 3 tok/s | **60-73 tok/s** |

The 42.5x gap against the 35B was **~7x contention and cold cache, and
~6x the model**. Both are real; only the second one is the price of the
tier.

## 1B-sexies. The escalation generator: a 27B dense, not the 176B

The escalation tier's generate arm was net zero with the 176B - 2 of 7
errors fixed, 2 of 21 correct broken - and the reading was that
escalation is worth nothing on this class. The reading was about the
*generator*, not the tier. The 176B is a judgement model: its generation
costs 20.7 s an item and is no better than the 35B's.

The same arm, same 28 organic tasks, same 35B answers, same checkers, on
**Bonsai 2 27B** (`Ternary-Bonsai-2-27B-PQ2_0`, 7.2 GB, one card):

| the arm | correct | errors fixed | correct broken | the cost |
|---|---|---|---|---|
| the 35B alone | 21/28 (0.750) | - | - | - |
| **Bonsai 2 27B alone** | **26/28 (0.929)** | 6 | 1 | 8.92 s/item |
| the 176B head gate (0.65) -> Bonsai | 23/28 (0.821) | 2 | 0 | 3 escalations |
| the oracle gate -> Bonsai | 27/28 (0.964) | 6 | 0 | 7 escalations |

**The tier is worth +5 net on the tasks where the 176B was worth zero**,
and the escalation is cheaper: 8.92 s/item against 20.68 s. (One sample
per task at temperature 0.7, the same protocol as the 176B arm; the one
broken answer - `math-0013` - is the kind a majority vote would settle,
so the +5 is a lower bound, not a point estimate.)

**The gate is now the bottleneck, and its misses are not noise.** The
176B's head sends up 3 of the 7 errors and none of the 21 correct
answers - a clean gate - yet it gives **four of the seven errors
`p >= 0.97`**: both arithmetic answers and both code answers. Those are
*plausible* wrong answers, and a head that reads the answer cannot see
the difference. The head's ceiling on this set is 3 of 7, and no
retraining of a judgement model moves it: the missing signal is in
*executing* the answer, not in reading it.

**So the local cascade's shape is now clear**, and it is not the one the
campaign started with:

1. the 35B answers the bulk;
2. a **checker** where the domain allows one (run the code, recompute
   the arithmetic) - the only tier that can see the four errors the head
   cannot;
3. the judge for the uncheckable remainder (3 of 7 here);
4. Bonsai for what the gate sends up (it fixes 6 of the 7);
5. the API only for what neither the check nor the local generator
   settles.

Steps 1, 3 and 4 are measured; step 2 is the next experiment, not a
result - the benchmark's own checkers are gold answers, and a
production checker is a sandbox the harness does not have yet.

`experiments/cascade/bonsai_arm.py`, the artifact in
`experiments/cascade/results/bonsai-escalation-20261021/` (the rows and
the serve recipe are in `docs/BONSAI-RUNTIME.md`).

## 1B-septies. The API tier, measured (2026-10-22)

The paid API key is expired, so the API arm was run on the free OpenCode
tier: `opencode/deepseek-v4.1-flash-free`, the same DeepSeek V4.1 Flash
family as the current controller, through a fresh OpenCode session per item, one sample per task. The task prompt was
passed exactly as the local arms saw it; the task's own checker was applied
only after the answer was fixed.

| the arm | result | median | mean | max | tokens in / out | cost |
|---|---|---|---|---|---|---|
| **the API (free OpenCode)** | **28/28 (1.000)** | **2.608 s** | 4.970 s | 44.04 s | 66,307 / 4,057 | 0 (free tier) |

The 28 tasks are `experiments/cascade/tasks-organic-28.json`; the per-item
answers and timings are in
`experiments/cascade/results/api-frontier-20261022/results.json`. The
measurement harness is `experiments/cascade/api_frontier_arm.py`.

**The checker correction.** The first pass scored 27/28: `math-0020` answered
`\boxed{6{,}290{,}000}` and the boxed checker did not strip the LaTeX
thousands separators. The answer was correct; the checker was not. The
normalizer now removes `{,}` and commas, with a regression test in
`tests/unit/test_cascade_checkers.py`; the rescored arm is 28/28.

**What this says.** On this 28-task set the API tier is the strongest arm
measured — above the 35B (21/28), Bonsai 27B (26/28) and the judge+Bonsai
route (23/28). The API-driven controller is not measured; that still needs
the tool schema and a driver. The checker remains the missing local tier.

## 1B-octies. The checker: execution first (2026-10-22)

The checker is gold-free. It reads the runtime spec only (`kind`, the task's
own tests, prelude/footer), never `checker.expect`; the 35B answers and the
gold labels are the same as §1B-sexies, and the gold is applied only after
the verdict. Two executable methods ship:

- **code_exec** runs the candidate's last fenced block against the task's
  tests in bwrap: network namespace off, read-only root, temp cwd, 5 s
  timeout, 512 MB address-space cap, no in-process `eval`.
- **arithmetic** re-evaluates every equation the answer asserts about itself,
  including prose equations and unit conversions (cents/dollars,
  inches/feet), with exact Fractions and an AST walker.
- **mutation** corrupts every passing claim (the code's returns, or the
  numeric right side of the first checked equation). A mutation that still
  passes turns the verdict into `unknown`, never `pass`.
- **unknown** is the answer for a choice, a missing frame, or a numeric
  answer with no checkable equation. It is never a pass and never a fail.

| the target | the result |
|---|---|
| the head's four `p >= 0.97` misses | **4/4 fail** (`gsm8k-0038`, `math-0016`, `humaneval-0003`, `humaneval-0015`) |
| the 21 correct 35B answers | **0/21 false rejects** |
| all seven 35B errors | **6/7 fail; 1 unknown** (`truthfulqa-0015`) |
| wrong answers passed | **0** |
| mutation self-test | **17/17 pass verdicts caught** |
| latency | **0.231 ms median; 5.479 ms mean; 29.442 ms p95; 29.707 ms max** |
| the head, for comparison | 2.52 s/verdict; 3/7 errors; 0/21 correct sent up |

The checker passes 17, fails 6 and returns `unknown` for 5. The five
unknowns are the three correct multiple-choice answers, the correct
`math-0020` (a boxed value with no equation to recompute) and the wrong
`truthfulqa-0015`. The four correct unknowns are not false rejects: the
head accepts them (`p` 0.962-0.996); the wrong choice is the head's own
low-confidence case (`p = 0.4215`), which it sends up, and Bonsai does not
fix. So the **checker + head** union catches all 7 errors with 0 correct
answers sent up; the checker alone is the 6/7 execution tier.

The failures are concrete. `gsm8k-0038` writes `24500 cents = $24.50`
(it is $245); `math-0016` writes `10 * 4/9 * 1/27 = 40/81` (it is
`40/243`); both code answers fail their task tests. The arithmetic checker
also catches `gsm8k-0015` and `gsm8k-0033`, the two head-caught errors,
without being told the gold.

The checker is 10,900x cheaper a verdict than the 2.52 s head on the
median, and the expensive tail is the sandbox (29 ms p95); the arithmetic
path is sub-millisecond. `grounding` is implemented as a separate citation
path (`check_grounding`) but is not measured here: these 28 tasks have no
claim/citation surface. The scripted controller is §1B-nonies.

Code and artifacts: `experiments/cascade/verifier.py`,
`experiments/cascade/measure_checker.py`, and
`experiments/cascade/results/checker-v0-20261022/report.json`; the tests are
`tests/unit/test_cascade_verifier.py`.


## 1B-nonies. The scripted controller: 28/28 (2026-10-22)

The loop is the four-tool contract on the measured 35B/head arms, with the
27B and API live:

1. ask the 35B (the measured answer), verify it;
2. `fail` -> 27B; `unknown` -> head `p`; if unconvincing -> 27B;
3. verify the 27B answer; if it still fails or is unknown -> API;
4. accept, logging every `pass` / `fail` / `unknown`, every score and every
   escalation.

The 35B and head are the campaign's measured arms, so the four `p >= 0.97`
misses and the 0/21 correct set stay paired. The 27B is the live
`Ternary-Bonsai-2-27B-PQ2_0` on `:8897`; the API is the free OpenCode
DeepSeek V4.1 Flash session. Gold is applied after each decision.

| the path | served | escalations | correct sent up | mean | conditional |
|---|---|---|---|---|---|
| the head only + live 27B/API | 24/28 (0.857) | 3: 1 27B, 2 API | **0/21** | 10.208 s | 17.810 s |
| **the scripted checker + head + 27B + API** | **28/28 (1.000)** | **7: 5 27B, 2 API** | **0/21** | **9.238 s** | **17.217 s** |

The scripted path is **+4 answers** over the head-only path, with four more
escalations, and is **0.97 s cheaper a task on the mean** (9.24 s vs 10.21 s)
and 0.59 s cheaper on the conditional path (17.22 s vs 17.81 s), because it
does not run the 2.52 s head on the 17 checker passes or the six direct fails.
The old 18.1 s figure is the head path with the recorded 27B; the live 27B/API
head path is 17.81 s.

**The checker path, item by item.** 17 35B answers pass and are served. Four
of the five `unknown` answers are correct and the head accepts them
(`p` 0.962-0.996); the fifth is `truthfulqa-0015` (`p = 0.4215`), the wrong
choice. The six checker failures are all errors. The live 27B serves five of
the seven: `gsm8k-0015`, `gsm8k-0038`, `math-0016`, `humaneval-0003`,
`humaneval-0015`; the API serves `gsm8k-0033` (the live 27B got confused about
the sixth side) and `truthfulqa-0015` (a choice the checker cannot execute).
Both API answers are correct. The 0/21 holds: no correct 35B answer leaves the
first tier.

**The head-only path** sends `gsm8k-0015`, `gsm8k-0033` and
`truthfulqa-0015` up; the live 27B fixes `gsm8k-0015`, the API fixes the other
two; the four `p >= 0.97` errors (`gsm8k-0038`, `math-0016`,
`humaneval-0003`, `humaneval-0015`) are accepted wrong. That is 24/28,
3 escalations, 0/21. The 3/28 escalation target belongs to this path; the
checker path pays four more escalations to catch those four errors and
reaches 28/28.

**Two scorer fixes found by the live run.** The live 27B writes currency as
`\$`, which the arithmetic checker treated as unknown; normalizing escaped
currency lets its `gsm8k-0038` answer pass. The number gold checker also took
the *first* `Final answer` marker; the 27B answer has the phrase "the final
answer is expected…" before the actual `Final answer: 245`, so a correct 27B
answer was scored false. The checker now uses the **last** marker; the same
run re-scores 28/28. The checker's own verdicts were unchanged: 4/4 at
`p >= 0.97`, 0/21 false rejects, 17/17 mutations.

Artifacts: `experiments/cascade/controller_loop.py`,
`experiments/cascade/results/controller-20261022/report.json` and
`steps.jsonl`; the task-checker fix is in `experiments/cascade/checkers.py`
with tests in `tests/unit/test_cascade_checkers.py`. The scripted loop is
measured; the **API-driven controller** (a frontier model calling the four
tools) is not.
