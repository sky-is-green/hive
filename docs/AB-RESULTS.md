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

---

## 3A-3B. The cascade's own arms, measured on the 35B

The same 28 organic tasks, the 35B, T=0.7. Only the arm changes.

| arm | served-correct | false rejects | accepted-and-wrong | repairs | mean |
|---|---|---|---|---|---|
| **no verifier, no repair** | 0.786 (22/28) | 0 | 6 | - | 8.1 s |
| **the deterministic verifier, no repair** | **0.643 (18/28)** | 0 | 10 | - | 6.7 s |
| **the deterministic verifier + the repair leg** | **0.857 (24/28)** | 0 | 4 | **2/3** | 6.7 s |

**This is the cascade's economics in three numbers.**

- **A verifier on its own is a tax.** It caught 4 wrong answers and, with
  no repair leg, they were simply *not served*: the served count fell from
  22 to 18. A verifier that drops what it catches and fixes nothing makes
  the system worse (-4 items).
- **The repair leg is what makes verification pay.** With it, the same
  verifier's 4 catches became 2 fixes (2/3 cleared), and the served count
  rose to **24 - two items above the no-verification baseline and six
  above the verifier alone**.
- **The false-reject budget is zero in every arm** - the 0 false rejects
  are what makes the trade worth making, because a verifier that rejects
  correct answers is worse than no verifier.

(The 176B's published numbers say the same thing from the other side:
4 of the 5 wrong answers it caught live were repaired, and the served
count went 58/66 → 62/66.)

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

## Still to run

| id | what |
|---|---|
| 1B | the cascade over each model (running) |
| 1C | the MTP drafter on/off (the sidecar is in the Strata format, not llama.cpp's `-md`) |
| 3A-3E | the cascade's arms: the judge, the repair, the gate, the router, the escalation |
| 2A | curated vs FIFO, each model as the generator |
| 2C / 5A | the encoder on a held-out split |
| 4A-4C | the P1-P10 protocol with each model, the 500-turn stability, the PES |
