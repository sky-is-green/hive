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

## Still to run

| id | what |
|---|---|
| 1B | the cascade over each model (running) |
| 1C | the MTP drafter on/off (the sidecar is in the Strata format, not llama.cpp's `-md`) |
| 3A-3E | the cascade's arms: the judge, the repair, the gate, the router, the escalation |
| 2A | curated vs FIFO, each model as the generator |
| 2C / 5A | the encoder on a held-out split |
| 4A-4C | the P1-P10 protocol with each model, the 500-turn stability, the PES |
