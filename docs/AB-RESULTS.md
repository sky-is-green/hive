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
1024 dims - against the stock MiniLM on the identical sample. Running.

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
latency. A 2.4 s verdict that catches a third of the errors and breaks
none of the correct answers is a better use of the 176B than a 27 s
generation that breaks more than it fixes.

**The honest limit.** The head's AUC on the 35B's answers (0.871) is
well below its AUC on its own model's answers (0.994) - it is reading a
*different* model's output. It is a **filter, not a replacement**: a
gate that lets a correct answer through untouched and flags a third of
the wrong ones, so the expensive generation is spent on a third of the
cases instead of all of them. That is a 3x cut in the escalation tier's
cost, not a 10x. The dataset's own community head reaches 0.852
per-sample (0.965 within-pool) and says the same thing: *"the in-domain
refit beats the frozen J-Bench head ... validate per model"*.

**What this changes in the design.** The escalation tier has two
options, and the campaign only measured one:

1. **generate** with the 176B (what the campaign did): 27 s warm, 230 s
   co-sited, and on this class it fixed 6 and broke 8.
2. **judge** with the 176B's head (what the dataset already supports and
   what this measures): 2.4 s, catches a third of the errors, breaks
   none of the correct answers.

Option 2 is 11x cheaper than option 1 warm and 96x cheaper than the
campaign's engine, and it is the only one of the two with a positive
expected value on the errors. The campaign's escalation arms should be
re-run against it.


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
