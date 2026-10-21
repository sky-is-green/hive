# The A/B campaign: the scope

**What we have, what we can compare, what each comparison costs, and what
the paper gets from it.**

The inventory the campaign rests on:

| | what it is | where it lives |
|---|---|---|
| a hand-crafted model | the Scion family: **35B-A3B** (35B, 3B active, ternary PQ2_0) and **FlashNext-176B-A6B** (176B, 6B active, ternary Q2_0), plus the k=1 MTP drafter | `SkyIsNotGreen/Scion-35B-A3B`, `SkyIsNotGreen/Scion-FlashNext-176B-A6B` (both public on the Hub) |
| a hand-crafted harness | the **Studio**: the console, the launch panel, the Global controls, the chat, the cascade console, the harvest | `hivebench/` (this repo) |
| a hand-crafted management pipeline | the **curation** (five layers) + the **Global controls** that make it configurable in one place | the sibling's `splinter/` + `harness/controls.py` |
| answer quality reporting | the cascade's `report.json` (per-item verdicts, the findings, the ladder's resolution), the `manifest.json` (provenance), the generated `MODEL-CARD.md` | `experiments/cascade/results/<run>/` |
| API escalation | the ladder's `--api-fallback` arm, behind the human gate, with the escalation lane harvesting every resolution | `experiments/cascade/cascade.py`, `harness/cascade/api.py` |

---

## 1. The comparisons, in five layers

Each row is one A/B: the two arms, the metric, and the falsification
condition. "Cost" is the wall clock on this box (the 176B is the slow
one: 1-10 tok/s measured; the 35B is roughly an order of magnitude
faster).

### Layer 1 — the two models, head to head

| id | arms | metric | falsification | cost |
|---|---|---|---|---|
| **1A** | 35B-A3B vs 176B-A6B, same prompts, same bench, run side by side | the agreement on the extracted final answer; the per-model latency; whether two engines co-host without starving each other | if the agreement is above 95% on the items both can do, the tiering buys nothing on that subset | ~1.5 h (the 176B arm dominates) |
| **1B** | the cascade over each model, alone | served-correct, the false-reject count, the escalation rate | if the 35B's served-correct is within one item of the 176B's at a tenth of the latency, the 176B is not worth its cost on that class | ~1 h each |
| **1C** | 35B with and without the MTP drafter | the decode rate, the acceptance rate | if the acceptance is below the measured 77%, the drafter is off | ~20 min |

### Layer 2 — the curation

| id | arms | metric | falsification | cost |
|---|---|---|---|---|
| **2A** | curated context vs the equal-size FIFO window, each model as the generator | the answer-level fact recall (the paired A/B: the same model, two contexts, the same turn) | if the curated arm does not win on ≥80% of the fact-retrievable turns, the curation is not earning its place | ~3 h per model |
| **2B** | the membrane's thresholds: shipped (0.92 / 0.6) vs recalibrated (0.57 / 0.77) | the fact-presence recall (P2's deterministic diagnostic) and the context size, on the same conversations | if the recalibrated arm loses a fact or inflates the context, hypothesis H is false | **done** - H is **false**: the duplicate test is what costs facts (50.55% → 39.70%), the topic-drift test is free (identical) |
| **2C** | the stock encoder vs the tuned one | the two membrane decisions and the ranking, on a **held-out** split of the conversations | if the tuned encoder does not beat the stock on the held-out split, it is a fit, not a generalisation | ~1 h. **First half done**: at the shipped thresholds the tuned model keeps the facts (50.93% vs 50.55%) and merges 6× more true duplicates at the same precision |

### Layer 3 — the cascade's arms

| id | arms | metric | falsification | cost |
|---|---|---|---|---|
| **3A** | the judge: deterministic vs tiny-jev vs the intern-0.8B head | the detection recall, the false rejects, the ms | the pre-registered rule: a head that false-flags more than twice is off | ~40 min per arm |
| **3B** | the repair: auto vs off | the fixed-wrong count, the content harm to the correct answers | if the repair fixes nothing, the second generation is a tax | ~30 min |
| **3C** | the gate: off vs intern-0.8b | the silent-wrong catches vs the false flags | the pre-registered ≤2 false-flag budget (the current verdict: 3, so off) | ~40 min |
| **3D** | the router: local vs jev, and the mode: gate vs measure | the routing accuracy, the wall clock, the spend | if the router never changes a decision, it is a tax | ~30 min |
| **3E** | the escalation: local vs the API fallback | the served-correct, the cost per item, the human-gate load | if the fallback does not add correctness, the escalation lane is not worth the key | ~30 min |

### Layer 4 — the whole suite, refreshed

| id | what | metric | falsification | cost |
|---|---|---|---|---|
| **4A** | the P1-P10 protocol, with each model | the per-prediction PASS/FAIL/SKIP with the evidence | the protocol's own conditions (P2: recall ≥90% and precision ≥85%; P3: ≥80% of the turns) | ~4 h per model |
| **4B** | the 500-turn stability | the throughput drift, the OOM count, the peak RSS | 0 OOM events, the throughput flat ±10% | ~1 h |
| **4C** | the PES, per configuration | the composite score | the ≥65 / ≥80 bands | rides along |

### Layer 5 — the encoder (already measured, one arm left)

| id | arms | metric | falsification | cost |
|---|---|---|---|---|
| **5A** | the stock L3-v2 vs the tuned v6, on a **held-out** split | the dedup and drift decisions, the ranking | the gate: no job may regress, the cost must hold | ~1 h |

---

## 1b. What the measurements changed

Three facts, measured on this box, reshape the plan.

**1. The 176B's decode rate is a state, not a capability.** The engine
logged 31-42 tok/s in its first sixty generations and 3.1 tok/s in its
last sixty. Alone, with card0 free and the full expert cache, it runs at
**8.2 tok/s** with an **89.8% cache hit rate** - so the storage is not the
bottleneck. The registry's 57 tok/s figure belongs to *Qwen3.8-Flash-Next
(Q2_0)*, a different model; the Scion 176B Q6_K has no such baseline.

**2. Co-siting costs 2.6x, in VRAM *and* in RAM.** The expert cache is
sized "auto" from the free VRAM, so the 35B on card0 halves it (8.55 ->
4.11 GiB) and the miss rate doubles. Worse, the 176B streams 33-87 GB of
experts per request, and **that evicts the 35B's weights from the same
page cache** - so the two engines destroy each other's warm state, and no
VRAM number shows it. Measured with `mincore`: the 176B's file is 12.5%
cached, the 35B's 0%.

**3. The RAID0 is a capacity project, not a speed one.** The physical
read is 75-300 MB/s against a Gen4 drive's 5-7 GB/s - the engine asks
for 5-10% of what the disk can do, and the reads are 8.8 KB scattered
faults, so the win would be queue depth, not bandwidth. SATA is a clear
no: 550 MB/s and ~90K IOPS, so a two-drive SATA RAID0 is slower than the
single NVMe it would replace.

So the plan, reordered by measured effect:

| | what | status |
|---|---|---|
| a | **run the engines one at a time** - the campaign's arms are sequential, and the Studio refuses a second launch on a leased card | **done** (the paired A/B's two arms, sequential) |
| b | **the drain** - an unload waits for `/slots` to go idle before stopping the engine (a 176B reply is minutes of work) | **done** |
| c | **the prefetch** - a launch asks the kernel to read the weights into the page cache first, so a model that fits starts warm | **done** for the 35B; **it cannot work for the 176B** (§1c) |
| d | **the warm/cold readout** - `mincore` per loaded engine in `/v1/server/status`: the number VRAM cannot show | **done** |
| e | the Q4/X160 re-quantisation - the real constraint is a 132 GB working set against a 13 GB page cache | to do (a large download) |
| f | the riser + RAID0 - more escalation candidates resident, on a dedicated ext4 array (not btrfs-on-mdraid) | to do (hardware) |
| g | the frontier key - the cascade's paid tier needs a refreshed key in the auth store | to do - **the key is expired** (`api_key_expired`, re-checked 2026-10-21: `/v1/models` returns `{"code":"api_key_expired"}`) |

**Two items are answered by measurement rather than by work.** (e) and
(f) are the *capacity* projects, and the paired A/B's result is why they
matter less than they looked: the curation's advantage on a strong
generator is a tie, and the large tier's own escalation was net zero
(§1c). What the box needs is not more capacity but a cheaper decision
- the 176B's *judgement* at 4.64 s, not its generation at 20.68 s.


Dropped from the earlier list: raising `--pool-workers` (the storage is
not the bottleneck, so it would buy nothing) and zswap (the swapfile plus
`swappiness=10` already fixed the actual failure mode, and both
alternatives make the *disk* faster, which is the path the fix moved off).

### 1c. The correction, and what replaced (c)

Fact 1 above was measured wrong, and the correction matters more than
the plan it changed.

**The 176B's decode rate is bimodal, and the campaign measured the cold
mode.** The engine's own log says it decodes at **67.7 tok/s** over the
4,424 generations it has served (4,095 of them above 20 tok/s). The
3.1 tok/s of the last sixty generations is the *cold* mode: a fresh
prompt read costs 5 tok/s, a reused prefix 71 tok/s, and the difference
is the OS pulling the model's expert weights off the NVMe. Re-running
the campaign's own 28 tasks against the 176B, **alone and warm**, gives
**26.9 s an item against the 229.7 s** the campaign recorded. About 7x
of the 42.5x was contention and a cold cache; about 6x is the tier.

**And (c), the whole-file prefetch, does not work - it cannot.** The
176B's Q6_K file is 60.2 GB and the box has ~13 GB of page cache.
`posix_fadvise(WILLNEED)` over the whole file, then polling `mincore`
for 80 s, left the file at **0.0% cached** the entire time. The
controlled A/B agrees: 36.9 s with the prefetch, 33.6 s without, both
at ~7 tok/s. The prefetch is still right for the **35B** (11.3 GB, which
fits), and it is left alone for the Strata path, which serves the 176B.

**What replaced it: the engine's resident tier.** Strata can keep the
experts its VRAM cache cannot hold **in ordinary RAM**
(`--resident-budget-gib N`), so they are never re-read from the file.
With an 8 GiB budget, from the same cold state:

| | the stock engine | the resident tier |
|---|---|---|
| the first (cold) request | 33.8 s (7.6 tok/s) | **17.3 s (14.8 tok/s)** |
| the warm rate | 63-74 tok/s | 74 tok/s |
| the storage read, the second request | 1,415 MB | **204 MB** |
| the blobs served from RAM | 0 | **19,178** |

**1.9x on the cold request, 10x less storage traffic after it, the warm
rate unchanged**, at the price of 8 GiB of the box's 30 GB. This is
DeepInfer's EPLB idea (re-place experts by their measured routing
counts) and Strata already has it: `--expert-profile-save` writes out
what the adaptive tier learned, `--expert-profile` reads it back.

**And the escalation tier's shape changed.** The tier asked the 176B to
*answer again*, and that was never measured - only priced. It is now:
the generation arm is **net zero** on this class (2 of 7 errors fixed,
2 of 21 correct answers broken, 20.68 s an item), while the 176B's own
decision head *judging* the small tier's answer is **4.64 s an item**
and finds 2 of 7 errors while breaking none of the 21 correct ones. The
runner carries the tier now (`--verdict-url`).

---

## 2. The order

The order is by *information per hour*, not by ambition.

1. **2B (the cheap half)** - the thresholds' effect on the fact-presence
   recall, no generation. It is the falsification of hypothesis H, and
   it is nearly free. **done** - H is falsified: the fact recall falls
   50.6% -> 39.7% at the recalibrated dedup threshold, because a false
   merge *deletes* a fact while a missed merge only wastes budget, and
   F1 weights the two alike. The fix is a better model, not a better
   threshold.
2. **1A** - the two models head to head. Everything else is downstream of
   it: if the 35B is as good as the 176B on the class the 35B can take,
   the 176B's budget goes somewhere else. **done** - the 35B is as good
   (0.800 both), and the escalation tier's own value is now measured too
   (§1c): the generation arm is net zero, the judge arm is 4.5x cheaper
   and finds the errors.
3. **1B** - the cascade over each model. This is the table the paper's
   §8 wants, per model. **done** - 26.9 s an item warm and alone
   against 229.7 s co-sited.
4. **3A-3E** - the cascade's own arms, on the cheaper model (the arms
   are about the loop, not the model) - so the expensive model is spent
   once, on the cascade that ships. **done** - the arms are in
   `docs/AB-RESULTS.md`; the gate stays off by its own pre-registered
   rule.
5. **2A** - the curation A/B, with each model as the generator. This is
   the memory paper's P3, refreshed. **done** - the paired arm is
   complete (66 comparable turns: a tie at the answer level, 85.15%
   against 84.67%, and a 35% smaller context with the same facts), and
   so is the **family-match arm** (the Qwen-family encoder: 0 discordant
   pairs against the FIFO window where the stock's loses 3, at 22x the
   cost). The hypothesis is confirmed at the mechanism and rejected as a
   deployment.
6. **2C / 5A** - the encoder, on a held-out split. **done** - the
   held-out gate passes for the contrastive tune (the separation
   generalises, the cash-in does not), and the hinge at the operating
   point *does* move the operating point (the recall at 0.92 goes 0.047
   to 0.986) **and breaks the drift decision** (the AUC falls 0.957 to
   0.731). The membrane's two decisions cannot both be served by one
   encoder at the shipped thresholds.
7. **4A-4C** - the full protocol and the stability run, last, on the
   configuration that survived 1-6. **done** - P1-P11 on the 35B
   (PASS 6/11, 2 skipped, the published verdict set) and the 2,000-turn
   stability run.

**What is left is not a measurement.** Every comparison in this plan is
now run. What remains is: the frontier key (the user refreshes it), the
capacity projects (e) and (f) - which the results above make less
urgent, not more - and the *ship* of what the measurements selected:
the verdict tier in the cascade (done, `--verdict-url`), the resident
tier's config in the launch panel, and a drift-specific encoder as a
configuration rather than a new mechanism.


**The 35B's weights** (11.3 GB) have to come from the Hub first, and a
llama.cpp ROCm serve has to come up on card 1 beside the 176B's Strata
serve - that co-hosting is itself the first thing 1A measures.

---

## 3. What the paper gets

The gaps in the Hub edition of the Splinter Memory white paper, and which
comparison fills each:

| the gap | filled by |
|---|---|
| §8's numbers are from 2026-08-23 | 4A, 4B, 4C - the whole table, refreshed |
| §9's Threat 6 (the encoder ceiling) reads as "stop looking for a better encoder" | the encoder measurement: the three jobs, the thresholds, and the fine-tune (already written) - **plus** 5A, the held-out arm |
| P10 (the drift reset) FAILs, unexplained | 2B - the drift detector's threshold was calibrated for another encoder, which is a *mechanistic* explanation the paper can now give |
| P2's precision clause is the open question | 2A, 2B - the precision story, with the curation's decisions measured rather than assumed |
| the paper never mentions the **cascade** | 3A-3E - the answer-quality reporting and the API escalation, as the sibling system's evidence for the curation's effect |
| §12 (the tooling) predates the Studio | the Studio itself: the Global controls, the launch panel, the harvest - with the two released models and the tuned encoder listed as artifacts |
| the reader has nothing to download | the Hub artifacts (the paper, the bench, the encoder, the two Scion releases) - the reproduction path is now a download, not a build |

---

## 4. The risks, and the discipline

- **The OOM.** `systemd-oomd` killed the desktop app at 32.7 GB during
  the August runs, taking the Studio and the Strata serve with it. Rule:
  check `MemAvailable` before every long run; no 3000-word generations
  against the live engine; the 35B's serve is pinned to card 1 and the
  176B keeps card 0.
- **The labels.** The bench's gold is eval-only and the runtime is
  gold-free - that contract does not bend for a benchmark run.
- **The clock.** The 176B arm is the bottleneck: 66 items at 640 tokens
  is about an hour per arm. The campaign is a multi-day GPU plan, and
  the order above is what makes the first day worth something.
- **The co-hosting.** Two engines on two cards is the point of 1A and the
  riskiest thing in the plan (the VRAM reserve, the pids cap, the
  threads). It is measured first, on a small slice, before the full 66.

---

## 5. The decision points

Where this stops and asks:

1. **The slice.** The 66-item organic set (the one the published
   numbers came from - the comparable choice) or the 100-item bench
   (the broader one)?
2. **The first run's size.** 1A on 8 items as a smoke test before the
   66?
3. **Overnight.** The long ones (4A, 4B, 2A) are multi-hour GPU runs -
   start them and leave them, or keep the box free?
4. **The publication.** Everything stays private until the results are
   in and the paper is written.
