---
license: mit
library_name: sentence-transformers
tags:
  - context-curation
  - long-context
  - memory-management
  - conversational-ai
  - retrieval
  - llm
  - research-paper
pipeline_tag: sentence-similarity
---

# Long-Context Memory Curation Management

**Splinter Memory — a managed-decay context curation architecture for long-horizon LLM conversations.**

This card is the Hugging Face edition of the Splinter Memory white paper. It
is written for the people who build LLM applications and want the context
window to *stay useful* as the conversation grows: the memory layer, the
mechanisms, the measured results, and the code that implements them.

The canonical paper lives with the code
([sky-is-green/splinter-memory](https://github.com/sky-is-green/splinter-memory));
this edition adds the encoder measurement taken on the harness's own traffic
(§7) and the operational notes a practitioner needs.

---

## The problem, in one paragraph

A long conversation is not a long document. Facts that mattered twenty turns
ago fall out of a rolling context window; the model answers from what
survives, not from what happened; and the surviving context is mostly noise
by then. Truncation and recency are cheap and wrong: they throw away
exactly the old fact the user came back for. Keeping everything is worse -
the prompt grows, the KV cache grows, the cost grows, and the model
"loses the middle" while the interesting fact sits somewhere in the middle.

## What this does

Splinter Memory keeps a **managed** memory outside the model and hands the
model a **bounded, relevance-ranked** context every turn:

- a small fleet of **bidirectional encoders** ("drones") scores every stored
  chunk against the current query - cheap, and far better at similarity than
  a generative model of the same size;
- a **membrane** drops what is said twice and notices when the topic
  changes;
- a **retention layer** makes forgetting a *design parameter* - an
  escalating-friction decay, a remembrance pass that rescues facts worth
  keeping, and a "comb" (an SSD tier) that resurrects topics that returned
  after being evicted;
- a **focal layer** assembles a bounded context on an adaptive token budget
  and hands it to any OpenAI-compatible backend (llama.cpp, LM Studio, vLLM,
  a hosted API).

The model never sees the raw history. It sees the part that is worth its
attention, every turn, at a cost that does not grow with the conversation.

**The separation is the point.** The generative model is good at producing
answers and the wrong tool for selecting context; measured live, using the
27B generative model as an encoder proxy cost tens of seconds per turn on
the hot path, while the 17M-parameter drone did the same job in
milliseconds.

## The measured results

| Metric | Naive FIFO (baseline) | Target | Measured |
|---|---|---|---|
| Throughput over 500 turns | decays | flat ±10% | **flat** (14.5 → 15.5 tok/s over 308+ turns), 0 OOM events |
| Retrieval recall (facts the model actually stated) | n/a | ≥75% / ≥90% | **90.3%** live, 93.5% on the isolated-store replay |
| Retrieval precision (sentence proxy) | near-chance past the window | ≥70% / ≥85% | **10.7%** - *the encoder ceiling* (§7) |
| Context sufficiency vs. an equal-size FIFO window | — | ≥80% of turns | **85.1%** of the fact-retrievable turns (4.4:1) |
| Routing accuracy | — | ≥80% / ≥92% | **100%** |
| Added per-turn latency | 0 ms | <50 / <30 ms | **≈18 ms** on a small store (3.4 ms assembly + ~15 ms scoring) - **and it grows with the store**: 22.8 ms at 212 chunks, 156.8 ms at 1,612 (§7.4) |
| Peak RSS over 500 turns | — | small | **34.7 MB** |
| Pipeline Efficiency Score | ~30 | ≥65 / ≥80 | **80.0 (GREEN)** live, 73.1 on the 20-conversation evidence run |

Full protocol, per-hypothesis logic chains and falsification conditions:
§5 of the canonical paper.

### The whole protocol, refreshed on the 35B-A3B

The P1-P11 protocol re-run live against the **35B-A3B ternary** (a
12-conversation fixture subset), to ask whether the verdict set is a
property of the architecture or of the generator it was measured on:

| | verdict | the evidence |
|---|---|---|
| P1 constant throughput | **PASS** | the throughput does not decay |
| P2 retrieval precision | **FAIL** | the precision clause (the known encoder ceiling, §7) |
| P3 context sufficiency | **PASS** | the curated context against an equal-size FIFO window |
| P4 domain-dependent decay | **PASS** | code m90 1.8 vs prose m90 1.2 - a gap of 0.6, beyond the 0.2 band |
| P5 targeted masking | SKIP | a training experiment |
| P6 confidence escalation | **FAIL** | the escalated recall is worse (0.174 vs 0.565) |
| P7 auditor-human agreement | SKIP | it needs human raters (90.25% measured 2026-08-23) |
| P8 routing accuracy | **PASS** | 200/200 |
| P9 densest-duplicate retention | **PASS** | the densest chunk wins 100% of the turns |
| P10 drift reset | **FAIL** | the reset changes the recall by 0.0 points |
| P11 comb resurrection | **PASS** | the pressure regime: 100% topic-return recall against 20% without the comb |

**PASS 6/11, 2 skipped - the published verdict set, exactly.** The
published run (2026-08-23/24, on a 27B, on the full corpus) reported
P1/P3/P4/P8/P9/P11 PASS, P2/P6/P10 FAIL, P5/P7 SKIP. The same set
reproduces on a different model and a smaller corpus, which says the
protocol measures the architecture rather than the generator.

**One of the failures now has a candidate mechanism.** P10 (the drift
reset) measures a detector whose threshold was calibrated for a different
encoder (§7): forced on, it fires on most turns, so "the reset has no
effect" may be measuring a detector that fires far too often rather than
a reset that does not work. (The exact rate was measured on the
pre-correction corpus and is withdrawn with the rest of §7.) The verdict
stands as measured; the mechanism is now in question, which is a
different claim.

**And the one result that does not carry over is P3's margin.** On the
27B the curated context won 115 turns outright to 26; on the 35B it is a
tie (§7.3). The *verdict* holds (the curated context is sufficient, and
at least as good on 88.4% of the turns); the *margin* does not. The
curation's accuracy advantage is a function of the generator - what
survives for a strong one is the bounded context and the cost.

## Use it

Three ways, in increasing order of control:

**1. As a sidecar** - the Studio serves an OpenAI-compatible
`/v1/chat/completions` that curates before it generates. Point your client
at it; nothing else changes.

```bash
git clone https://github.com/sky-is-green/hivebench
cd hivebench && python -m harness --port 8765
```

**2. As a library** - the memory system is importable, engine-agnostic and
has no GPU requirement for the encoder tier.

```python
from cortex.splinter import Splinter
from cortex.config import SplinterConfig

s = Splinter(config=SplinterConfig(max_context=8192))
result = s.process_turn("what did we decide about the KV cache?", "my-conversation")
print(result.assembled.content)     # the curated context, bounded
```

**3. As the benchmark** - the P1-P12 protocol, the deterministic
diagnostics and the self-contained run bundles are in the repository, so the
claims can be re-measured on your own hardware and your own backend.

## How it works

| Layer | Function | Core mechanisms |
|---|---|---|
| **Cortex** | Orchestration & routing | task classification, drone dispatch, congestion detection, graceful degradation |
| **Sieve** | Relevance scoring | ultra-small encoder (17M params, 384-dim) for fast similarity; a medium encoder for uncertain cases; confidence via prediction variance |
| **Membrane** | Selective filtering | semantic deduplication (cosine > 0.92, keep the densest), topic-drift detection and reset |
| **Retention** | Memory & decay | remembrance pass, escalating-friction decay, stale-context acceleration, the "comb" (surplus SSD tier, topic-return resurrection) |
| **Focal** | Assembly | adaptive token budget (1k-6k by route tier), confidence-weighted sorting, bounded context construction |

**Per turn:** user input → Cortex classifies it → Sieve scores every stored
chunk → Membrane deduplicates and checks the topic → Retention applies decay
and the remembrance pass → Focal assembles a bounded context → the backend
generates → the reply is observed back into the store.

### The four postulates

1. **Context curation** - for a fixed token budget, a relevance-ranked
   selection from the full history dominates a contiguous window of the same
   size, and the gap widens with conversation length.
2. **Managed decay** - forgetting is a design parameter, not a failure mode;
   escalating friction beats both unbounded growth and uniform forgetting.
3. **Separation** - bidirectional encoders for comprehension, an
   autoregressive decoder for generation, is the better allocation of the
   same compute.
4. **Ground-truth bootstrap** - an LLM auditor asked about *context
   utilization* (not answer correctness) agrees with human labels 90.25% of
   the time, which is enough to drive parameter tuning without a human
   labelling pass.

## The harness that runs it, and the pipeline that manages it

The memory system is a library; the **HiveBench Studio** is the harness
that runs it, and it is where the management pipeline lives.

### The management pipeline

A long-horizon memory has to be *managed*: every knob that decides what
the model sees used to live in three different menus, and a wrong setting
was invisible. The Studio has one place for all of it - the **Global
harness controls** - and every block of it is a *baseline* that the more
specific thing beats:

| block | what it decides |
|---|---|
| **the encoder** | which small model does the retrieval and the memory management - you choose it, and the next conversation runs on it (no restart) |
| **Memory & RAG** | the curation baseline: the context window, the stale wall, the dedup and drift thresholds, the comb, the medium drone |
| **Quality** | the cascade's judge, repair, gate and escalation - the defaults every run starts from |
| **MCP** | what an external client may read and write of the memory |
| **Runtime** | the sampling a reply is drawn with, and the context, GPU layers and threads an engine is launched with |

The point is not tidiness. It is that **a setting which is not consulted
is indistinguishable from a setting which does not work**, and the only
way to know which one you have is to wire them and then measure.

### The harness itself

| | |
|---|---|
| the console | Home, Settings, Global, Cascade, and a "More tools" group (Agent, Engines, Local library, Providers, Hub, Inspector, Training) |
| the chat | Splinter mode (one curated reply) or Agent mode (the tool loop); the reply says who produced it |
| the launch panel | llama.cpp or Strata, per engine, from the profiles; stop closes the connection so the *engine* stops; unload gives the VRAM back and retires the orphans |
| the cascade console | every runner knob in one form, with the Global tab's Quality block as its starting values |
| the harvest | every turn (the query, the reply, the curation that was used) written to a jsonl, off by default - this is the data the encoder is trained on |

### Answer quality reporting and API escalation

The curation's effect is not only felt, it is reported. Every cascade run
writes a per-item report (the verdict, the detector findings, the repair
attempt, the ladder's resolution), a provenance manifest (the model
sha256, the stack, the git commit, the task-set sha, the decode
settings) and a generated model card; and the escalation lane harvests
every ladder-resolved item to `escalation.jsonl` next to the report -
training-side only, and read by nothing in the runtime.

The escalation ladder itself is the last stage: an answer the local
verifiers still flag can go to a human gate (verbose on the first
trigger, compact afterwards) and then to an API tier, with the served
answer, the cost and the reason recorded. The companion paper
(*The Hive Cascade*) documents the loop, the gold-free runtime contract
and the measured verdicts; the benchmark and the run reports are
published next to this card.

## The evidence, hypothesis by hypothesis

| | Hypothesis | Verdict |
|---|---|---|
| P1 | Constant throughput over 500 turns | **PASS** |
| P2 | Retrieval precision ≥85% at ≥90% recall | **SPLIT** - recall PASS (90.3%), precision FAIL (the encoder ceiling) |
| P3 | Context sufficiency ≥80% of turns vs. an equal-size FIFO window | **PASS** (85.1% on the 27B); on the 35B-A3B the answer-level result is a tie (88.4% at-least-as-good, 13 strict wins to 17) - see §7.3 |
| P4 | Domain-dependent decay curve (code vs. prose separate) | **PASS** (code m90 1.8 vs. prose 1.2) |
| P5 | Targeted masking | **PASS** |
| P6 | Confidence escalation | **FAIL** |
| P7 | Auditor-human agreement ≥90% | **PASS** (90.25% on 400 items) |
| P8 | Heuristic routing floor | **PASS** (100%) |
| P9 | Densest-duplicate survives deduplication | **PASS** |
| P10 | Drift reset | **FAIL** |
| P11 | Comb resurrection of returned topics | **PASS** (deterministic and live-validated) |
| P12 | Store-time fact distillation | protocol only |

The failures are kept in the paper with their evidence. A pre-registered
protocol that only reports passes is not evidence.

## 7. The encoder: what we measured on our own traffic

> **Correction (2026-10-21).** The corpus behind this section included an
> ingested knowledge-base attachment: 17 of its 82 files were chunked
> `[block n/m]` documents (5,531 of 6,977 chunks), and one conversation
> was loaded from two state directories. The drift figures below (the
> 0.945 AUC and the "fires on 571 of 628 turns" mechanism) are
> **withdrawn**, and the tuned encoder was rejected by the replay (see
> below). The shipped encoder and the shipped 0.92 stay. The correction
> and the clean measurement are in `AB-RESULTS.md` §2C-septies; the
> retrieval results elsewhere in this card come from different runs and
> are unaffected.

The harness gives its encoder three jobs, and only one of them is a ranking:

| job | question | where |
|---|---|---|
| **rank** | which stored chunk answers this query? | the sieve |
| **dedup** | do these two chunks say the same thing? | the membrane (cosine > 0.92) |
| **drift** | did the conversation change topic? | the membrane (1 - cosine > 0.6) |

The prior encoder comparison measured **rank** only, on the synthetic
fixtures, and found that six encoders (a 22M MiniLM, a 400MB cross-encoder,
a 568M retrieval-specialised encoder and two contrastively tuned variants)
all landed on the same top-K curve - so it concluded the ceiling was
*data-structural* and that the remedy was "stop looking for a better
encoder".

We re-measured all three jobs on the harness's **own** traffic - 81
conversations, 6,975 stored chunks, 4,056 duplicate pairs and 6,533
topic-change cases, labelled lexically so the label is independent of the
encoder being measured (`experiments/encoder_suite.py`). The result
changes the picture, and it is not about the ranking:

**The thresholds that ship with the harness were calibrated for a different
encoder.** The production drone was swapped from `all-MiniLM-L6-v2` to
`paraphrase-MiniLM-L3-v2` on 2026-08-23; the membrane's two thresholds were
not re-derived.

| operating point | measured |
|---|---|
| dedup at the shipped threshold (0.92) | precision 1.00, **recall 0.03** - the duplicate test almost never fires |
| dedup at its own best threshold (~0.56) | F1 **0.75** (precision 0.62, recall 0.95) |
| drift at the shipped threshold (0.6) | precision 0.08, **recall 0.96** - the topic-change test fires on almost every turn |
| drift as a *ranking* signal | **withdrawn** - measured on the pre-correction corpus (79% an ingested attachment), see above |

### 7.1 The F1 is the wrong target, and the threshold is not the lever

The obvious conclusion - "move the thresholds to the encoder's optimum" -
is **falsified** by a deterministic replay of the corpus through the real
pipeline (50 conversations, 493 measurable turns, the P2 diagnostic's own
notion of a fact; `experiments/threshold_ab.py`):

| arm | dedup / drift | fact recall | context | verdict |
|---|---|---|---|---|
| shipped | 0.92 / 0.6 | **50.6%** | 159 tok | - |
| the dedup alone | 0.57 / 0.6 | **39.7%** | 90 tok | **falsified** |
| the drift alone | 0.92 / 0.77 | **50.6%** | 159 tok | **survives** |

**The duplicate test's two errors are not interchangeable.** A duplicate
test that fires wrongly *deletes* a chunk, and the fact inside it is gone
irreversibly; a duplicate test that fails to fire only leaves a redundant
chunk in the context, which costs budget and nothing else. F1 weights
them equally, so F1 points at the wrong threshold: at 0.57 the encoder
merges chunks that are *about* the same thing but carry *different* facts,
and the fact recall falls by a fifth. The shipped 0.92 is conservative
**for a reason** - its precision is 1.00, so it never deletes a fact - at
the price of catching only 5% of the redundancy.

The topic-drift test is the opposite case: it only *penalises* the
off-topic chunks, it never deletes them, so its recalibration is free.

**The model is not the lever either.** A small encoder tuned on the
harness's own traffic (17M parameters, one epoch, the dedup task only)
raises the dedup decision at 0.92 from a recall of 0.046 to 0.986 on the
held-out job test. The replay then prices that decision: at the shipped
0.92 the tuned encoder keeps **32.66%** of the facts against the stock's
**60.51%**, for a context 2.7x smaller. The decision improved; the
curation nearly halved. The shipped encoder and the shipped 0.92 stay.

The reason is structural: a true duplicate merged is a fact deleted from
the context, so improving the dedup decision is improving the deletion.
The threshold and the model are two dials on the same trade-off curve,
and the fact-preservation constraint fixes where on it the system sits.
The clean replay and the three failed fine-tunes are in `AB-RESULTS.md`
§2C-quater through §2C-septies.

### 7.4 The curation's cost grows with the conversation it curates

The stability run - 2,000 turns of a growing conversation, the real drone,
no model (`experiments.stability_500.py`):

| turns | the store | the assembled context | the curation's cost (p50) | the peak RSS |
|---|---|---|---|---|
| 250 | 212 chunks | 322 tok | 22.8 ms | 519 MB |
| 500 | 412 chunks | 320 tok | 37.3 ms | 564 MB |
| 1,000 | 812 chunks | 335 tok | 89.7 ms | 594 MB |
| 2,000 | 1,612 chunks | 332 tok | 156.8 ms | 667 MB |

**The context is flat and the cost is not.** The assembled context moves
by 10 tokens across a 7.6x range of store - the curation's promise holds
✓✓ - and nothing is OOM-killed over 2,000 turns ✓. But the curation's
per-turn cost is **linear in the store**: about 7x for 7.6x the chunks,
because every chunk is scored every turn.

So the memory's trade is explicit, and the published "≈18 ms added per
turn" is the *small-store* number: the prompt stays bounded (the
expensive part for the model) while the curation gets more expensive as
the conversation grows. The two costs trade against each other, and the
crossover - the point where curating costs more than the tokens it saves
- is a measurement the system does not yet make.

### 7.3 The curation's advantage is generator-dependent

The paired A/B - the same model, the same turns, the curated context
against the last-4k-token window of the same history - is the memory
system's headline measurement (P3). It was measured on a 27B, and it has
now been measured on the **35B-A3B ternary** (180 measurable turns):

| | the curated arm | the FIFO arm |
|---|---|---|
| the answer's fact recall | 81.1% | 81.7% |
| the context fidelity (the answer's terms sourced from its own context) | **0.241** | 0.208 |
| the turns where the answer is at least as good | **88.4%** (the bar: 80%) | - |
| the strictly-better turns | 13 | 17 |
| the context sufficiency (P3-style) | **97.5%** >= FIFO | - |

**On the 27B the curated arm won 115 turns outright to 26 (4.4:1). On the
35B it is 13 to 17** - a tie - and the answers are equal (81.1% against
81.7%). The "neither" bucket collapses too (145 of 175 on the 27B, 34 of
180 here): the 35B answers from *whatever* context it is given.

**The scoping result: the better the generator is at long context, the
less the curation has to add.** The FIFO window is the most recent turns
verbatim, and a model that uses them well does not need a curation to find
them. This is what Postulate 3 (the separation) implies - the curation is
a substitute for a *weaker* reader - and it is the first thing the
measurements had not tested.

**What survives.** The curated context is still sufficient on 97.5% of the
turns, its fidelity is ahead, and - the part the numbers above do not
touch - the curation is what keeps the context **bounded**: the FIFO arm
is 4k tokens of pure recency, and the two arms are equal *at that
budget*. The honest claim for a strong generator is the bounded context
and the cost, not the better answers.

### 7.2 What we changed, and what we did not

- The shipped configuration stays: the stock encoder at the shipped
  thresholds, because that is the configuration the published evidence was
  produced with.
- The next fine-tune weights a false merge above a missed merge
  (precision-weighted at the operating point) instead of optimising F1 -
  the asymmetry makes F1 the wrong target - and is measured on a held-out
  split, because the current one was trained and measured on the same
  traffic (a fit, not a generalisation).
- The full sweep, the fine-tune runs and the replay are in
  `docs/ENCODER-NOTES.md` in the harness repository.

## Limits, honestly

- **Precision is the open question.** The context that reaches the model is
  fatter than it needs to be. The recall evidence says the facts the model
  stated are in there; an independent, non-circular precision number is
  still open.
- **The auditor sees the answer.** P7's agreement check mitigates the
  circularity but does not remove it: the auditor is the same model family
  that produced the answer.
- **Synthetic-corpus bias.** The long-horizon corpus is generated. The
  measurement in §7 is on real traffic; the P1-P12 verdicts mostly are not.
- **Cross-conversation contamination and hedge-replies** are harness-level
  failure modes that look like architectural failures if you do not
  isolate conversations per store and filter refusals out of the store.
- **The decay curve is an engineering heuristic** with cognitive-science
  inspiration, not a validated model of human forgetting.
- **The large tier is a state, not a model.** The 176B-A6B answers at
  60-73 tok/s warm and 3 tok/s cold, on the same engine and the same
  request: a fresh prompt read costs 5 tok/s and a reused prefix 71, and
  the cold phase is the OS pulling 60 GB of expert weights off the NVMe
  (the file is 20.8% page-cached on a 30 GB box). Warm and alone it
  answers the cascade's 28 tasks in 26.9 s an item; co-sited with the
  small tier, in 229.7 s. A deployment that measures the large tier
  cold, or with another engine resident, measures the box and not the
  model. Two levers pay and the rest do not: keep it warm, and run it
  alone. A whole-file prefetch does not - a 13 GB page cache cannot hold
  a 60 GB file, and the kernel's `WILLNEED` left it at 0% cached for
  80 s. The engine's resident tier (`--resident-budget-gib`) does: it
  keeps what the VRAM cache cannot hold in ordinary RAM, and the storage
  read on the second request falls from 1.4 GB to 146 MB.

## Reproduce

```bash
# the offline suite (deterministic diagnostics, no model needed)
pytest tests/unit tests/integration -q

# the encoder sweep on your own traffic
python -m experiments.encoder_suite --encoders l3 l6 bge-small e5-small gte-small

# the threshold replay (deterministic: no model, no generation)
python -m experiments.threshold_ab

# the cascade pilot (local model, deterministic verifiers)
open http://127.0.0.1:8765/cascade
```

The repository ships the run bundles, the event logs and the label files, so
a verdict can be traced back to the bytes that produced it.

### The artifacts

| artifact | what it is |
|---|---|
| `SkyIsNotGreen/splinter-memory` | this paper, the figures, the license and the citation |
| `SkyIsNotGreen/splinter-encoder` | the tuned drone encoder, its card, its configuration and its gate results |
| `SkyIsNotGreen/hive-cascade` | the cascade's benchmark (a 70/30 split) and two full run reports |
| `SkyIsNotGreen/Scion-35B-A3B` | the ternary 35B-A3B model and its decision head |
| `SkyIsNotGreen/Scion-FlashNext-176B-A6B` | the ternary 176B-A6B model |

The releases are private while the results are being refreshed; they open
when the refreshed numbers are in.

## License and citation

MIT. See `LICENSE` and `CITATION.cff` in this repository.

```
@misc{splinter-memory-2026,
  title  = {Splinter Memory: A Managed-Decay Context Curation Architecture
            for Long-Horizon LLM Conversations},
  author = {{HiveBench Studio contributors}},
  year   = {2026},
  url    = {https://github.com/sky-is-green/splinter-memory}
}
```

**Community invitation.** The protocol is written to be challenged: every
prediction has a logic chain, a measurement and a falsification condition.
If your hardware produces a different verdict, that is a result - open an
issue with the run bundle.
