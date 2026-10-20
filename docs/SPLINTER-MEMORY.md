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
| Added per-turn latency | 0 ms | <50 / <30 ms | **≈18 ms** (3.4 ms assembly + ~15 ms scoring) |
| Peak RSS over 500 turns | — | small | **34.7 MB** |
| Pipeline Efficiency Score | ~30 | ≥65 / ≥80 | **80.0 (GREEN)** live, 73.1 on the 20-conversation evidence run |

Full protocol, per-hypothesis logic chains and falsification conditions:
§5 of the canonical paper.

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

## The evidence, hypothesis by hypothesis

| | Hypothesis | Verdict |
|---|---|---|
| P1 | Constant throughput over 500 turns | **PASS** |
| P2 | Retrieval precision ≥85% at ≥90% recall | **SPLIT** - recall PASS (90.3%), precision FAIL (the encoder ceiling) |
| P3 | Context sufficiency ≥80% of turns vs. an equal-size FIFO window | **PASS** (85.1%) |
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
| drift as a *ranking* signal | **AUC 0.945** - the signal is strong; the threshold is wrong |

The encoder is not the bottleneck. The **operating points** are - and they
are per-encoder, so they have to travel with the model. The full sweep
(model families, sizes, dimensions, CPU cost per chunk) is in
`docs/ENCODER-NOTES.md` in the harness repository, and the evaluation sets
are published as a dataset next to this card.

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

## Reproduce

```bash
# the offline suite (deterministic diagnostics, no model needed)
pytest tests/unit tests/integration -q

# the encoder sweep on your own traffic
python -m experiments.encoder_suite --encoders l3 l6 bge-small

# the cascade pilot (local model, deterministic verifiers)
open http://127.0.0.1:8765/cascade
```

The repository ships the run bundles, the event logs and the label files, so
a verdict can be traced back to the bytes that produced it.

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
