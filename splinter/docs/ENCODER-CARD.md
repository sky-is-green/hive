---
license: mit
library_name: sentence-transformers
tags:
  - sentence-transformers
  - embedding
  - context-curation
  - long-context
  - memory-management
  - retrieval
pipeline_tag: sentence-similarity
base_model: sentence-transformers/paraphrase-MiniLM-L3-v2
---

# Splinter Encoder v1

**The drone encoder for long-context memory curation, tuned on the
harness's own traffic.**

This is a 17M-parameter, 3-layer sentence encoder - the same size and
shape as `paraphrase-MiniLM-L3-v2` - tuned for the three decisions a
long-context memory system actually has to make. It runs on the CPU
(about one millisecond per chunk), so the memory it serves never competes
with the model it serves.

> **Status (2026-10-21): a research checkpoint, not a release.** The
> corpus behind these numbers included an ingested knowledge-base
> attachment (79% of its chunks) and one conversation loaded twice. The
> drift column is **withdrawn**, the dedup figures are marked for
> re-measurement, and the replay priced the recommended 0.67 threshold
> as a fact deletion: at the shipped 0.92 this checkpoint keeps 32.66%
> of the facts against the stock's 60.51%. **Do not deploy it with the
> thresholds below**; the shipped encoder and the shipped 0.92 stay. The
> checkpoint and the recipe are published as a record of the training
> and the measurement instrument (`AB-RESULTS.md` §2C-septies).

---

## What it is for

A memory system that curates a long conversation into a bounded context
gives its encoder three questions:

| job | the question | what a good answer looks like |
|---|---|---|
| **dedup** | do these two chunks say the same thing? | the duplicate's cosine is clearly above the rest |
| **drift** | did the conversation change topic? | the recent window and the older one separate cleanly |
| **rank** | which chunk answers this query? | the relevant chunk is at the top |

Most embedding models are trained for the third. This one is trained for
the first two as well, on the traffic of a real long-horizon memory
system.

## The training

- **Base**: `sentence-transformers/paraphrase-MiniLM-L3-v2` (17.4M params,
  384 dims, 3 layers)
- **Data**: 561 duplicate pairs and 2,838 topic-change windows, built from
  81 real conversations (6,975 stored chunks; the corpus included an
  ingested attachment - see the status note) and labelled **lexically** -
  the label is the shared content words, not a model's opinion, so it is
  independent of the encoder being trained
- **Loss**: in-batch negatives over the anchor/positive pairs, plus a
  margin term on a hard negative
- **Curriculum**: the dedup task only. The first two runs that included
  the drift task *damaged* the drift signal (AUC 0.945 → 0.762): the
  in-batch negatives of that task pull the embedding of a window toward
  the embedding of a chunk, which moves the detector's own cosines
- **Hardware**: CPU, one epoch, 35 steps, about a minute

## The measured results

Measured on 4,056 duplicate pairs and 6,533 topic-change cases, against
the base model, at the operating points the system ships.

| | dedup F1 | drift F1 | drift AUC | ms/chunk |
|---|---|---|---|---|
| the base model, at the thresholds the harness shipped | 0.089 | withdrawn* | withdrawn* | 1.3 |
| the base model, at its own best thresholds | 0.761 | withdrawn* | withdrawn* | 1.3 |
| **this model, at its own thresholds** | **0.862** | withdrawn* | withdrawn* | 1.1 |

\* The drift column was measured on the pre-correction corpus and is
withdrawn; the dedup row is from the same corpus and is marked for
re-measurement.

**The thresholds are per-encoder - and that is exactly why they are not
shipped here.** This card used to recommend `{"dedup_threshold": 0.67,
"drift_threshold": 0.74}`. The replay says the dedup half of that
recommendation halves the facts (32.66% against 60.51% at the shipped
0.92), and the drift corpus that produced 0.74 was invalid. Treat the
table above as the recipe's gate, not as an operating point.

## How to use it

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("SkyIsNotGreen/splinter-encoder", device="cpu")
vectors = model.encode(["the chunk", "the other chunk"],
                        normalize_embeddings=True)
similarity = float(vectors[0] @ vectors[1])   # > 0.67: the same fact
```

The architecture is unchanged from the base model: one embedding
function, mean pooling, a cosine. It is a **research checkpoint, not a
drop-in** for the Studio's drone: the replay in `AB-RESULTS.md`
§2C-quater shows why the shipped encoder keeps its place. The checkpoint
loads, and the gate reproduces with
`experiments/encoder_finetune.py --evaluate <path>`.

## The measurement

The evaluation suite is in the harness repository
(`experiments/encoder_suite.py`) and the fine-tuner next to it
(`experiments/encoder_finetune.py`). The measurement is deterministic:
the corpus is read from disk, the pairs are sampled with a fixed seed,
and the labels are lexical.

**The honest limits.** The labels are lexical, so a paraphrase that
shares no content words counts as two different facts. The corpus is the
harness's own traffic (81 conversations), not the world's. And nothing
here claims the model is better at *ranking* - the ranking numbers are
the same as the base model's, on the same curve every stock encoder
lands on. What changed is the two decisions the membrane makes.

## Citation

```
@misc{splinter-encoder-v1,
  title  = {Splinter Encoder v1: a drone encoder for long-context memory curation},
  author = {{HiveBench Studio contributors}},
  year   = {2026},
  url    = {https://huggingface.co/SkyIsNotGreen/splinter-encoder}
}
```

## License

MIT. See `LICENSE`.
