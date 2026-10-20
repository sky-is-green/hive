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
  81 real conversations (6,975 stored chunks) and labelled **lexically** -
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
| the base model, at the thresholds the harness shipped | 0.089 | 0.151 | 0.945 | 1.3 |
| the base model, at its own best thresholds | 0.761 | 0.554 | 0.945 | 1.3 |
| **this model, at its own thresholds** | **0.862** | **0.573** | 0.943 | 1.1 |

**The thresholds are part of the release.** They are per-encoder - the
cosine distribution moves when the model is trained - so the operating
points that ship with this model are:

```json
{"dedup_threshold": 0.67, "drift_threshold": 0.74}
```

Using the base model with these thresholds is much worse than using the
base model with its own (0.761 / 0.554), and using this model with the
old thresholds loses most of the gain. Model and thresholds travel
together.

## How to use it

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("SkyIsNotGreen/splinter-encoder", device="cpu")
vectors = model.encode(["the chunk", "the other chunk"],
                        normalize_embeddings=True)
similarity = float(vectors[0] @ vectors[1])   # > 0.67: the same fact
```

The architecture is unchanged from the base model: one embedding
function, mean pooling, a cosine. It is a drop-in replacement for the
model the HiveBench Studio loads, and the Studio can run it directly -
pick it in the Global tab's encoder block, set the two thresholds above,
and restart the Studio (the drone is built once per choice).

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
