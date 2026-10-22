---
license: mit
task_categories:
  - text-generation
  - question-answering
language:
  - en
tags:
  - evaluation
  - llm
  - local-first
  - cascade
  - gold-free
  - verification
pretty_name: Hive Cascade task sets
size_categories:
  - n<1K
---

# Hive Cascade task sets

Two evaluation sets used to measure the Hive Cascade: a **100-task
bench** and a **664-task community set**. Each task is a prompt, a
checker and the checker's expected value. The runtime protocol never sees
the checker; the evaluation pass reads it only after the answers are
fixed (the gold-free contract).

The paper is not this card. It lives in the repository:
[`cascade/CASCADE-PAPER.md`](https://github.com/sky-is-green/hive/blob/main/cascade/CASCADE-PAPER.md).
The visual companion is the static Space
[`SkyIsNotGreen/hive-cascade-paper`](https://huggingface.co/spaces/SkyIsNotGreen/hive-cascade-paper).

## Contents

| the file | n | the sources |
|---|---|---|
| `tasks-bench.json` | 100 | gsm8k 40, MATH 25, HumanEval 20, TruthfulQA 15 |
| `tasks-community.json` | 664 | gsm8k 300, MATH 200, HumanEval 164 |

Splits: bench **70 train / 30 test**; community **465 train / 199 test**.
Buckets: **reasoning** (gsm8k + MATH), **code** (HumanEval), **qa**
(TruthfulQA, bench only).

The two files under `results/` are the runner's reports from the
2026-10-18 sampled runs (the 35B/27B cascade), included as provenance for
the paper's numbers.

## The task format

```jsonc
{
  "id": "gsm8k-0005",
  "benchmark": "gsm8k",            // gsm8k | math | humaneval | truthfulqa
  "source": "openai/gsm8k",        // the source dataset and, for MATH, the subject
  "bucket": "reasoning",           // reasoning | code | qa
  "prompt": "Rory is retrieving ...",
  "checker": {"type": "number", "expect": 10.0},
  "meta": {"source_split": "test"},
  "split": "test"
}
```

The checker is one of four kinds:

| `checker.type` | what it does |
|---|---|
| `number` | compares the final numeric answer with `expect` |
| `boxed` | parses the last `\\boxed{...}` and compares it |
| `code` | runs the candidate against the HumanEval harness (`prelude` + tests, `entry_point`) |
| `choice` | compares the multiple-choice letter with `expect` |

**The checkers carry the gold on purpose.** They are the evaluation
instrument, not the runtime gate; the cascade's runtime uses verifiers
that never read them.

## Loading

```python
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download("SkyIsNotGreen/hive-cascade", "tasks-bench.json",
                       repo_type="dataset")
tasks = json.load(open(path))["tasks"]
train = [t for t in tasks if t["split"] == "train"]
```

## Sources and attribution

The task text is redistributed from:

- [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) — MIT
- [`EleutherAI/hendrycks_math`](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
  (algebra, counting and probability, geometry, intermediate algebra,
  number theory, prealgebra, precalculus) — MIT
- [`openai/openai_humaneval`](https://huggingface.co/datasets/openai/openai_humaneval) — MIT
- [`truthfulqa/truthful_qa`](https://huggingface.co/datasets/truthfulqa/truthful_qa) (mc1) — Apache-2.0

The packaging and the metadata are MIT; follow each source's licence for
the task text.

## Provenance

- built **2026-10-02** (bench) and **2026-10-07** (community), seed `0`;
- the bench is a stratified 100-task cut of the four sources; the
  community set is the larger gsm8k/MATH/HumanEval pool;
- the paper's runs used these files unchanged (their `sha256` is in the
  runner reports).

## Links

| | |
|---|---|
| the paper | [cascade/CASCADE-PAPER.md](https://github.com/sky-is-green/hive/blob/main/cascade/CASCADE-PAPER.md) |
| the visual companion | [hive-cascade-paper](https://huggingface.co/spaces/SkyIsNotGreen/hive-cascade-paper) |
| the code and the full results log | [github.com/sky-is-green/hive](https://github.com/sky-is-green/hive) |
| the memory system | [SkyIsNotGreen/splinter-memory](https://huggingface.co/SkyIsNotGreen/splinter-memory) |
| the encoder (research checkpoint) | [SkyIsNotGreen/splinter-encoder](https://huggingface.co/SkyIsNotGreen/splinter-encoder) |
