# Hive

**A local-first stack for memory, verification, and routing in language models.**

Hive began as a benchmark for a long-context memory system. It now contains
two papers, the systems they describe, and the harness that measures them.

| the paper | what it is |
|---|---|
| **[Splinter Memory](splinter/SPLINTER-WHITE-PAPER.md)** | managed-decay context curation for long-horizon conversations: a bounded context, a membrane for dedup and drift, and a small encoder fleet. The [HF card](https://huggingface.co/SkyIsNotGreen/splinter-memory) is the Hub edition; the [project](splinter/README.md) has the code. |
| **[The Hive Cascade](cascade/CASCADE-PAPER.md)** | a gold-free verification-and-repair loop: solve, detect, repair, escalate. It carries the role taxonomy, the controller contract, the measured tier results, and the negative results I kept. The [visual companion](https://huggingface.co/spaces/SkyIsNotGreen/hive-cascade-paper) walks through it. |

**Two papers, one stack.** The memory decides what the model sees; the cascade
decides whether the answer is trusted; HiveBench measures both without an
answer key in the serving path.

## What this repo does

HiveBench is the evaluation suite and Studio sidecar. It answers two questions
that ordinary benchmarks do not:

1. **Was the answer right?** Deterministic verifiers (the task's own tests,
   arithmetic consistency, canonical forms), not an LLM judge in the evidence
   path.
2. **Was the context the reason?** The same turns and the same model, one arm
   curated and one naive, scored the same way.

The harness is local-first: models, verification and memory stay on the box;
the API is opt-in. It runs the cascade, the memory arms and the A/B batteries
from the same result schema, and every run leaves a report, a manifest and a
harvest. The detailed guide is in [`hivebench/README.md`](hivebench/README.md).

## Layout

| path | what lives there |
|---|---|
| [`splinter/`](splinter/README.md) | the memory system, its white paper, the encoder notes and figures |
| [`cascade/`](cascade/README.md) | the cascade paper, the pilot, the controller contract and the prior-art survey |
| [`hivebench/`](hivebench/README.md) | the evaluation suite and Studio sidecar docs |
| [`docs/`](docs/) | the shared record: [status](docs/STATUS.md), [A/B results](docs/AB-RESULTS.md), failures, engine notes |
| `harness/`, `experiments/`, `tests/`, `testing/` | the implementation that runs the suite |

The paper and the implementation live side by side; the papers are written to
be re-run, not just read.

## Start here

| | |
|---|---|
| the two papers | [Splinter Memory](splinter/SPLINTER-WHITE-PAPER.md) · [The Hive Cascade](cascade/CASCADE-PAPER.md) |
| the current state | [measured / invalid / unmeasured / blocked](docs/STATUS.md) |
| the visual companion | [hive-cascade-paper](https://huggingface.co/spaces/SkyIsNotGreen/hive-cascade-paper) |
| the task sets and run reports | [SkyIsNotGreen/hive-cascade](https://huggingface.co/datasets/SkyIsNotGreen/hive-cascade) |
| the memory cards | [splinter-memory](https://huggingface.co/SkyIsNotGreen/splinter-memory) · [splinter-encoder](https://huggingface.co/SkyIsNotGreen/splinter-encoder) |

## Install

```bash
git clone https://github.com/sky-is-green/hive.git
cd hive
python3 -m venv .venv
.venv/bin/python -m pip install -e splinter   # the memory system
.venv/bin/python -m pip install -e .          # the suite and the Studio
```

The suite's grouped runs and the Studio are in
[`hivebench/README.md`](hivebench/README.md).

## The state

The short version, from [`docs/STATUS.md`](docs/STATUS.md): the 35B and 27B
tiers are measured; the 176B's generation is net zero; the memory's dedup
decision is measured, and the drift benchmark is invalid and being rebuilt. The
API tier is measured (28/28 on the 28 tasks, 2.6 s median, free OpenCode), so
is the execution checker (6/7 errors, 4/4 at p >= 0.97, 0/21 false rejects,
0.23 ms median), and so is the scripted controller (28/28, 7 escalations,
0/21 correct sent up, 9.24 s mean / 17.22 s conditional). The grounding path
and the API-driven controller are unmeasured. This is a research harness, not
a product.

## Engine notes

The serving engine, the hardware constraints and the engine options are in
[`docs/ENGINE.md`](docs/ENGINE.md).

## License

MIT. The paper citations are in each paper, and the software citation is in
[`CITATION.cff`](CITATION.cff).
