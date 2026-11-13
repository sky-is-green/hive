# Phase 3 encoder — native ONNX vs Python sentence-transformers

**Status:** encoder slice built and measured (2026-11-13). Memory core and
runtime integration are not ported yet.

## What is built

`rust/crates/hive-encoder` loads the cached
`sentence-transformers/paraphrase-MiniLM-L3-v2` ONNX model with `ort` and the
`tokenizer.json` with `tokenizers`. It provides:

```sh
hive-encoder info --json
hive-encoder encode --text "..." --json
hive-encoder encode --file texts.txt --json
hive-encoder bench --file texts.txt --repeat 3 --batch 32 --json
```

Model discovery order: `HIVE_ENCODER_DIR`, `<repo>/models/encoder`, then the
Hugging Face cache snapshot. No Python, torch, transformers, or
sentence-transformers process is involved.

## Parity

`tests/conformance/test_encoder_parity.py` compares 1,000 deterministic texts
(natural language, code, math, short/long) plus 20 queries against Python:

- minimum per-text cosine: **> 0.99999**
- maximum absolute element error: **< 1e-4**
- top-5 retrieval ids for every query: **identical**

The ONNX model exposes `sentence_embedding` (mean pooling already applied), so
the native path does not re-pool. Tokenizer padding uses the attention mask;
the earlier all-ones-mask bug was caught by this corpus.

## Benchmark

Release build, AMD Ryzen 7 7800X3D (16 threads), 1,000 short texts, batch 32,
three repeats. Python is `sentence-transformers` 6.1 with torch 2.6 CPU.
Numbers vary between runs; the shape is stable.

| metric | Rust `hive-encoder` | Python `sentence-transformers` |
|---|---:|---:|
| cold process load (`info`) | **0.085 s** | **5.6 s** (import + load); 2.3–2.5 s model load |
| bulk throughput (1k × 3) | 2,266 texts/s | 1,945–2,284 texts/s |
| one-text-at-a-time (200) | **1,081 texts/s** | **490 texts/s** |
| peak RSS | **166 MB** | **636–928 MB** |
| release binary | 27 MB | torch/transformers environment: hundreds of MB |

The runtime turn path encodes the query and chunks separately, so the
one-text-at-a-time result is the relevant one: native is about **2× faster**.
Bulk offline encoding is roughly parity. The native path wins decisively on
startup, RSS, and install footprint.

## Reproduce

```sh
cargo build --release -p hive-encoder
HIVE_ENCODER_BIN=rust/target/release/hive-encoder \
  .venv/bin/python tests/conformance/bench_encoder.py
.venv/bin/python -m pytest tests/conformance/test_encoder_parity.py -q
```

## Recommendation

Use the native encoder for the runtime. Keep Python sentence-transformers as
the bulk/offline reference and for other `ultra_model` choices until each is
exported and added to the parity corpus.

## Still missing for G3

- `ContextStore`, decay, dedup, drift, comb, assembly, and budget selection are
  still Python.
- The sidecar still imports torch; install size is not reduced yet.
- P1–P11 / 28-task cascade replay has not been run against a native memory core.
- Only MiniLM-L3-v2 is validated.
