# Phase 3 encoder — native ONNX vs Python sentence-transformers

**Status:** encoder built, parity-tested, benchmarked, and integrated into the
daemon/sidecar path (2026-11-13). The memory core is still Python.

## What is built

`rust/crates/hive-encoder` loads the cached
`sentence-transformers/paraphrase-MiniLM-L3-v2` ONNX model with `ort` and the
`tokenizer.json` with `tokenizers`. It provides:

```sh
hive-encoder info --json
hive-encoder encode --text "..." --json
hive-encoder encode --file texts.txt --json
hive-encoder bench --file texts.txt --repeat 3 --batch 32 --json
hive-encoder serve --host 127.0.0.1 --port 8172
```

Model discovery order: `HIVE_ENCODER_DIR`, `<repo>/models/encoder`, then the
Hugging Face cache snapshot. No Python, torch, transformers, or
sentence-transformers process is involved.

With `HARNESS_EMBEDDING_BACKEND=rust`, the Rust daemon spawns `hive-encoder
serve`, waits for `/health`, and passes `HARNESS_EMBEDDING_URL` to the Python
sidecar. `sieve.rust_encoder.RustEncoderDrone` batches query+chunks into one
`/encode` call. `tests/conformance/test_memory_parity.py` swaps only the drone
and verifies the assembled context and selected chunk ids are identical to
sentence-transformers.

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

## Native memory slice

`rust/crates/hive-memory` ports the first retention layer:

- chunk storage, MD5 fingerprint dedup, turn indexing, LRU eviction;
- `DecayMatrix` and cosine retrieval over native embeddings;
- hygiene/codec repair (mojibake, boilerplate, secrets, length cap);
- semantic dedup, topic drift, adaptive budget, remembrance, and the comb
  (JSONL surplus store with lexical recall);
- native context assembly (without comb/remembrance wiring yet);
- parity tests for store operations, hygiene/codec, dedup, drift, remembrance,
  and assembly against Python.

Still Python: comb/remembrance integration in the assembler, the P1–P11
replay, and making the native path the default.

## Torch-free install

A fresh venv with only `.[studio]` and `splinter` (no torch/transformers/
sentence-transformers) runs the daemon + native encoder + chat end-to-end;
`test_live_torchfree_sidecar_chat` formalizes it when
`HIVE_TORCHFREE_PYTHON` is set. The venv is **360 MB** versus **1.6 GB** for
the full torch environment; the encoder adds a 27 MB binary plus a 66 MB model.

`hive-encoder fetch --dir models/encoder` downloads the ONNX model and
tokenizer and verifies the model's SHA256 against the Hugging Face LFS oid.

## Still missing for G3

- The remaining memory stages (hygiene/codec, comb, drift, dedup, assembly,
  budget) are still Python.
- A torch-free runtime venv has not been tested (P3-09).
- P1–P11 / 28-task cascade replay has not been run against a native memory core.
- Only MiniLM-L3-v2 is validated.
- The native scorer fixes confidence at 1.0 (no MC-dropout) (P3-05).
