# Clef-Flash ternary attempt — recipe, measurements, next step (2026-10-03)

Goal: an ultra-lean local validator from Clef-Flash (Qwen3.5-9B + joint schema
head), benchmarked bf16 first and after quantization.

## bf16 baseline (done, CPU, text-only)

`ClefValidator` (harness) scored the bench labels against the checkers:
load ~5 s (warm cache), **~3.2 s/check**, 67/70 correct verdicts on train
(0 false accepts, 3 false rejects) and 26/30 on held-out test (1 FA, 3 FR) —
better than Tiny-Jev and matching the API judge on train.  See
`SUMMARY.md` and `clef-flash-judge-eval-{train,test}.json`.

## The correct Scion recipe (what we got wrong first)

Scion's PQ2_0 (ggml type 142, legacy `Q1_0_g128`) is produced by
`llama-quantize` with **`GGML_PQ2_0_LLOYD=1`** (Lloyd-Max scale rule) plus
**trained rank-512 corrections** merged into the body (`scion` repo:
`scion_moe/quant.py`, `rotation.py`; README: ternary experts, Q8_0 rest,
corrections trained with the deployed quantizer in the loop).  Without the
env var the quantizer uses absmax and the model is destroyed: hidden states
orthogonal (cos 0.007) to the f16/HF reference.  With Lloyd: cos 0.417.

## What was built

- f16 GGUF: fork converter (`--no-mtp`; Qwen3_5 text backbone), 17.9 GB.
- PQ2_0 Lloyd: `GGML_PQ2_0_LLOYD=1 llama-quantize ... PQ2_0`, **3.26 GB**
  (2.90 BPW).  Fork: `/home/penis/llama.cpp` (`moe-corr-runtime`).
- Hidden-state bridge: `clef_embed` (C++ helper, ~90 lines, libllama
  `llama_set_embeddings` + `llama_get_embeddings_ith`, CPU or GPU) feeding
  per-token states to the joint schema head in torch.  Verified exact:
  f16 GGUF embeddings vs HF `last_hidden_state` = **cos 1.0000**.

## Parity result (uncorrected ternary)

10 train records, ternary p vs bf16 p: e.g. 0.431 vs 0.922, 0.484 vs 0.879,
0.565 vs 0.915 — every decision shifted down ~0.35-0.49.  At threshold 0.5
most correct answers would be flagged.  This matches the Scion record:
uncorrected body PPL 566.7 vs 14.48 with corrections; the corrections are the
recipe, not optional.

## Next step

Train clef-flash corrections (output-KD against the bf16 teacher, deployed
Lloyd quantizer in the loop, rank-512 branches; attention-out and MLP-out
placements need a design pass for the dense/hybrid Qwen3.5 stack), then merge
and re-run this parity benchmark.  That is a training run; decide local
7900 XT pair vs a rented card before starting.

Artifacts: `/home/penis/Desktop/work/models/clef-flash-ternary/` (f16, PQ2_0
Lloyd, lm_head, hf-head/), bridge source kept at `/tmp/opencode/clef_embed.cpp`
(port to `tools/clef-bridge/` when the model lands).
