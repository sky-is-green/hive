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

## Stage A: reverse loader + runtime-kernel match (2026-10-03)

No HF backbone on disk, so the in-loop torch student is reverse-mapped from the
deployed GGUF (`scion/dense/clef_dense_load.py`; the dense route is its own
sibling to `scion/moe/`, not part of the MoE harness).

- f16 GGUF -> `Qwen3_5TextModel`: `missing=0 unexpected=0`; the inverted
  converter is validated.
- PQ2_0 dequant is **byte-identical** to the fork's `dequantize_row_pq2_0`
  (called via a small C harness; max abs diff 0.0). Embedding Q4_K and output
  Q6_K dequant also match `gguf.quants.dequantize`.
- The deployed forward uses the **recurrent** GDN prefill; transformers defaults
  to the **chunked** one. Chunked decays vs the CPU bridge (f16 per-token cos
  1.0 -> 0.82 at token 39); recurrent gives f16 f32 **cos 0.9975** vs the CPU
  bridge. The training forward must patch `chunk_gated_delta_rule` ->
  `torch_recurrent_gated_delta_rule`.
- Bridge note: `n_gpu_layers=0` still built a ROCm context in one run; force
  CPU with `HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""` when co-running
  with torch.

**Incident (2026-10-03):** the first full 9B validation load used a naive
state-dict builder with `dtype=float32` and OOM'd the 30 GB host (a 36 GB fp32
state dict, plus the desktop). Mitigation: a streamed loader
(`load_text_model_streamed`) builds bf16 and adopts one tensor at a time -- no
full state dict, never fp32 for a 9B model -- and every heavy stage is gated on
a measured host/VRAM peak.

**Do not train this 9B in bf16 torch on this box.** Two identical chunked
evaluations of the same PQ2 body produced *different* non-finite record sets
(23 then 50 of 70): the ROCm/bf16 forward is nondeterministically non-finite.
f32 fixes it (31.8 GB body, does not fit the 21.5 GB card), so training ran on
a rental.

## Stage B: rental correction pilot (2026-10-03)

One A40 48 GB (RunPod, ~$0.6, deleted after), f32 body + recurrent GDN + rank-512
branches on the attention and MLP outputs + decision KD, one epoch (78 steps).
Artifacts: `models/clef-flash-ternary/corrections/pilot-rental/` (checkpoint +
`eval-{base,corr}.json`).  Decision parity under the f32-recurrent **torch
proxy** (the PQ2 f32 torch forward is cos 0.948 vs the CPU bridge, so this is a
proxy, not the ship number):

| split | uncorrected | corrected | bf16 Clef | Tiny-Jev |
|---|---|---|---|---|
| train correct | 26/70 | **68/70** (2 FA, 0 FR) | 67/70 | 62/70 |
| test correct | 7/30 | **29/30** (1 FA, 0 FR) | 26/30 | 25/30 |
| hidden cos | 0.23 | 0.37 | 1.0 | — |

**Paused before packaging.** The shippable number needs the branches packaged
(`attn_out` as a LoRA on `attn_output`/`ssm_out`; `mlp_out` needs a dense
`qwen35` `ffn_out` fork hook) and the benchmark re-run through the CPU bridge.
Hidden cos 0.37 leaves headroom for a longer f32 run.
