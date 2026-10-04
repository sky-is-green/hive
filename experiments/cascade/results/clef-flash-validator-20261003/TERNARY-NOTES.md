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

## Stage C: packaged + honest CPU-bridge benchmark (2026-10-03)

Both taps packaged (`attn_out` as LoRAs on `attn_output`/`ssm_out`; `mlp_out`
via the new dense `qwen35` `ffn_out` virtual target in the fork), merged into
the body (`adapter.embedded=true`, 3.10 GiB), and benchmarked through the
llama.cpp CPU bridge + joint head sidecar over all 100 bench records.  **The
honest numbers match the proxy, with zero verdict flips (max |Δp| 0.027):**

| split | uncorrected bridge | **corrected bridge** | corrected proxy | bf16 Clef | Tiny-Jev |
|---|---|---|---|---|---|
| train | 18/70 (0 FA, 52 FR) | **68/70 (2 FA, 0 FR)** | 68/70 | 67/70 | 62/70 |
| test | 8/30 (0 FA, 22 FR) | **29/30 (1 FA, 0 FR)** | 29/30 | 26/30 | 25/30 |
| mean p train/test | 0.471 / 0.471 | 0.586 / 0.588 | 0.595 / 0.594 | 0.811 / 0.792 | — |
| hidden cos train/test | 0.231 / 0.235 | 0.371 / 0.382 | 0.372 / 0.383 | 1.0 | — |

CPU latency ~49 ms/token (recurrent GDN prefill; 13.8-28.6 s/record plus
40-90 ms head) — ~5-8x the bf16 transformer validator's ~3.2 s/check.  The
torch-vs-bridge body cos gap (0.948) does not reach the decision metric.
Artifacts: `models/clef-flash-ternary/corrections/packaged/` (adapter, merged
release, `bridge-{uncorrected,corrected}.json`).  Scripts:
`scion/dense/clef_export.py`, `scion/dense/clef_bridge_eval.py`; bridge batch
mode + embedded-adapter attach in `tools/clef-bridge/`.

## Stage D: harness backend + label benchmark (2026-10-03)

`clef-ternary` is now a first-class validator backend:
`harness/cascade/validator.py::ClefTernaryValidator` keeps the packaged body in
a resident `clef_embed --serve` process (fresh context per check) and runs the
joint head in torch fp32 on CPU; `load_validator` selects it for ids containing
`ternary`, and `judge_eval.py` has the matching `--family ternary`.  The same
`judge_eval` pipeline that scored bf16 Clef re-run over the train/test reports:

| split | ternary harness | bf16 Clef (recorded) | API judge | mean ms/check |
|---|---|---|---|---|
| train | **68/70** (2 FA, 0 FR) | 67/70 (0 FA, 3 FR) | 67/70 (2 FA, 1 FR) | ~17,900 |
| test | **29/30** (1 FA, 0 FR) | 26/30 (1 FA, 3 FR) | 28/30 (1 FA, 1 FR) | ~17,700 |

Bit-for-bit with the offline bridge benchmark (max |Δp| 5e-5); both models now
live side by side in `clef-flash-judge-eval-{train,test}.json`.

**Calibration caveat:** corrected probabilities are compressed into
[0.57, 0.60] and the accept cliff is at 0.60 (one record per split clears it).
Threshold 0.5 is safe (no false rejects anywhere), but the model carries no
graded confidence; recalibrate before using another operating point.  Latency
is the recurrent GDN CPU prefill (~17.9 s/check); the HIP bridge on the
non-display card is the fallback if that binds.

## Stage E: held-out generalization checkpoint — the parity number is majority-class (2026-10-03)

Before any HF release, the packaged artifact was scored on records **outside**
the cascade-bench-v1 distribution: 27 unseen correct answers from the
easy+hard task sets (reasoning 12, code 8, qa 7) plus 19 constructed negatives
(same last-number perturbation rule as training, all verified wrong by the
checker), run through the same harness backend; the uncorrected PQ2_0 body was
run identically for contrast.

| body | operating point | bench-train | bench-test | held-out 46 | AUC train / test / held-out |
|---|---|---|---|---|---|
| uncorrected PQ2_0 | thr 0.46 (train-fit) | 43/70 (0 FA, 27 FR) | 22/30 (0 FA, 8 FR) | 24/46 (13 FA, 9 FR) | 0.801 / 0.759 / 0.569 |
| **corrected (shipped candidate)** | thr 0.40-0.55 | 68/70 (2 FA, 0 FR) | 29/30 (1 FA, 0 FR) | 27/46 (19 FA, 0 FR) | **0.412** / 0.759* / **0.435** |
| always-accept baseline | — | 68/70 | 29/30 | 27/46 | 0.500 |

\* bench-test has a single negative record, so its AUC is not meaningful.

**What this means.** The corrected model's headline 68/70 and 29/30 is exactly
the always-accept baseline on a 97%-positive bench; its within-split AUC is
**0.41 (worse than chance)** on train and 0.44 held-out — the corrections
flatten the ranking into a task-agnostic band around p≈0.58.  The uncorrected
PQ2_0 body actually *retains* ranking signal (AUC 0.80 train, 0.57 held-out)
but is miscalibrated (its useful threshold is ~0.46, not 0.5).  So: the
corrections did not teach rejection, they learned "accept" and destroyed
ranking; the earlier "above bf16 Clef" framing is not supported.  The bf16
validator (67/70 with 0 FA/3 FR) discriminates; the corrected ternary does
not.

**Protocol fix (apply from now on):** verdict accuracy on an imbalanced set is
not evidence.  Report AUC / TPR at fixed FPR / balanced accuracy, and always
include the always-accept baseline.  Keep this 46-record probe frozen; the
next iteration needs a fresh holdout.

**Consequences.** Do not post the current corrections.  The cheap baseline to
build on is the uncorrected body with a calibrated head (ranking ceiling
unchanged); a new correction run needs a ranking-aware decision loss, class
balance, and residual-magnitude regularization (the branch delta is O(hidden
norm): ~110 vs ~120 per token, which is why the fine structure drowns), or the
rotation-first V2 conversion.  Latency is the recurrent GDN CPU prefill
(~17.9 s/check); the HIP bridge on the non-display card is the fallback if
that binds.

## Stage F: calibration baseline + diagnostics — what failed and why (2026-10-04)

Two free follow-ups after the checkpoint.

**Calibration baseline** (`scion/dense/clef_calibrate.py`, uncorrected body):
fit set = bench-train (68 pos) + 55 usable training negatives (of 58; 5
perturbations stayed checker-correct).  Teacher (f16, cached option logits) on
that fit set: **AUC 0.939** (finite entries); the uncorrected student:
**AUC 0.539**.  Fitted Platt slope a=0.549, raw boundary t=0.48:

| set | raw@0.5 | calibrated@0.5 |
|---|---|---|
| bench-train | 18/70 (0 FA, 52 FR) | 37/70 (0 FA, 33 FR) |
| bench-test | 8/30 (0 FA, 22 FR) | 12/30 (0 FA, 18 FR) |
| held-out 46 | 24/46 (6 FA, 16 FR, BA 0.546) | 25/46 (10 FA, 11 FR, BA 0.533) |

So calibration repairs the in-distribution shift (and the earlier "uncorrected
AUC 0.80" was a 2-negative artifact — against real negatives the body is near
chance: 0.539 fit / 0.569 held-out).  It is an honest v0, not a useful general
validator.

**Why the corrections failed** (`scion/dense/clef_probe_diag.py`):

| model | fit-neg AUC | held-out AUC | held-out @0.5 |
|---|---|---|---|
| f16 teacher (GGUF bridge) | 0.939 | **0.844** | 34/46 (2 FA, 10 FR), BA 0.762 |
| uncorrected PQ2 + calibration | 0.539 | 0.569 | 25/46 (10 FA, 11 FR), BA 0.533 |
| corrected merged (shipped candidate) | **accepts 58/58** | 0.435 | 27/46 (19 FA, 0 FR) = always-accept |

The corrected model accepts **every one of the 58 training negatives**
(p 0.574-0.605) while the teacher rejects 50/58 — the failure is not
overfitting: the decision KD never learned rejection on any data; it is a
constant-accept shift.  Mechanism: hidden fidelity is 0.23-0.37 cos and the
branch residual is O(hidden norm) (~110 vs ~120 per token), so the frozen head
is reading a signal that was already weak and is then swamped; the pointwise
KD on hard negatives collapsed to the prior.

**Conclusion.**  The sidecar recipe as trained cannot produce a general
validator, and its failure is at the objective/body-fidelity level, not the
plumbing.  The evidence-backed route to a genuinely useful ternary Clef is the
rotation-first V2 conversion (same-arch Qwen3.5-0.8B precedent at k1/k2),
which fixes body fidelity instead of patching the head.  A sidecar retry would
need (a) a ranking/pairwise decision loss with class balance, (b) residual
regularization, (c) acceptance on AUC / balanced accuracy, not verdict counts
— a bounded rental pilot (~1-1.5 h, ~$1) can test it, but only against the
frozen probe and only after V2 is derisked on a smaller model.

## Stage G: V2 rotated-basis conversion — fidelity wins, decisions don't (2026-10-04)

The rotation-first route reached its first artifact:
`clef-flash-v2-pq2_0-rot.gguf` (5.52 GiB, 427 tensors) is a re-conversion of
the **f16** body with `W' = W·Rᵀ` folded into 176 attention/MLP linears
(block-Hadamard 1024, identity signs), Lloyd-g128 ternarization, `ssm_out`
unrotated-ternary, everything else copied, and `prism.hadamard.*` metadata so
the fork applies the activation transform.  Converter/self-test:
`scion/dense/clef_v2_convert.py`.

**Body fidelity (the rotation is real):**

| metric | f16 | deployed PQ2_0 (unrotated) | **V2 rotated** | V2 no-rotation control |
|---|---|---|---|---|
| wikitext-2 PPL (c512, 100 chunks, GPU) | **12.59** | 8684.23 | **514.35** | 9640.77 |
| hidden cos vs f16 (48 tokens) | 1.0 | 0.336 | **0.498** | 0.311 |

The **no-rotation control** (identical tensor selection/exemptions, folding and
metadata disabled) is statistically identical to the deployed body (9516 GPU /
9640 CPU vs 8684), so the entire gain is the rotation — the GDN exemptions
contribute nothing on their own.  V2 already exceeds the *trained-corrections*
proxy hidden fidelity (0.372) with no training, and improves PPL 16.9x over the
deployed body.

**Mixed-precision sweep** (`scion/dense/clef_v2_sweep_gpu.sh`, 100 chunks, GPU,
same settings; kept-F16 targets are unrotated and absent from the metadata):

| variant (kept F16) | PPL | vs V2 |
|---|---|---|
| f16 | 12.59 | — |
| V2 all-ternary | **514.35** | 1.00x |
| no-rotation control | 9516.34 | 18.5x worse |
| nodown (`ffn_down`) | **265.13** | 1.94x better |
| noqkv (`attn_qkv`) | 280.41 | 1.83x better |
| noedge (layers 0/1/30/31) | 272.60 | 1.89x better |
| nomlp (all FFN) | 846.36 | 1.65x worse |

**Non-monotone:** keeping *all* FFN exact is worse than quantizing it, so the
residual damage is not additive — ternary errors partially cancel across the
stack.  The best single intervention recovers only ~2x; the 514 -> 12.6 gap is
the PTQ ceiling the forensics predicted (trained placement / KD is the missing
lever).  The best variant (`nodown`) is run through the frozen decision probe
to test whether body fidelity moves the head at all.

**Decisions (frozen probe + bench, V2 body, frozen head):**

| set | raw @0.5 | best/other | AUC |
|---|---|---|---|
| held-out 46 (27 pos + 19 neg) | 20/46 (2 FA, 24 FR) | BA 0.519 @ thr 0.45 (TP 11/27, TN 12/19) | **0.464** |
| bench-train 70 | 14/70 (0 FA, 56 FR) | accept-all 68/70; 65/70 @0.40 (2 FA, 3 FR) | 0.404* |
| bench-test 30 | 7/30 (0 FA, 23 FR) | accept-all 29/30; 27/30 @0.40 (1 FA, 2 FR) | 0.793* |

\* 2 / 1 negatives respectively: not meaningful estimates.

**V2 does not fix decisions.**  Probe AUC 0.464 (within noise of chance) is not
better than the deployed unrotated body (0.569), and the head's p values
compress into ~[0.39, 0.54].  The teacher is 0.844 on the same probe, so the
signal exists; the quantized body's states through the frozen head still don't
carry it.  A usable ternary validator therefore still needs **training** — and
V2 is the better platform for it (better hidden fidelity, and the fork supports
rotated-basis adapters).  Two local follow-ups remain cheap: a controlled
no-rotation variant to attribute the fidelity gain, and GPTQ/Hessian placement
to lift PPL further.

**Fork bug found and fixed:** rotated models segfaulted the CPU path whenever
repacking was active (rotation/sign tensors allocated on a CPU repack buft).
`llama-model.cpp` now falls back to the plain CPU buft (mirroring the adapter
loader); verified on V2 (cos unchanged, speed restored).  The bridge also
exposes `CLEF_EMBED_NO_REPACK` for older builds.  The V2 decision battery ran
under the no-repack workaround (~70 s/check); with the fix it is back to
~18-20 s/check.

Handoff for the next session (assets, findings, next-step menu):
`scion/dense/HANDOFF.md`; external prior art (TAARDIS):
`scion/docs/TAARDIS-PRIOR-ART.md`.
