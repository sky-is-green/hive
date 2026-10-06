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

> The **current objective is Goal B** (a good ternary community quant; metric =
> PPL), so the quantizer table at the end of this stage is the live thread.
> The decision results earlier in the stage remain for the record: the ternary
> validator is not achievable through PTQ, and bf16 Clef keeps that role.

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
deployed body.  A PRF-signed basis (seed 1337, explicit sign metadata) improves
it further: **PPL 476.08** vs 514.35 with identity signs.

**The missing quantizer step (GPTQ) — measured.** V2 is rotation + plain Lloyd
RTN; the TAARDIS 27B reference (13.61 PPL "weights alone") is rotation +
Hessian-weighted GPTQ + flip-polish.  The local pipeline now exists:
`dense/clef_v2_hessians.py` (per-linear ``XᵀX/N`` from the bf16 recurrent-GDN
forward, layer-group passes) and GPTQ wiring in `dense/clef_v2_convert.py`
(`R H Rᵀ` via the forensics' `rotate_hessian`, group 128, PQ2_0 pack).  First
measurements isolate the scale rule from the error compensation:

| quantizer (signed basis, c512, 100 chunks) | PPL |
|---|---|
| RTN + **Lloyd** scales | **476.08** |
| GPTQ + absmean scales | 2460.25 |
| RTN + absmean scales | 15936.78 |
| GPTQ + Lloyd scales | 277.25 |
| GPTQ + Lloyd + act-order, 48-w Hessians | 141.56 |
| GPTQ + Lloyd + act-order, 128-w Hessians | 120.62 |
| **GPTQ + Lloyd + act-order, 128-w Hess + `ffn_down` F16** | **75.20** |
| + flip-polish 4 (same recipe) | 3012.85 — falsified |

The scale rule dominates (33x between Lloyd and absmean RTN at the same
rotation) — consistent with the original Clef finding that the deployed Lloyd
rule is load-bearing.  GPTQ's compensation is real once the rule is right:
277.25 vs 476.08 signed RTN, **act-order brings it to 141.56** (3.4x total),
and **128-window Hessians take it to 120.62** (1.17x more, same 5.52 GiB
container).  Mixed precision stacks on top of that: keeping `ffn_down` F16
lands at **75.20 in 8.13 GiB** (1.60x from 120.62) — the wide, weakly measured
12288-d Hessians stay out of the ternary set.  Flip-polish was then tried and
**falsified**: greedy per-trit descent drops the Hessian proxy 35-95% per
tensor yet PPL explodes **75.20 -> 3012.85** — the per-layer proxy is not
trustworthy beyond GPTQ's constrained point (end-to-end compounding; the
forensics' local-KD warning, F6/F7).  PTQ is exhausted (`--gptq-refine` is a
no-op with Lloyd scales; damping/sign-seed/window stacking are marginal by the
lever ranking).  Next: rotation-in-the-loop KD/QAT (rental), base choice
75.20 mixed vs 120.62 all-ternary.

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
lever).

**The best variant does not move the head.**  `nodown` through the frozen
decision probe: **AUC 0.503** (pos med 0.433, neg med 0.435), best BA 0.550 at
an unusable operating point — exactly chance, like V2 (0.464) and the deployed
body (0.569), against the teacher's 0.844.  Body PPL and head decisions are
decoupled in this regime, so local PTQ/mixed-precision work is exhausted: it
improves fidelity (~2x toward f16) but not decisions.  A usable ternary
validator needs a training run on the decision channel (ranking-aware loss on
the V2/nodown base), or the bf16 Clef validator stays in that role.

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

**Solver derisk (2026-10-05).** On Qwen3.5-0.8B (same hybrid arch),
`scion/dense/qat_derisk.py` (rotation fold + Lloyd STE + f16 hidden-state KD +
Adafactor) recovers a post-hoc rotated ternary body from **PPL 34,938 to 57.1
(612x) in 300 steps / 21 min** (2.1x f16 27.2; peak 7.8 GB VRAM, 4x512 batch).
Longer / mixed follow-ups: all-ternary 600 steps -> **48.4** (1.78x f16);
mixed base (`down_proj` F16) 10,020 -> **48.8 in 300 steps**.  The training
loop for the 9B rental is therefore validated at small scale; both bases
converge to ~48-49, the mixed one ~4x faster.  Note the 0.8B is far more
ternary-fragile than the 9B (isolated damage: self-attn 1.4x, MLP 8.9x,
GDN 27.5x; ~1,200x combined) — a hard, useful stress case.

**9B QAT result (2026-10-05).**  The h128 all-ternary body trained 1200 steps
(rotation fold + Lloyd STE + f16 hidden-state KD, Adafactor; one L40S 48 GB
secure pod, 1.7 h, ~$3.9) goes **120.62 -> 23.25 PPL** (5.2x) in the same
5.52 GiB container (f16 12.59 = 1.85x).  Export gotcha for the record: the
rental trains the reverse-loaded HF layout, so masters must be re-permuted
with the inverse of `clef_dense_load._undo_gdn` (attn_qkv / attn_gate /
ssm_out) before packing — without it PPL explodes (~17k) while the pod-side
hidden KD still looks healthy.  Frozen probe on the QAT body for the record:
**AUC 0.573** (pos med 0.428, neg med 0.421), best BA 0.500 — decisions remain
decoupled; bf16 Clef keeps the validator role.  **Pre-post generation test
caught a blocker:** hidden-KD-only QAT (v1) has degenerate free generation
(loops, wrong math, code garbage) despite PPL 23.25 — f16 control is clean,
teacher-forced NLL on the f16's own text 2.02 vs 1.24.  Fix queued: QAT v2
with chunked logits KD (the forensics' objective).  **Logits-KD derisk
(0.8B):** hidden-only 56.3, hidden+logits 45.5, mixed+logits 40.4 PPL — all
still degenerate in free generation; artifact GGUF hidden cos through the
bridge is 0.838 (f16 0.9999), so the export is faithful and the limit is
training fidelity.  Free-generation samples now gate the release.

**Post-QAT reality check + community landscape (2026-10-06).**  The QAT body
is faithful but its free generation degrades (above); logits-KD and mixed-base
derisks at 0.8B did not fix it at our training scale.  Meanwhile the
community: Clef/Clef-Flash already have many 4-8 bit quants (bartowski +
ggml-org GGUFs, MLX / FP8 / NVFP4 / EXL3 / OpenVINO, W4A16 AutoRound/GPTQ for
both sizes) — **no ternary Clef exists**; the ternary bar is PrismML
**Ternary-Bonsai-8B** (trained low-bit, Q2_0 g128, 2.03 GiB, ~1.44x PPL,
near-base benchmarks) and TAARDIS-27B (13.61); the community formats are
mainline TQ1_0/TQ2_0 plus the fork's **PTQ1_0** (Prism codec, type 143);
evaluation norm is KL/top-1 vs bf16 on ~250k mixed tokens incl. code.  Next
(gating, local, ~1 h): quantize the Clef-Flash f16 to TQ1_0/TQ2_0/PTQ1_0
(+optional imatrix) and compare PPL + generation — the gate for whether the
rotation+GPTQ+QAT pipeline is worth a release or a bigger (Bonsai-style)
training investment.

Handoff for the next session (assets, findings, next-step menu):
`scion/dense/HANDOFF.md`; external prior art (TAARDIS):
`scion/docs/TAARDIS-PRIOR-ART.md`.

## Stage H (cont.): community-standard ternary gate — TQ1_0/TQ2_0/PTQ1_0 vs QAT (2026-10-06)

The HANDOFF item-4 gate: can a plain, community-standard ternary quant of the
Clef-Flash f16 body match the rotation+GPTQ+QAT artifact?  Local, ~1.5 h, no
rental, non-display GPU, all GPU steps serialized.  Harness:
`scion/dense/clef_gate_chain.sh` (imatrix -> quantize -> PPL -> probes) and
`scion/dense/clef_gen_probe.py` (llama-server + `/completion`; probes are the
QAT acceptance suite: prose temp 0.7 x120, code temp 0.7 x120, math temp 0.0
x100, seed 1337 pinned sampler).  Machine record:
`community-gate-20261006.json` next to this file.

| model | size | bpw | wikitext PPL (c512, 100 ch) | free generation (8-gram repeat p/c/m) |
|---|---|---|---|---|
| f16 control | 17.9 GB | 16 | 12.5894 | 0 / 0 / 0 |
| QAT artifact (`v2/clef-flash-v2-qat-a1.gguf`) | 5.52 GiB | ~2.1 | 23.2508 | 0 / 0 / **0.846 (math loop)** |
| TQ1_0 + imatrix | 2.68 GiB | 2.56 | **1,860,827** | token soup (server 500) |
| TQ1_0 no imatrix | 2.68 GiB | 2.56 | 1,860,827 | same |
| TQ2_0 + imatrix | 2.98 GiB | 2.85 | 1,860,827 | token soup |
| PTQ1_0 + imatrix | 2.73 GiB | 2.61 | 2,097,117 | token soup |
| Q2_K + imatrix (non-ternary ref) | 3.56 GiB | 3.41 | **13.0553** | 0 / 0 / 0.298 (template echo) |

**Why the ternary codecs collapse (measured, not inferred).**  The fork's
`quantize_row_tq1_0_ref` is naive absmax: `d = max|w|`, no Lloyd/deadzone
search (`ptq1_0` is the same rule at group 128 and never reads
`GGML_PQ2_0_LLOYD`).  Dequantizing a tensor back with `gguf-py` gives
per-tensor cos 0.69 vs the f16 source (the deployed Lloyd `PQ2_0` sits at
~0.90), and imatrix is a no-op for these codecs: TQ1_0 with and without the
matrix are identical to 4 dp, and TQ1_0/TQ2_0 even carry identical weight
values (same rounding rule, different packing) -> identical PPL.  These are
BitNet-style codecs for weights already trained ternary, not post-hoc quants
of a standard f16 model.  Two independent runtimes (fork 0.2.0-dev and the
Prism 0.5.0-dev build) produce the same token soup from the TQ1_0 file, so
this is codec behavior, not a kernel bug.  The raw samples are in
`v2/gate-gen/gate-{tq1_0,tq2_0,ptq1_0}-prism.txt` (copies under
`gate-samples/` here; the fork's server returns HTTP 500 "output that does not
match the expected Content-only format" on them; the Prism server returns the
raw text).

**What the gate says.**  Our QAT clearly wins against plain ternary — by 4-5
orders of magnitude — but that comparison flatters the pipeline: those codecs
are not meant for this model, and the *non-ternary* reference beats the
artifact outright.  A plain Q2_K (+imatrix, mainline, no fork metadata) is
**13.06 PPL (1.04x f16), clean generation, 3.56 GiB** vs the QAT artifact's
23.25 / degenerate math / 5.52 GiB; it also loads and runs on the Prism build
(3-chunk sanity 15.35).  The real ternary bar remains PrismML's *trained*
Ternary-Bonsai (~1.44x) and TAARDIS (13.61).  The QAT artifact's generation
blocker reproduces in the fixed-seed harness: the math probe answers
"120 km/h" then loops "1. 1. 1..." at greedy temperature.

**Decision: no release yet; not a dead end.**  The pipeline is the only route
to a viable ternary Clef (plain ternary cannot represent this model), but the
current artifact is not postable and the honest competitive target is the
trained-low-bit bar.  Proposed next run (needs approval): one proper QAT —
mixed corpus incl. code/math, 5-10M tokens (vs 1.2M), hidden + top-k logits
KD, ~$8-15 — accepted on (a) clean generation on this probe suite, (b) PPL
<= ~18, (c) KL/top-1 vs bf16 on ~250k mixed tokens (the community eval norm),
before any HF upload.  Alternative if not pursued: park the community-quant
lane (Q2_K-class quants already cover size/value) and keep the ternary
pipeline as research with the sidecar/decision lane on bf16 Clef.

## Stage H (cont. 2): derisk-first gate — negative (2026-10-06)

Before any scaled QAT run, the mixed-corpus + logits-KD objective was tested at
0.8B (Qwen3.5-0.8B, same hybrid arch; `qwen35-0.8b-qat-derisk/run_mixedcorpus`):
1500 steps, batch 4x512, generation samples every 300 steps, same probe suite.
Training corpus from `dense/clef_mixed_corpus.py`: 50/20/30
prose(wikitext-103)/math(hendrycks_math)/code(local permissive trees), 2049
windows = 1.05M tokens; eval unchanged (wikitext-2 test).

| step | 0 | 100 | 300 | 600 | 900 | 1200 | 1400 | 1500 |
|---|---|---|---|---|---|---|---|---|
| PPL | 34,938 | 89.8 | 63.9 | 62.6 | 52.6 | 44.8 | **42.4** | 43.0 |

f16 = 27.24 (best 1.56x).  Free generation at every sample point (8-gram
repeat fractions in `derisk-mixedcorpus-20261006.json`):
- code: nonsense, e.g. `n = n(n + 1)` self-referential loops at 600-1500;
- math: wrong answers (100 / 120 / 15 km/h) and repeated-question loops;
- prose: topical drift and template repetition ("the Roman Empire is the
  largest power in the Roman Empire ...").

**5x the earlier derisk budget bought ~3 PPL points over the 300-step
hidden+logits run (45.5) and is *worse* than the 300-step mixed-base run
(40.4): the curve is flat, and the generation blocker is not a budget problem
at this recipe/scale.**  Decision per the pre-registered gate (the "Red" rule
of the v2 plan): the scaled 9B run is not justified; the dense community-quant
lane is **on hold**.  The v1 9B artifact (23.25 PPL, 5.52 GiB, math-loop
generation) remains research; no release, no HF upload.  If the lane is ever
resumed it needs a different mechanism (much larger Bonsai-style training),
not more steps of this one.
