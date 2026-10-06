# Prior art: LLMs as decision models / self-verification (scan 2026-10-06)

Scope: before building the "generator verifies itself" experiments, check what
is already established for (a) reading typed decisions out of general LLMs,
(b) self-verification vs an independent verifier, (c) whether model size
matters, and (d) whether readouts survive quantization.  Confidence tags:
[verified] = primary abstract/repo read here; [single-source] = one write-up;
[inferred] = our synthesis.

## 1. Typed decisions from general LLMs — settled, recent

- **[verified] LLM-as-Jev** (Li & Wagle, `arXiv:2610.02076`, 1 Oct 2026):
  architecture-preserving readout of bracketed option identifiers from
  next-token probabilities.  Qwen3.5-4B **without training matches community
  Jev-style models built on the same backbone**; fine-tuning gives targeted
  benefits (weaker models, many-option intent routing) and diminishing returns
  on strong backbones; KL anchors protect conversational generation.
- **[verified] AnyJev** (Nokia, `arXiv:2610.00831`, 30 Sep 2026): L0
  (rotation marginalisation + label-prior correction, zero labels) and L2
  (closed-form LDA/ridge head on a hidden state ~2/3 depth, 100-300 labels per
  question).  Qwen3-1.7B L2 0.730 ~ Jev's published 0.727; Qwen3-4B 0.786 >
  fine-tuned Laya 0.768; zero-label L0 is weak at small scale (1.7B 0.494,
  8B 0.647, 32B 0.700).  Heads do not transfer across questions; labels are
  needed per question.

Consequence: the premise of the Clef-shrinking line (train a separate
decision model on a capable base) is superseded.  The readout is the product.

## 2. Self-verification vs independent verification — the live debate

- **[verified] Kadavath et al. 2022** (`arXiv:2207.05221`): P(True)
  self-evaluation of own samples works, is well calibrated few-shot, and
  improves with model size; models find their own samples more plausible.
  This is the origin of "the ability is latent".
- **[verified] Panickssery et al. 2024** (`arXiv:2404.13076`): self-preference
  and self-recognition bias are real and causally linked; evaluators favour
  their own outputs.
- **[verified] Huang et al. 2023** (`arXiv:2310.01798`): prompted intrinsic
  self-correction of reasoning fails, at times degrading.
- **[verified] Lu et al. 2025/26** (`arXiv:2512.02304`): across 37 models and
  9 benchmarks, **cross-family verification is more effective than
  self-verification; benefits decrease as solver and verifier become more
  similar**; reasoning post-training weakens self-improvement.  This is the
  strongest argument to keep an independent verifier in the cascade.
- **[single-source] Zhang et al. 2025** (`arXiv:2504.05419`): probes on
  reasoning-model hidden states verify answers with high accuracy and
  calibration, and encode future correctness.
- **[single-source] Gnosis** (`arXiv:2512.20578`): ~5M-parameter head on
  frozen internal traces (hidden-state evolution + attention), backbones
  1.7B-20B, outperforms large external judges on accuracy and calibration;
  generalises to partial generations.

Synthesis: *readout-based* self-evaluation (logits, probes) >> prompted
self-critique; independent verifiers still beat self-verification.  The
"generator judges itself for free" hypothesis is the weakest variant.

## 3. Does size matter?

- **[verified]** LLM-as-Jev: 4B is the practical knee; 0.6B needs fine-tuning.
- **[verified]** AnyJev: L2 lifts 1.7B to Jev-level; zero-label L0 needs size.
- **[verified]** Kadavath: P(True) improves with scale.
- **[verified]** Lu et al.: verifier *diversity* matters more than raw size for
  selection gains.
- **[inferred]** For our verifier: a 4B-class base + readout is the expected
  sweet spot; below that, fine-tuning or a task-specific head is required.

## 4. Readouts under quantization — our former blocker, largely answered

- **[verified] Aiersilan 2026** (`arXiv:2606.02628`): on three 7-8B
  instruction-tuned models in **4-bit NF4**, a single *linear* probe on a
  mid-network layer reaches **0.904-1.000 AUROC** on held-out hallucination
  benchmarks; MLP probes rarely beat linear by >0.01 AUROC; peak layers are
  consistent (Qwen2.5-7B blocks 19-25 of 28).  Sampling-based detectors were
  far worse under the paired-label protocol.
- **[single-source] SAPLMA** (Azaria & Mitchell 2023): linear probes on
  hidden states detect statement truthfulness; middle layers best.
- **[single-source] `arXiv:2510.09033`** (counterpoint): hidden states may
  reflect knowledge recall rather than truthfulness; unknown-unknown
  hallucinations overlap with factual outputs.
- **[single-source] `arXiv:2609.31796`**: probes are sensitive to inference
  numerics/precision; fix instrumentation across comparisons.

Consequence for us: the earlier "ternary kills the decision signal"
conclusion was measured through *Clef's frozen final-layer head*.  Mid-layer
linear readouts are documented to survive 4-bit quantization; **2-bit ternary
survival is the genuinely open question**, and our QAT/V2 artifacts are
ready-made subjects.

## 5. What remains for this project

1. **Self vs independent on our task**: cascade verifier with checker gold,
   mixed math/code/QA, on the actual local generator.  Prediction (Lu et al.):
   independent wins; if it does, the "no separate model" claim fails for the
   hallucination lever even though typed decisions need no dedicated model.
2. **Ternary/2-bit readout survival** (novel): linear probes at mid-depth on
   2-bit ternary bodies, fitted on bf16 and refit on ternary.
3. **Footprint fit** on this box (RAM/VRAM floor beside the generator) —
   systems question, not in the literature.
4. **Calibration/threshold** for the cascade's reject/escalate lever.

Test plan (frozen set first): see `PLAN.md` in this directory.
