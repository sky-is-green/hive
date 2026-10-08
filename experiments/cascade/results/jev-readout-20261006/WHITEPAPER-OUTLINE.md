# Whitepaper outline — decision heads, not decision models

Status: draft outline, 2026-10-06.  Evidence lives in this directory
(`COMPARISON.md`, `source-audit.json`, `PLAN.md`, `PRIOR-ART.md`).

## Thesis

A verifier decision ("is this candidate answer correct?") is already encoded
in a language model's hidden states.  A few-kilobyte linear head reads it out
at near-perfect accuracy; asking the same model in text does not.  The
capability does not require a purpose-trained decision model: fitted heads on
0.8B-35B bases match or beat dedicated judges (Intern-Decision, Tiny-Jev,
JEV-9B-class) at ~1e-5 of the memory footprint, and the head can be audited for
self-bias because it has no behavioural channel.

## Evidence

1. **Frozen protocol** — J-Bench: 378 items (97 pos / 281 checker-verified
   negatives; 68/164 train, 29/87 test), perturbations *and* 28 natural
   generator errors, all splits fixed by sha256.  (Earlier work read decisions
   only through Clef's frozen head; this set fixes the readout as a variable.)
2. **Readout class dominates** (frozen test, AUC / best BA):
   - base 4B prompted: 0.494 (83% option-order flips -> label bias)
   - base 4B probe: 0.960 / 0.937
   - generator prompted on foreign answers: 0.903; on **its own** answers:
     0.559 (accepts 21/28 of its own errors)
   - generator probe: **0.999 / 0.994**; frozen head on natural errors: 0.971
3. **Scaling and labels** — probes: 0.8B 0.924, 1.7B **0.988**, 4B 0.960,
   35B 0.999.  Trained comparators: Intern-0.8B 0.774, Intern-2B 0.932,
   Tiny-Jev-1.7B 0.890, Intern-4B 0.979.  Label budgets: 4B saturates at 20
   labels (0.959); 0.8B scales 0.784 -> 0.831 -> 0.924 at 20/100/262.
4. **Generalization** — cross-type transfer (fit on number-flips -> boxed
   0.998, code 0.874); frozen head catches 28 natural errors (0.971).
5. **Self-bias audit** — uncontrolled source swap +0.078/+0.018 (style, not
   authorship).  Controlled attribution (identical text, own vs other, n=200)
   is **format-locked**: an out-of-format head shows +0.178/+0.098, but
   in-format refits are attribution-neutral (baseline -0.003/-0.013, balanced
   +0.007/-0.013 over 60 held-out candidates).  The prompted judge shows a
   real **+0.099** on its own wrong answers.  The MTP-head claim: fit the head
   on the serving prompt and it is attribution-neutral (caveat: 9 held-out
   negatives; more data for the wrong-answer claim).
6. **Quantization** — T3 measured on the Clef bodies (same prompts, final
   layer): f16 refit 0.952; refit on 2-bit ternary: QAT 0.712, h128 GPTQ
   0.648 (tracks PPL); **frozen f16 head does not transfer** (0.502 / 0.510,
   chance).  Contrast with prior art: 4-bit probes keep 0.90-1.00 AUROC;
   2-bit is where readout fidelity breaks materially, and heads are
   per-body (consistent with AnyJev heads not transferring across models).
   Mid-layer extraction is the open follow-up.
   **Update 2026-10-07 (Bonsai dense):** a trained 2-bit ternary body keeps
   the readout (J-Bench 0.988; Scion MoE 0.999), so the 2-bit break was the
   post-hoc quantisation regime, not the bit-width or dense-vs-MoE.  Heads
   remain per-body and per-format.
7. **Systems** — head = `[hidden, K]` + stats (KB), zero resident model, no
   second ROCm context; engine-loaded sidecar parity measured at Δ≈1e-8
   (`SIDECAR-PROTOTYPE.md`).  Online monitoring is measured (2026-10-08): the
   trajectory signal is real in generation context but early cancellation is
   rejected (oracle ceiling 21–34% savings; realistic 10–19% at 3–15% false
   cancels) — the monitor is an end-of-trace verifier, not an early-exit.

## Related work to position against

AnyJev (2610.00831), LLM-as-Jev (2610.02076), Kadavath 2022, Gnosis
(2512.20578), Zhang 2025 (2504.05419), SAPLMA (2023), Aiersilan 2026
(2606.02628), Lu 2025 (2512.02304, cross-family), Panickssery 2024
(self-preference), Huang 2023 (self-correction), "Do LLMs really know"
(2510.09033).

## Claims we can and cannot make

- Can: readout >> prompted; cheap heads match trained judges on this verifier
  task; natural-error transfer; audited source effects; tiny footprint; the
  ternary survival data (if positive).
- Cannot: state a general law for all tasks; claim cross-question transfer;
  claim online early-exit gains (measured negative); claim better-than-JEV on
  JEV's own benchmarks (different task family and gold).

## Update 2026-10-08 (session close)

- **Raw answer-slot distribution vs fitted head.**  The same judge prompt's
  next-token `p(A)−p(B)` is a strong zero-training readout (test AUC .880
  Scion / .858 Bonsai).  The fitted head is not universally better (frozen
  .901 on Bonsai, .859 on Scion; blends best on Scion only).  The "prompted
  does not work" claim concerns *generated text*; the distribution readout is
  a third channel and should be reported as a baseline.
- **Accuracy over baselines.**  Veto (majority + head override): +7/199 Scion,
  +5/199 Bonsai (0 breaks), pooled exact p=.004 at zero extra generation.
  Gating reaches best-of-3 accuracy at ~half the pool cost.  No marked gains
  beyond this.
- **Early-exit and trajectory selection ruled out** (see §7 above and
  `MONITOR-FIX.md`).
- **Fine-tuned decision models are not competitive here** (external evidence:
  LiquidAI d1-3B regressed against its base on ImajevBench, 64.0 vs 66.8).
- Final scorecard: `CONCLUSION.md`.

## Open items before submission

- T3 result (and refit vs transfer numbers).
- Controlled attribution test.
- 1-2 additional bases for the trained-judge comparison (JEV-9B on J-Bench
  if it fits a card; Intern-4B already there).
- Calibration section (ECE, no-false-accept thresholds are in the table).
- One cascade end-to-end run with a head-backed validator.
- Artifact: `anyjev`-style packaged head + runtime note (fork sidecar).
