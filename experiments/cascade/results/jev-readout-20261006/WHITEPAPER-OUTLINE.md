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
5. **Self-bias audit** — paired own/other answers, equal labels: +0.078 on
   correct, +0.018 on wrong (no error inflation); controlled attribution test
   pending.
6. **Quantization** — T3 pending: f16 -> ternary (h128 GPTQ, QAT) cross-body
   transfer at the final layer.  Prior art: 4-bit probes keep 0.90-1.00 AUROC
   (Aiersilan 2026); 2-bit is the open point.
7. **Systems** — head = `[hidden, K]` + stats (KB), zero resident model, no
   second ROCm context; fits the MTP/router sidecar pattern; online-monitor
   variant (read states during generation) is future work.

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
  claim online steering gains yet; claim better-than-JEV on JEV's own
  benchmarks (different task family and gold).

## Open items before submission

- T3 result (and refit vs transfer numbers).
- Controlled attribution test.
- 1-2 additional bases for the trained-judge comparison (JEV-9B on J-Bench
  if it fits a card; Intern-4B already there).
- Calibration section (ECE, no-false-accept thresholds are in the table).
- One cascade end-to-end run with a head-backed validator.
- Artifact: `anyjev`-style packaged head + runtime note (fork sidecar).
