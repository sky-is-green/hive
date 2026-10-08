# Ternary-Bonsai community arms — head-only measurement (2026-10-08)

**Task set:** `tasks-community.json` — GSM8K-300 + MATH-200 + HumanEval-164
(664 tasks, 465 train / 199 test, seed 0, sha `83baf6f3…`), identical to the
Scion community run.
**Model:** `prism-ml/Ternary-Bonsai-2-27B-gguf` (`…PQ2_0.gguf`,
7,206,168,928 B, sha `3907dc16…`), served by the moe-corr fork at **card
sampling** (thinking, temp 1.0, top_p 0.95, top_k 20, min_p 0.05, 4,096 cap).
The published model-card numbers are the baseline reference; no baseline
re-run.
**Protocol:** 3 samples/task on all 664 tasks (1,992 samples); judge-prompt
last-token states via `/embedding`; in-domain heads fitted on the 1,395 train
samples (answer + trace candidates); the frozen J-Bench head
(`bonsai-2-27b-decision-head.json`, dim 5120) applied without refit.
All arm numbers are the held-out **199 test tasks** unless stated.

## Test-split arms

| arm | correct | acc | tok/correct | vs single |
|---|---|---|---|---|
| A single (sample 0) | 168/199 | 0.844 | 1,284 | — |
| C gated, in-domain head (thr 0.3) | 173/199 | 0.869 | 1,797 | +5 at 1.44× |
| D best-of-3, in-domain head | 173/199 | 0.869 | 3,630 | +5 at pool cost |
| D best-of-3, J-Bench head | **176/199** | **0.884** | 3,568 | +8 at pool cost |
| majority of 3 | 168/199 | 0.844 | **3,738** | ±0 at pool cost |
| oracle best-of-3 | 178/199 | 0.894 | — | — |

Full-set (664): single 579/664 (0.872, 1,104 t/c); gated 604 (0.910, 1,634);
best-of-3 in-domain/J-Bench 604 (0.910, 3,121); majority 583 (0.878; honest
pool cost 3,234); oracle 609 (0.917).

**Cost accounting.** Earlier arms tables charged the majority vote only the
picked candidate (1,265 t/c).  A vote needs the whole pool: the honest cost is
the same 628k pool as best-of-3, i.e. **3,738 t/c**.  The gated policy stops
early and pays 1.44× single for best-of-3 accuracy.

Oracle-gap captured (test): single baseline, majority 0%, gated 50%,
J-Bench best-of-3 80%; all-664: gated/best-of-3 25 of 30 tasks (83%).

## Head quality and natural-error catch

| head | test AUC (answer) | first-sample catch @0.5 | keep |
|---|---|---|---|
| in-domain refit (train fit) | 0.884 | 75/85 (88%) | 576/579 (99.5%) |
| frozen J-Bench | 0.860 (all 1,992) | 55/85 (65%) | 577/579 (99.6%) |

In-domain trace-candidate head: test AUC 0.943.  Train AUC is 1.0
(in-sample); only held-out numbers are quoted.

## Veto: majority + head override (2026-10-08)

Scion's veto policy (majority canonical answer; override to the highest-p
sample when it beats the majority base by a margin tuned on train), applied to
Bonsai:

| policy | test | fixes/breaks | exact p |
|---|---|---|---|
| majority | 168 | — | — |
| frozen head, plain argmax (d=0) | **176** | 8/0 | 0.008 |
| frozen head, veto d=0.05 (train-tuned) | 173 | 5/0 | 0.063 |
| raw answer-slot margin, argmax / veto d=0 | 174 | 6/0 | 0.031 |
| clean blend (frozen+margin), best-of-3 | 175 | — | — |

Bonsai is the clean replicate: **zero breaks at every margin**, train curves
prefer a small margin (425 vs 423 train tasks), and every policy beats
majority.  Pooled with Scion (train-tuned margins): 14 fixes / 2 breaks over
398 held-out tasks, exact p=.004.

## Raw answer-slot margin (2026-10-08)

The model's own next-token `p(A)−p(B)` on the same judge prompt
(`analyze_logit_community.py`): test AUC **0.858**, best-of-3 **174/199** — a
strong zero-training baseline, behind the frozen head (0.901 / 176) but well
above chance.  On the Scion community pool the ordering flips (raw 0.880 >
frozen 0.859; blend best-of-3 159 > raw 157 > frozen 156), so neither signal
dominates and per-model validation should pick.  Raw margin is free (same
forward pass as the head) and worth exposing as an auxiliary readout.

## Findings

1. **Gating is the efficiency lever**: best-of-3 accuracy at ~half the pool
   cost (311k vs 628k tokens).
2. **The frozen head transfers to Bonsai's own candidates**: 176/199
   (0.884, 80% of the oracle gap), catch 65% at 99.6% keep.
3. **The veto replicates**: zero breaks, +5/+6 over majority; pooled p=.004.
4. **Raw self-assessment is strong but model-dependent** — a second, free
   readout, not a replacement.
5. **No think-harder arm by design**: Bonsai's serving recipe *is* thinking at
   card sampling; `B_think_hard` is 0 because the run script skips it.  The
   `sampling` metadata block in `bonsai-community-eval.json` shows harness
   defaults (temp 0.7 / non-thinking / 700 cap) and does **not** describe the
   generation run.

## Caveats

- One generator and one task mix; Bonsai ran at card sampling (thinking), so
  cross-model comparisons with Scion's temp-0.7 non-thinking recipe mix model
  and recipe.
- In-domain heads and their thresholds are train-fitted; the frozen head and
  raw margin need no fitting.
- (Metadata) `bonsai-community-eval.json.sampling` — see finding 5.

## Reproduce

`run-community-bonsai.sh` (server → gen → embed → eval).
New arms: `veto_eval.py --prefix bonsai-community --head-json
bonsai-2-27b-decision-head.json`; `analyze_logit_community.py --prefix
bonsai-community --head-json bonsai-2-27b-decision-head.json`.
Raw: `bonsai-community-eval.json`, `veto-bonsai-community.json`,
`logit-bonsai-community.json`; `bonsai-community-samples.jsonl` and
`bonsai-community-states.npz` are gitignored.
