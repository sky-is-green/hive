# Scion-35B-A3B community arms — head vs think-harder vs majority (2026-10-07)

**Task set:** `tasks-community.json` — GSM8K-300 + MATH-200 + HumanEval-164
(664 tasks, 465 train / 199 test, seed 0, sha `83baf6f3…`), built offline from
the cached datasets by `bench_tasks.py --counts`.
**Model:** `SkyIsNotGreen/Scion-35B-A3B` (PQ2_0-corr, 11,336,908,288 B), served
by the moe-corr fork, temp 0.7, non-thinking, 700-token cap (the serving recipe
of the prior sessions).  Arm B additionally enables thinking with a 2,048 cap.
**Protocol:** 3 samples/task on all 664 tasks (1,992 samples) + 664
think-harder samples; judge-prompt last-token states via `/embedding`; the
in-domain head is fitted on the 1,395 train samples in the serving format; the
frozen J-Bench head (`probe-states-scion35b.npz`) is applied without refit.
All arm numbers below are the **held-out 199 test tasks**.

## Test-split arms

| arm | correct | acc | tok/correct | vs single |
|---|---|---|---|---|
| A single (sample 0) | 142/199 | 0.714 | 375 | — |
| B think-harder (thinking, 2,048) | 150/199 | 0.754 | 819 | +8 at 2.2× tokens |
| C gated, in-domain head (thr 0.3) | 152/199 | 0.764 | 500 | +10 at 1.33× |
| C gated, J-Bench head (thr 0.95) | 154/199 | 0.774 | 560 | +12 at 1.49× |
| D best-of-3, in-domain head | 149/199 | 0.749 | 1,072 | +7 at 2.9× |
| **D best-of-3, J-Bench head** | **156/199** | **0.784** | 1,024 | **+14 at 2.7×** |
| majority of 3 | 154/199 | 0.774 | 345 | +12 at 0.92× |
| oracle best-of-3 | 169/199 | 0.849 | — | — |

Oracle-gap captured: majority 44 %, J-Bench best-of-3 52 %, J-Bench gated 44 %,
in-domain gated 37 %.

## Per benchmark (test)

| benchmark | n | A | majority | C in-dom / J-Bench | D in-dom / J-Bench | oracle |
|---|---|---|---|---|---|---|
| GSM8K | 90 | 75 | 81 | 79 / 81 | 79 / 82 | 87 |
| MATH | 60 | 28 | 34 | 33 / 32 | 33 / 34 | 38 |
| HumanEval | 49 | 39 | 39 | 40 / 41 | 37 / 40 | 44 |

Headroom is concentrated in MATH (oracle +10) and GSM8K (+12); HumanEval has
only 5 recoverable tasks and code is the hardest for the chooser.

## Natural-error catch (first samples, test: 57 errors / 142 correct)

| head | catches | keeps | per-sample AUC (all samples) |
|---|---|---|---|
| in-domain refit | 30/57 | 120/142 (22 false rejects) | 0.662 |
| **frozen J-Bench** | 19/57 | **140/142 (2 false rejects)** | **0.859** |

## Findings

1. **Selection pays where errors exist.** Majority voting already adds +12 at
   equal tokens; the J-Bench head adds +14 but at ~2.7× tokens; the gated
   policy buys +12 at ~1.5×. The single-shot arm is 16–17 points above the
   pool oracle gap, so 30–52 % of it is recoverable — far more than on the
   near-ceiling Bonsai bench.
2. **The frozen J-Bench head generalizes better than the in-domain refit.**
   Train AUC is 1.0 and test AUC 0.66 for the in-domain head (it overfits the
   community train mix), while the curated frozen head sits at 0.86 and keeps
   98.6 % of correct answers.  On a diverse multi-benchmark mix, a head trained
   on perturbed/curated negatives beats refitting on the model's own samples.
3. **Think-harder is dominated.** +8 points for 2.2× tokens; the gated head
   gets +10–12 for less added cost, and majority gets +12 for free.
4. **Majority is the baseline to beat.** On this mix the head only beats
   majority at ~3× tokens (D J-Bench 156 vs 154); the honest pitch is
   accuracy-per-token, not raw accuracy.
5. **Calibration matters more than AUC here.** The in-domain head's false
   rejects (22/142) would abort correct answers in a product loop; the frozen
   head's 2/142 is usable.  Gate thresholds must be set on held-out data.

## Caveats

- One generator, one task mix; arm B differs from A in both thinking and cap.
- In-domain thresholds are tuned on train (in-sample for the head), so the
  gated arm's threshold is optimistic; the J-Bench head's threshold is not.
- HumanEval is executed through the repo checkers (canonical solution verified
  to pass, broken to fail).

## Reproduce

`run-community-scion.sh` (server → gen → hard → embed → eval);
`community_eval.py` for the arms, `community_breakdown.py` for this table.
Raw: `scion-community-eval.json`, `scion-community-breakdown.json`,
`scion-community-samples.jsonl` (gitignored), `scion-community-states.npz`
(gitignored).
