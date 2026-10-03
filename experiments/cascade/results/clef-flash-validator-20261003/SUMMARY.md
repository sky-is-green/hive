# Local validator comparison — Clef-Flash vs Tiny-Jev (2026-10-03)

Both scored offline against the programmatic checker labels of the held-out
`cascade-bench-v1` reports, on **CPU**, at the shared accept threshold 0.5
(the no-leak point for Clef on train).  A "correct verdict" accepts a correct
answer or rejects a wrong one; false accepts are the quality leak, false
rejects are wasted escalations.

| validator | train 70 | test 30 | false accepts (train/test) | false rejects (train/test) | CPU ms/check |
|---|---|---|---:|---:|---:|
| Tiny-Jev-1.7B (bf16) | 62/70 | 25/30 | 0 / 1 | 8 / 4 | ~960 |
| **Clef-Flash-9B (bf16, text-only)** | **67/70** | **26/30** | **0 / 1** | **3 / 3** | ~3,200 |
| frontier API judge (recorded) | 67/70 | 28/30 | 2 / 1 | 1 / 1 | paid, ~1,500 |
| Clef-Flash-9B ternary r512 (deployed) † | 68/70 | 29/30 | 2 / 1 | 0 / 0 | ~17,900 |
| always-accept baseline | 68/70 | 29/30 | 2 / 1 | 0 / 0 | — |

Clef-Flash is the better **local** validator: it matches the API judge's train
verdicts (67/70) with zero false accepts and no API spend, and beats Tiny-Jev
on both splits.  On the held-out test the API judge still leads (28 vs 26);
the one wrong answer there is not separated until Clef's 0.80 threshold,
which then rejects 6 correct answers — so the honest operating point remains
0.5 with 1 leak on test.

† **The ternary row is the always-accept baseline, not discrimination** (see
`TERNARY-NOTES.md` Stage E, 2026-10-03).  The corrected head's probabilities
collapse into [0.57, 0.60], so at any threshold in [0.40, 0.55] it accepts
every record; on this benchmark (97% of records correct) that reproduces the
always-accept row exactly.  Its AUC is **0.41 train / 0.44 held-out** (chance
is 0.5), while the uncorrected PQ2_0 body — miscalibrated, useful threshold
~0.46 — retains AUC **0.80 train / 0.57 held-out**.  The 46-record held-out
probe (27 unseen correct + 19 constructed negatives) confirms it: corrected
27/46 with 19 false accepts, uncorrected 24/46 with 13 FA / 9 FR, bf16 Clef
and Tiny-Jev remain the only validated discriminators here.  Do not use the
ternary row as evidence of quality, and do not post the current corrections.

Mechanically: the ternary plumbing is sound (bridge + head sidecar, bit-exact
against the offline benchmark, ~17.9 s/check on the recurrent GDN CPU prefill);
the correction *training objective* is what failed — the branch residual is
O(hidden norm) (~110 vs ~120 per token) and the pointwise decision KD on a
97%-positive set taught "accept".  A retrain needs a ranking-aware loss,
class balance and residual regularization — or the rotation-first V2 route.

Artifacts: `clef-flash-judge-eval-{train,test}.json` (per-task p_correct and
the full sweep, now carrying both the bf16 `clef-flash` and the
`clef-flash-ternary` entries).  Load: ~5 s warm page cache (bf16 sidecar) /
~12 s (ternary, bridge + head); the sidecar adapters use the text-only path
(backbone + joint head + tokenizer), avoiding the released processor's
torchvision dependency.

## Why this matters for the ternary plan

bf16 Clef-Flash is 18.8 GB and ~3.2 s/check on CPU — usable as the harness
validator when the engine is stopped, but not co-resident-friendly.  The
ternary *plumbing* (PQ2_0 body 3.3 GB + head sidecar, no torch on the 9B, no
GPU) is the co-resident lane at ~17.9 s/check, but its current corrections do
not discriminate; the uncorrected body retains the ranking and is the honest
base to build on (calibration first, then a ranking-aware retrain or V2).  If
latency becomes binding, the same bridge builds for HIP and can use the
non-display card (one heavy process policy).
