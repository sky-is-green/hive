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
`TERNARY-NOTES.md` Stages E-F, 2026-10-03/04).  The corrected head's
probabilities collapse into [0.57, 0.60], so at any threshold in [0.40, 0.55]
it accepts every record; on this benchmark (97% of records correct) that
reproduces the always-accept row exactly.  Its AUC is **0.41 train / 0.44
held-out** (chance 0.5); it also accepts 58/58 of its own training negatives,
so the decision train never learned rejection.  The uncorrected PQ2_0 body is
miscalibrated (useful threshold ~0.46) and near chance against real negatives:
AUC **0.539 on the balanced fit set / 0.569 held-out** (an earlier "0.80" was
a 2-negative artifact).  A calibrated uncorrected variant gives bench-test
12/30 (0 FA, 18 FR) and held-out 25/46 (10 FA, 11 FR) — an honest v0, not a
useful general validator.  The f16 teacher itself scores **AUC 0.938 fit /
0.844 on the frozen probe** (34/46 at 0.5), so the probe is valid and the
student gap is real.  Do not use the ternary row as evidence of quality, and
do not post the current corrections.

Mechanically: the ternary plumbing is sound (bridge + head sidecar, bit-exact
against the offline benchmark, ~17.9 s/check on the recurrent GDN CPU prefill);
the correction *training objective* is what failed — the branch residual is
O(hidden norm) (~110 vs ~120 per token) and the pointwise decision KD on hard
negatives taught "accept".  Next: rotation-first V2 (body fidelity), or a
bounded ranking-loss retrain with residual regularization evaluated on AUC/BA.

**Update (V2, `TERNARY-NOTES.md` Stage G):** the rotated-basis re-conversion
lifts body fidelity a lot — wikitext PPL 8684 -> **514** (f16 12.59) and hidden
cos 0.336 -> **0.498** — but it does **not** fix decisions: frozen-probe AUC
**0.464** vs the deployed body's 0.569 (teacher 0.844), with the head's p
compressed into [0.39, 0.54].  V2 is the better base for a training fix; the
decision channel still needs one.

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
