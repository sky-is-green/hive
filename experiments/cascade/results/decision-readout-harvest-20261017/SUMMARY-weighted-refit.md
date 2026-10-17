# Weighted refit — pre-registered organic boost (w3 / w10)

Stage 2 found the refit MLP holding the bench but catching only 4/9 organic
wrongs. The pre-registered follow-up was an **organic-weighted refit**: boost
the 14 train-side harvested wrongs during fitting so they weigh ~25% of the
negative mass at 10× (primary) and ~9% at 3× (sensitivity) instead of ~3%.
No test-set hyperparameter picking. Fits reuse the stage-2 extraction directly
(`--states-name`), so no re-extraction: 838 train / 182 test items, ridge +
Platt + MLP and all per-depth probes refit with sample weights.

## Results — refit variants, bench-test (n=116) vs organic test-66 (9 wrongs)

| run | variant | bench AUC | bench acc (FA/FR) | harvest AUC | harvest acc (FA/FR) | caught test-9 |
|---|---|---|---|---|---|---|
| base | ridge-concat | 0.922 | 91/116 (23/2) | 0.673 | 55/66 (6/5) | 3/9 |
| base | MLP-concat | **0.980** | **110/116** (2/4) | 0.774 | 52/66 (5/9) | 4/9 |
| base | L0.5 | 0.862 | 93/116 (19/4) | **0.823** | 54/66 (3/9) | **6/9** |
| base | L0.75 | 0.891 | 89/116 (24/3) | 0.772 | 54/66 (6/6) | 3/9 |
| base | L1 | 0.875 | 84/116 (29/3) | 0.762 | 54/66 (4/8) | 5/9 |
| w3 | ridge-concat | 0.931 | 92/116 (22/2) | 0.686 | 56/66 (6/4) | 3/9 |
| w3 | MLP-concat | 0.979 | 109/116 (3/4) | 0.776 | 52/66 (5/9) | 4/9 |
| w3 | L0.5 | 0.865 | 93/116 (19/4) | 0.823 | 55/66 (3/8) | **6/9** |
| w3 | L0.75 | 0.890 | 89/116 (24/3) | 0.774 | 55/66 (5/6) | 4/9 |
| w3 | L1 | 0.877 | 85/116 (28/3) | 0.760 | 55/66 (4/7) | 5/9 |
| w10 | ridge-concat | 0.934 | 97/116 (17/2) | 0.684 | 56/66 (6/4) | 3/9 |
| w10 | MLP-concat | 0.977 | 109/116 (3/4) | 0.762 | 52/66 (5/9) | 4/9 |
| w10 | L0.5 | 0.866 | 93/116 (19/4) | 0.819 | 54/66 (3/9) | **6/9** |
| w10 | L0.75 | 0.890 | 93/116 (20/3) | 0.776 | 54/66 (5/7) | 4/9 |
| w10 | L1 | 0.877 | 87/116 (26/3) | 0.760 | 55/66 (4/7) | 5/9 |

OR-ensemble `L0.5 + MLP + L1` on the organic test-66: base 7/9 caught at 20/57
corrects rejected; w3 7/9 at 18/57; w10 7/9 at 19/57.

## Reading

- **The pre-registered intervention is a null.** The organic frontier does not
  move: refit MLP stays at 4/9 under both boosts (AUC 0.774 → 0.776 → 0.762),
  the L0.5 probe stays at 6/9 (0.823 → 0.823 → 0.819), and the ensemble stays
  at 7/9. No monotone trend with boost strength.
- Bench is held (MLP 109/116 at both boosts vs 110 base) — the "don't lose the
  bench" condition passes, so the null is not a guardrail artifact.
- The only real shifts are on the bench-side ridge boundary (91 → 97 corrects,
  FA 23 → 17 at w10) and the L0.75 probe picking up `math-0020:hs2`
  (3 → 4/9). Weighting moves *where the head cuts*, not what the features see.
- Why: the 3072-dim decision states already separate all 838 training items
  perfectly; upweighting 14 negatives by 10× changes ~140 of ~990 negative
  weight, but cannot create a direction that encodes a fluent arithmetic slip
  the backbone states do not represent. The same two items
  (`math-0013:hs1`, `humaneval-0001:hs1`) evade every run and every variant.
- Consequence for the next move: the gap is **representational**, not a class-
  weighting problem. Remaining levers, in order: (1) LoRA the 0.8B backbone on
  the decision skeleton so features move; (2) targeted harvest to grow the
  organic-negative mass on the error-prone families (gsm8k/math/humaneval
  non-greedy tails); (3) operating-point/ensemble engineering if escalation is
  cheap enough (L0.5 gate: 6/9 at 16% correct-rejection; 3-probe union: 7/9 at
  ~32%).

## Artifacts

- `readout-head-intern-decision-0.8b-harvest-w3.npz`, `-w10.npz` (gitignored),
  `readout-intern-decision-0.8b-harvest-w3.json`, `-w10.json` — weighted fits
  and per-source metrics.
- `weighted-refit.json` — base/w3/w10 variants: bench + harvest metrics,
  test-9 catch lists, per-item p, ensemble numbers.
- `fit-w3.log`, `fit-w10.log` — fit shape (on disk only).

`decision_readout.py` gained `--states-name` (refit an existing extraction) and
`--organic-boost` (sample-weighted ridge/Platt/MLP; weight recorded in the fit
JSON under `training_weights`).
