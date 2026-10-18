# Open-ended held-out test: leave-one-family-out on truthfulqa

The frozen organic-66 contains no open-ended wrongs (five truthfulqa items, all
correct), so the gate's open-ended performance was unmeasured. This fold removes
**all truthfulqa tasks** (bench + harvest, train + test) from the r2 states —
1,186 of 1,315 items remain — refits the same heads, and applies them to the 129
held-out truthfulqa items the heads have never seen:

- 10 organic wrongs (8 harvest, 2 recorded-wrong)
- 75 perturbed wrongs (harvest + bench choice-flips)
- 44 corrects

The in-family r2 heads (trained *with* truthfulqa) are shown as the contaminated
upper bound.

## Probability-threshold result (@0.5)

| head | in-family organic | in-family FR | LOFO organic | LOFO FR | LOFO AUC (organic) |
|---|---|---|---|---|---|
| ridge-concat | 10/10 | 2/44 | 10/10 | 24/44 | 0.730 |
| MLP-concat | 10/10 | 0/44 | 4/10 | 18/44 | 0.516 |
| L0.5 | 10/10 | 2/44 | 10/10 | 28/44 | **0.809** |
| L0.75 | 10/10 | 0/44 | 0/10 | 0/44 | 0.466 |
| L1 | 10/10 | 2/44 | 0/10 | 0/44 | 0.434 |
| raw readout | 7/10 | 31/44 | 7/10 | 31/44 | 0.390 |

(The perturbed-wrong column is in `lofo-truthfulqa.json`; LOFO heads catch
51–58/75 at the same thresholds, also with heavy false rejection.)

## Calibration-free view: fixed escalation budget

Escalate the lowest-scoring k% of the 54 open-ended items (10 wrong + 44
correct), i.e. a rank threshold instead of a probability threshold:

| head (LOFO) | k=10% | k=20% | k=30% |
|---|---|---|---|
| ridge-concat | 2/10, 3/44 FR | 3/10, 8/44 | 5/10, 11/44 |
| MLP-concat | 0/10, 5/44 | 0/10, 11/44 | 3/10, 13/44 |
| L0.5 | 2/10, 3/44 | **5/10, 6/44** | **7/10, 9/44** |
| L0.75 | 1/10, 4/44 | 2/10, 9/44 | 3/10, 13/44 |
| L1 | 0/10, 5/44 | 1/10, 10/44 | 2/10, 14/44 |

In-family, for contrast: L0.5 at k=20% catches 9/10 at 2/44 FR; MLP at k=20%
catches 10/10 at 1/44 FR.

## Reading

- **The in-family numbers are family memorization, not open-ended skill.** Every
  in-family head catches 10/10 on truthfulqa; the same heads without that family
  in training lose most of it. The 0.844/0.801 organic-AUC numbers on the
  three checkable families say nothing about an unseen open-ended family.
- **The L0.5 rank ordering partially transfers** (AUC 0.809; 5/10 at 6/44 FR on
  a 20% budget, 7/10 at 9/44 on 30%) but its probability calibration does not:
  at a fixed 0.5 cut it rejects 28/44 corrects. A production open-ended gate
  would need a budget/quantile policy, not a probability threshold — and would
  need a validation family to set the budget.
- **The MLP does not transfer at all** (organic AUC 0.516), and **L0.75/L1
  collapse to accept-all** (0/10 at 0 FR). The representation that wins in-family
  is not the one that generalizes.
- Removing truthfulqa did not hurt the checkable families: LOFO bench 100 still
  L0.5 0.888 / L0.75 0.981 / MLP 0.988; harvest-61 L0.5 0.846 / MLP 0.831.

## Caveats

- Family-held-out is harsher than production: a real open-ended workload may
  resemble one of the training families. But it is the honest test of "can the
  gate judge a kind of task it has not seen?"
- One open-ended family, 10 organic wrongs. The budget numbers move by ±1 item
  per catch; treat them as a direction, not a spec.
- No threshold or budget was tuned on the held-out family (that would defeat the
  test).

## Implication for the cascade

The local tier that works for checkable tasks is the **verifier**, not the
learned gate. For open-ended tasks the current gate is either miscalibrated or
has no ranking signal on an unseen family; routing those to the hosted API (or
training a gate with multi-family open-ended data) is the honest choice. The
checks in `decision-cascade-accounting-20261018` and this fold now agree: the
gate's measured value is as a cheap prefilter/fallback, not as the open-ended
verifier.

## Artifacts

- `lofo-truthfulqa.json` — held-out groups, per-head catches/AUC, budget table.
- `readout-intern-decision-0.8b-r2-lofo-truthfulqa.json` — LOFO fit metrics.
- `readout-head-intern-decision-0.8b-r2-lofo-truthfulqa.npz` and filtered states
  (gitignored).
- Fold construction: drop states whose `task_ids` start with `truthfulqa`
  (129 items), then
  `decision_readout.py fit --name intern-decision-0.8b-r2-lofo-truthfulqa
  --states-name intern-decision-0.8b-r2-lofo-truthfulqa --out <dir>`.
