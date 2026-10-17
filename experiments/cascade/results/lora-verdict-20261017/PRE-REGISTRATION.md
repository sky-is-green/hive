# Pre-registration — LoRA verdict adaptation (written during training, before eval)

Run: `lora_verdict.py`, 1,133 train items (J-Bench train 262 + harvest-r2 train
871; 655 wrong / 478 correct), 2 epochs, rank 16 / alpha 32 / lr 1e-4 / accum 8,
bf16 CPU, verdict cross-entropy at the decision marker (yes=9405, no=2083).
Evals after each epoch: J-Bench test 116, organic test-66 (frozen), test-math
extension 4. Epoch snapshots kept so a better epoch can be selected.

This file fixes the decision rules before any LoRA eval number has been seen.
Reference points (frozen backbone, commit `7bfa7fb`):

| gate | bench AUC (acc) | organic test AUC | wrongs caught /9 | corrects rejected /57 |
|---|---|---|---|---|
| raw verbalizer | 0.771 (74/116) | 0.603 | 0/9 | — |
| r2 MLP head | 0.991 (111/116) | 0.844 | 5/9 | 7 (12%) |
| L0.5 probe | 0.815 (90/116) | 0.827 | 6/9 | 9 (16%) |
| 5-probe union | — | — | 6/9 | 17 (30%) |

## Rules

1. **The LoRA verdict is the object under test** — its own yes/no logits at the
   decision marker, no trained head on top. The best epoch by organic AUC is
   selected only if it does not lose more than 2 bench items vs the best-bench
   epoch; otherwise the best-bench epoch is used.
2. **Pass (proceed to adapted-backbone refit + gate trial):** organic test
   AUC ≥ 0.84 **and** bench AUC ≥ 0.95, or wrongs caught ≥ 6/9 at FA ≤ 4.
3. **Marginal (one more data round, no gate trial):** organic AUC 0.80–0.84
   with bench ≥ 0.95, or 5/9 caught with FA ≤ 3.
4. **Fail (stop representational work):** bench AUC < 0.92, or organic caught
   ≤ 4/9. Pivot to the operating-point decision among the existing heads
   (MLP / L0.5 / L0.75) and the escalation-cost trade-off.
5. **Adapted-backbone refit bar** (only if rule 2 passes): extract the 1315
   items with the chosen adapter (`decision_readout.py extract --adapter`),
   refit the same heads, and require bench ≥ 0.98 **and** organic caught ≥ 6/9
   before proposing the gate for trial. Report AUCs against the frozen-head
   reference, not in isolation.
6. n = 9 organic wrongs, so ±1 catch = ±0.11 AUC. A single-item swing is not a
   pass or a fail on its own; the bench and the organic frontier must agree.

Known risks recorded before the result: 2 epochs on 1,133 rows can overfit the
train organic wrongs (43) relative to the 418 perturbations; class balance is
55% wrong; the adapted features may still not encode `math-0013:hs1` or
`humaneval-0001:hs1`, which no frozen probe catches.
