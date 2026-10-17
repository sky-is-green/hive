# Decision readout — stage 1 (trained head on the frozen 0.8B)

The "train the smallest format" feasibility gate: keep
`internlm/Intern-Decision-0.8B` **frozen**, extract the hidden state at the
decision position (the token before `<decision>` — the same position whose
next-token logits the model's own verbalizer reads) for all 378 frozen J-Bench
items through the model's own skeleton (`inference.compile_row` /
`HFBackend.encode`), fit tiny heads on the **train** split (262 items / 70
tasks), and score the frozen **test** split (116 items / 30 tasks; **zero
task_ids shared with train**).

## Results (test split, n=116: 29 correct / 87 wrong)

| readout | AUC | acc@0.5 | FA | FR | BA@best | ECE |
|---|---|---|---|---|---|---|
| model's own verbalizer (raw) | 0.771 | 74/116 | 35 | 7 | 0.759 | 0.296 |
| ridge, layer 12 (depth 0.5) | 0.915 | 93/116 | 20 | 3 | 0.868 | 0.167 |
| ridge, layer 18 (depth 0.75) | 0.935 | 102/116 | 13 | 1 | 0.943 | 0.147 |
| ridge, layer 24 (final) | 0.931 | 93/116 | 20 | 3 | 0.891 | 0.173 |
| ridge, concat(12, 18, 24) | 0.974 | 100/116 | 15 | 1 | 0.937 | 0.169 |
| MLP(64) on the concat | **0.981** | **108/116** | **3** | 5 | 0.943 | **0.052** |

For context, the raw readouts of the other on-disk judges on the same split:
`tiny-jev-1.7b` 0.889 / 87-116, `intern-decision-2b` 0.933 / 95-116 (FA 19,
FR 2, ECE 0.198), `intern-decision-4b` 0.979 / 104-116 (FA 11, FR 1, ECE 0.094).

The trained head on the frozen 0.8B reaches **4B-class quality at 0.8B cost**:
AUC 0.981 ≥ 4B's raw 0.979, accuracy 108/116 > 104/116, false accepts 3 < 11,
calibration 0.052 < 0.094 — with no model update at all, 0.4 s/verdict,
1.7 GB resident.

Integrity: the extraction's replica of the model's own readout matches the
decision-oracle record bit-for-bit on all 116 test items (max |Δp| = 0.0).

## Where the gain lives

Per-bucket test AUC (raw / ridge-concat / MLP):

- reasoning (19 correct / 58 wrong): 0.926 / 0.995 / 0.995
- code (6/17): 0.706 / 0.941 / 0.853
- qa (4/12): **0.323** / 0.958 / 1.000 — the verbalizer is worse than chance here

Per-origin acceptance @0.5 — of *wrong* items, lower is better (raw / ridge /
MLP): num-flip 9 / 3 / **0** of 36; code-edit 5 / 1 / 1 of 17; boxed-edit
**15** / 6 / **1** of 21; choice-flips 5 / 4 / 0 of 12; recorded-wrong 1 / 1 /
1 of 1. Of the 29 correct items: raw accepts 22, ridge 28, MLP 24.

## Reading

- The thesis holds at the readout level: the 0.8B's representations already
  carry ~4B-class verdict signal. The failure was in the two-symbol
  verbalizer, not the features — the trained head recovers it.
- Zero shared train/test task_ids rules out task memorization; the remaining
  risk is bench-family overfit (same sources and perturbation recipes). The
  agreed amplify+harvest step is exactly that test: fit on amplified bench +
  fresh 176B answers, validate on a held-out harvested set.
- The single "recorded-wrong" test item — an organic mistake, not a
  perturbation — is accepted confidently by everything, heads included. The
  synthetic perturbations are the easy half; organic failures are the real
  target distribution.
- Serving shape: the head adds two small matmuls to a forward pass the verdict
  path already runs. Ridge-concat (~3k weights) is the minimal integration;
  MLP-concat (~0.2M weights, calibrated) is the better gate. Both live in
  `readout-head-intern-decision-0.8b.npz` (on disk, gitignored).
- Policy note: at @0.5 the MLP is conservative (FA 3, FR 5) while ridge-concat
  is balanced (FA 15, FR 1); for escalation economics (a false accept is a
  silent failure) the MLP direction is right, and its ECE 0.052 makes
  threshold selection meaningful.

## Next (agreed direction)

Amplify the bench (`judge_bench.py` perturbations) + harvest fresh
(task, answer, label) pairs from the live 176B, then refit this head or LoRA
the 0.8B on the decision skeleton — evaluated on both the frozen J-Bench test
split and a held-out harvested set. Only after that does it ship into the
validator path as the resident gate.

## Artifacts

- `readout-states-intern-decision-0.8b.npz` — decision-position hidden states
  (layers 12/18/24), per-item raw verbalizer p, labels/splits/buckets/origins
  (gitignored, on disk).
- `readout-head-intern-decision-0.8b.npz` — trained heads (ridge per depth +
  concat with standardization and Platt params; MLP state dict) (gitignored).
- `readout-intern-decision-0.8b.json` — full metrics (train + test, per
  readout) plus the oracle cross-check.
- `extract-intern-decision-0.8b.json`, `memwatch.log` — run shape.

Run shape: CPU-only (no ROCm context beside the serve), 378 items in 158 s,
MemAvailable min 18.64 GiB, `/health` 200 before and after.
