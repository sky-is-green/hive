# Decision readout — stage 2 (refit on J-Bench + harvested 176B answers)

Stage 1 measured the frozen-0.8B readout on J-Bench, whose negatives are
checker-verified *perturbations* of correct answers. This stage refits the same
heads on J-Bench **plus** the harvested corpus (commit `1ea6231`): 1020 items
(bench 378 + harvest 642; train 838 / test 182). The harvest **test** slice is
strictly organic — answers the live 176B actually produced, labeled by each
task's own checker, no perturbations — and task-disjoint from every training
item, so it measures the deployment distribution the gate will face. The
stage-1 heads are applied to the same corpus unchanged (`--apply-head`) for the
cross-distribution comparison.

Extraction: 1020 items in 497 s (median 434 ms/item), MemAvailable min
19.64 GiB, serve `/health` 200 before and after; same CPU-only prompt path.

## Results — frozen bench test (n=116) vs organic harvest test (n=66: 57/9)

| readout | bench AUC | bench acc | bench FA/FR | harvest AUC | harvest acc | harvest FA/FR |
|---|---|---|---|---|---|---|
| raw verbalizer | 0.771 | 74/116 | 35 / 7 | 0.603 | 48/66 | 9 / 9 |
| refit ridge-concat | 0.922 | 91/116 | 23 / 2 | 0.673 | 55/66 | 6 / 5 |
| refit MLP-concat | **0.980** | **110/116** | **2 / 4** | **0.774** | 52/66 | 5 / 9 |
| stage-1 ridge-concat | 0.974 | 100/116 | 15 / 1 | 0.758 | 59/66 | 7 / 0 |
| stage-1 MLP-concat | 0.981 | 108/116 | 3 / 5 | 0.646 | 53/66 | 6 / 7 |

Pooled (bench+harvest, n=182): raw 122/182 (FA 44/FR 16) · refit MLP 162/182
(FA 7/FR 13, ECE 0.096). Organic wrongs rejected @0.5, of 9: raw **0**,
refit-ridge 3, refit-MLP **4**, stage-1 ridge 2, stage-1 MLP 3.

Per-bucket harvest AUC (raw / refit-MLP / stage-1 MLP): reasoning (5 wrong)
0.693 / 0.790 / 0.657; code (4 wrong) 0.250 / 0.667 / 0.583; qa (0 wrong) —
the bench columns remain 0.323-0.993 / 0.958-0.993 as in stage 1.

## Reading

- **The refit holds the frozen bench** (0.980 vs 0.981 stage-1; acc 110 vs 108;
  FA 2 vs 3; ECE 0.045) — adding harvest data did not cost bench performance,
  and slightly improved its calibration.
- **The organic slice is much harder for everything.** The trained heads find
  2-4 of 9 organic wrongs; the raw verbalizer finds 0 of 9. Bench-level
  validation would have overstated the gate — this is exactly what the harvest
  was for.
- The amplification thesis did not move the organic frontier: the refit trains
  to a perfect 838/838 separation (6144-dim features over 838 items) and the
  bench test transfers, but the held-out organic failures stay largely
  accepted. With 9 negatives, ±1 item ≈ 0.11 AUC: the ridge/MLP ordering among
  themselves is noise; the raw-vs-heads gap is the robust part.
- What gets caught vs missed: caught wrongs are structurally flawed (two
  humaneval logic bugs, a truncated math answer, one rounding miss the MLPs
  reject); missed wrongs are fluent, confident, internally consistent but
  wrong (a $345+$315 sum error delivered as "Final answer: 660", a wrong boxed
  digit, a wrong `humaneval` early-return). Hallucination-shaped failures
  remain the hard class, as in stage 1's single recorded-wrong item.
- False rejects concentrate on *greedy* correct answers (5/26 rejected at 0.5):
  calibration/threshold work is needed before any serving shape.
- **Ship decision: not shipping into the validator.** The 0.8B+readout is not
  yet a trustworthy escalation gate on the fresh distribution. Next options,
  cheapest first: (1) score the harvest-test slice with the raw intern-2b /
  4b verdicts (`decision_oracle.py` path) — if a bigger judge also fails these
  items, the bottleneck is the task, not the readout; (2) organic-weighted
  training (14 organic wrongs vs 418 perturbations in train — oversample or
  weight them); (3) LoRA the 0.8B on the decision skeleton so the backbone
  features move, not just the head; (4) harvest harder negatives (higher
  temperature, more samples, error-prone tasks) to grow the organic-negative
  mass before refitting again.

## Artifacts

- `readout-states-intern-decision-0.8b-harvest.npz` — combined-corpus states
  (L12/L18/L24, raw p, labels/splits/buckets/origins/**sources**; gitignored).
- `readout-head-intern-decision-0.8b-harvest.npz` — refit heads (gitignored).
- `readout-intern-decision-0.8b-harvest.json` — full metrics: train/test, per
  source, and per-bucket / per-origin breakdowns for raw, concat, MLP and both
  applied stage-1 heads.
- `extract-intern-decision-0.8b-harvest.json`, `memwatch.log` — run shape.

`decision_readout.py` gained: repeatable `--bench` (multi-corpus extraction),
`source`/`task_id` per item, `--apply-head` (saved heads applied to any corpus),
by-source test metrics and the breakdowns section.
