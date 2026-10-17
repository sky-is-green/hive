# Decision readout — round-2 refit (expanded organic corpus)

Same frozen 0.8B, same extraction path, same heads; only the corpus changed
(commit for harvest round 2): 937 harvest items instead of 642, and 43 train-side
organic wrongs instead of 14. Extraction: 1315 items (bench 378 + harvest 937)
in 651 s (0.50 s/item), MemAvailable min 18.84 GiB, serve `/health` 200 before
and after. Fit: train 1133 (bench 262 + harvest 871), test 182 (bench-test 116 +
organic test-66). The organic test-66 is untouched, so every organic number
below is directly comparable with stage 2.

## Bench + organic test, stage-2 vs round-2 refit

| variant | bench AUC | bench acc (FA/FR) | harvest AUC | harvest acc (FA/FR) | caught test-9 |
|---|---|---|---|---|---|
| **stage 2** | | | | | |
| ridge-concat | 0.922 | 91/116 (23/2) | 0.673 | 55/66 (6/5) | 3/9 |
| MLP-concat | 0.980 | 110/116 (2/4) | 0.774 | 52/66 (5/9) | 4/9 |
| L0.5 | 0.862 | 93/116 (19/4) | 0.823 | 54/66 (3/9) | 6/9 |
| L0.75 | 0.891 | 89/116 (24/3) | 0.772 | 54/66 (6/6) | 3/9 |
| L1 | 0.875 | 84/116 (29/3) | 0.762 | 54/66 (4/8) | 5/9 |
| **round 2** | | | | | |
| ridge-concat | 0.924 | 94/116 (20/2) | 0.688 | 56/66 (6/4) | 3/9 |
| **MLP-concat** | **0.991** | **111/116 (2/3)** | **0.844** | 55/66 (4/7) | **5/9** |
| L0.5 | 0.815 | 90/116 (19/7) | **0.827** | 54/66 (3/9) | **6/9** |
| L0.75 | 0.952 | 101/116 (14/1) | 0.819 | 60/66 (6/0) | 3/9 |
| L1 | 0.880 | 86/116 (27/3) | 0.770 | 56/66 (5/5) | 4/9 |

**The corpus growth moved the head on both distributions at once:** the refit MLP
gains organic AUC 0.774 → 0.844 (4/9 → 5/9 wrongs caught) while the bench also
improves (0.980 → 0.991, 110 → 111 correct, FA stays 2). This is the first
intervention in the cascade that moved the organic test at all. L0.75 became
the best bench probe (0.952, 101/116, 1 FR) and a **zero-false-reject** organic
option (60/66, catches 3/9), but the per-depth probes did not gain organic
catches.

## Ensembles (reject if any member < 0.5) — organic test-66

| members | stage-2 caught / correct-rejected | round-2 caught / correct-rejected |
|---|---|---|
| L0.5 | 6/9, 9 (16%) | 6/9, 9 (16%) |
| MLP | 4/9, 9 (16%) | **5/9, 7 (12%)** |
| L0.5 + MLP | 6/9, 17 (30%) | 6/9, 14 (25%) |
| L0.5 + MLP + L0.75 | — | 6/9, 14 (25%) |
| L0.5 + MLP + L0.75 + L1 + concat | 7/9, 21 (37%) | 6/9, 17 (30%) |

The full union slips by one item (below), while the MLP and the two-probe union
both get cheaper on correct rejections.

## The nine organic test wrongs, round-2 p (C = caught < 0.5)

| item | ridge | MLP | L0.5 | L0.75 | L1 |
|---|---|---|---|---|---|
| gsm8k-0009:hs2 | 0.690 | 0.936 | **C 0.426** | 0.970 | 0.765 |
| gsm8k-0015:hs2 | 0.790 | **C 0.000** | **C 0.009** | **C 0.074** | **C 0.083** |
| gsm8k-0033:hs1 | 1.000 | 1.000 | 0.525 | 0.998 | 1.000 |
| humaneval-0001:hs1 | 0.930 | 0.877 | 0.760 | 0.916 | 0.985 |
| humaneval-0001:hs2 | **C 0.018** | **C 0.032** | **C 0.005** | **C 0.112** | 0.684 |
| humaneval-0015:hs1 | **C 0.069** | **C 0.001** | **C 0.001** | **C 0.093** | **C 0.075** |
| humaneval-0015:hs2 | 0.741 | **C 0.058** | **C 0.449** | 0.570 | **C 0.298** |
| math-0013:hs1 | 1.000 | 0.724 | 0.999 | 0.910 | 0.936 |
| math-0020:hs2 | **C 0.011** | **C 0.016** | **C 0.003** | 0.591 | **C 0.314** |

- `math-0013:hs1` moved from "everyone at ~1.0" to MLP 0.724 — the first sign
  the added organic negatives reach the fluent arithmetic-slip class, but it is
  still accepted.
- `gsm8k-0033:hs1` now sits at 0.525 under L0.5 (was 0.042) — a threshold slip
  that costs the 5-probe union its seventh catch. With nine wrongs, ±1 item is
  ±0.11 AUC; treat the net union comparison as flat.
- `humaneval-0001:hs1` remains invisible to every head (min p 0.760).

## Reading

- The data-side lever works: organic negatives at 3× the previous count moved
  both the bench and the organic frontier for the MLP, with no trade-off. The
  corner that was stuck at 4/9 now has a 5/9 head holding 0.991 on the bench.
- The probe lattice did not gain from the extra data (L0.5 flat at 6/9; L0.75
  better on bench only), so the ensemble frontier is unchanged-to-slightly
  cheaper. Operational options on the table: MLP (5/9, 12% correct-rejection),
  L0.5 (6/9, 16%), L0.75 (3/9, 0%) — an escalation-cost decision.
- The two universal misses survive; the residual is still representational.
  Pre-registered next move: **LoRA the 0.8B backbone** on the same expanded
  corpus (verdict-objective training; the driver is prepared), then re-extract
  and refit if the adapted model output warrants it.

## Artifacts

- `readout-states-intern-decision-0.8b-r2.npz` (gitignored) — 1315-item
  extraction; `draw` in `extract-intern-decision-0.8b-r2.json`.
- `readout-intern-decision-0.8b-r2.json` — full fit: all readouts × sources,
  best-BA thresholds, ECE.
- `r2-readout-comparison.json` — stage-2 vs round-2 rows, test-9 catch lists,
  per-item p, ensemble numbers.
- `readout-states-intern-decision-0.8b-r2-testmath.npz` (gitignored) +
  `extract-intern-decision-0.8b-r2-testmath.json` — 4/4 extension items accepted
  (0 false rejects, no wrongs to test rejection).
- `memwatch.log` — on disk only.
