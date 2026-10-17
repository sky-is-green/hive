# Decision oracle — organic-slice verdicts (harvest 2026-10-17)

Stage 2 showed the refit MLP holding the frozen bench (0.980) but catching only
4/9 organic wrongs. Before touching the 0.8B (LoRA or another refit), this stage
asks whether **bigger raw judges** see what the small readout misses: intern-2b,
intern-4b and tiny-jev score the same 224 organic items the 176B actually
produced for the harvest — 201 correct / 23 wrong, 66 test items with 9 wrongs,
task-disjoint from every training item (63 train tasks vs 28 test tasks, zero
overlap). Judged zero-shot on CPU beside the serve: same verdict protocol
(`noul` P(yes) at the token before `<decision>`, symbols A/B, temp 2.7478 /
2.1005 / 1.9924).

Run: 974 s total (2b 224 s, 4b 588 s, tiny-jev 162 s), MemAvailable min
19.5 GiB, serve `/health` 200 before and after.

## Raw judges — organic slice (all 224, and unbiased test-66)

| judge | all-224 AUC | all-224 acc (FA/FR) | test-66 AUC | test-66 acc (FA/FR) | caught wrongs (test-9) |
|---|---|---|---|---|---|
| intern-2b | 0.757 | 191/224 (20/13) | 0.668 | 54/66 (9/3) | **0/9** |
| intern-4b | 0.837 | 201/224 (17/6) | **0.799** | 58/66 (7/1) | **2/9** |
| tiny-jev-1.7b | 0.717 | 192/224 (18/14) | 0.628 | 54/66 (8/4) | 1/9 |

4b's two catches are both gsm8k arithmetic slips (`gsm8k-0009:hs2` p=0.281,
`gsm8k-0015:hs2` p=0.392). 2b accepts all nine (min p=0.574); tiny-jev's only
catch is `humaneval-0015:hs1` (p=0.478).

## 0.8B family + refit probes — organic test-66 (9 wrongs)

All 0.8B scores come from the stage-2 extraction states; stage-1 heads are
bench-only, stage-2 heads and probes are refit on bench + harvest.

| scorer | AUC | acc (FA/FR) | caught | bench-test reference |
|---|---|---|---|---|
| 0.8b raw verbalizer | 0.603 | 48/66 (9/9) | 0/9 | 0.771 / 74-116 |
| stage-1 ridge-concat | 0.758 | 59/66 (7/0) | 2/9 | 0.974 / 100-116 |
| stage-1 MLP-concat | 0.646 | 53/66 (6/7) | 3/9 | 0.981 / 108-116 |
| refit ridge-concat | 0.673 | 55/66 (6/5) | 3/9 | 0.922 / 91-116 |
| refit MLP-concat | 0.774 | 52/66 (5/9) | 4/9 | **0.980 / 110-116** |
| refit L0.75 probe | 0.772 | 54/66 (6/6) | 3/9 | 0.891 / 89-116 |
| refit L1 probe | 0.762 | 54/66 (4/8) | 5/9 | 0.875 / 84-116 |
| **refit L0.5 probe** | **0.823** | 54/66 (3/9) | **6/9** | 0.862 / 93-116 |

**New finding:** the refit's mid-depth ridge probe (L0.5) is the best single
organic scorer — 6/9 wrongs caught (AUC 0.823) — better than the refit MLP it
sits beside (4/9), at the price of 17 bench corrects (93 vs 110 accepted). The
probe lattice trades bench accuracy for organic reach; selecting heads by bench
alone had hidden this.

## Probe ensembles (reject if any member < 0.5) — organic test-66

| members | caught | wrongs accepted (FA) | corrects rejected (FR/57) |
|---|---|---|---|
| L0.5 | 6/9 | 3 | 9 (16%) |
| MLP | 4/9 | 5 | 9 (16%) |
| L0.5 + MLP | 6/9 | 3 | 17 (30%) |
| L0.5 + MLP + L1 | **7/9** | 2 | 20 (35%) |
| + L0.75 + concat | 7/9 | 2 | 21 (37%) |

Union-rejection buys one more wrong for roughly double the correct-rejection
rate — an escalation-cost decision, not an accuracy one.

## The nine organic test wrongs — who catches what (p at 0.5)

| item | L0.5 | MLP | L1 | 4b raw |
|---|---|---|---|---|
| gsm8k-0009:hs2 | **C** 0.012 | 0.993 | 0.546 | **C** 0.281 |
| gsm8k-0015:hs2 | **C** 0.041 | **C** 0.000 | **C** 0.173 | **C** 0.392 |
| gsm8k-0033:hs1 | **C** 0.042 | 0.997 | 1.000 | 0.937 |
| math-0013:hs1 | 0.997 | 1.000 | 0.986 | 0.974 |
| math-0020:hs2 | **C** 0.146 | **C** 0.028 | **C** 0.482 | 0.933 |
| humaneval-0001:hs1 | 0.999 | 0.993 | 0.897 | 0.836 |
| humaneval-0001:hs2 | 0.685 | 0.844 | **C** 0.401 | 0.965 |
| humaneval-0015:hs1 | **C** 0.000 | **C** 0.000 | **C** 0.015 | 0.905 |
| humaneval-0015:hs2 | **C** 0.271 | **C** 0.000 | **C** 0.005 | 0.956 |

**Two items are invisible to every scorer** (raw judges and all refit probes):

- `math-0013:hs1` — 2007th digit of 1/7: the candidate writes
  `2007 = 6×334 + 1`, takes remainder 1, boxes `1` (correct: 2). Fluent 411-token
  solution; min probe p = 0.952.
- `humaneval-0001:hs1` — `digits()` multiplies *even* digits into the product
  (missing the odd-digit guard), so `digits(235)` returns 30 not 15. Clean,
  well-formed code; min probe p = 0.589 (L0.75, still accepted).

`humaneval-0001:hs2` is caught only by concat-style heads (stage-1 ridge 0.448,
refit ridge 0.337) and L0.75 — one probe family away from the MLP frontier.

## Reading

- **A bigger judge does not close the organic gap.** 4b is the strongest raw
  scorer (0.799, 2/9) and 2b is blind (0/9); both see only the gsm8k
  arithmetic slips. The residual failures are fluent, well-formed reasoning and
  code with embedded micro-errors — the decision-prompt modality does not
  expose them, at 0.8B or 4B.
- The refit's own probe lattice contains the best organic scorer (L0.5, 6/9),
  but no single probe is good on both distributions: bench-best (MLP 0.980/110)
  catches 4/9; organic-best (L0.5) drops 17 bench corrects. An OR-ensemble
  reaches 7/9 at 35% correct-rejection — usable only if escalation is cheap
  enough.
- Two wrongs evade every current scorer, and they are exactly the LoRA /
  targeted-harvest case: feature-space heads cannot separate them because the
  backbone features do not encode the micro-error. The pre-registered weighted
  refit (w10 primary, w3 sensitivity) runs next and should evaluate the full
  probe lattice, not just the MLP.
- Small-n caveat stands: 66 test items / 9 wrongs, so ±1 item ≈ 0.11 AUC.
  The raw-vs-heads and bench-vs-organic directions are the robust parts.

## Artifacts

- `harvest-organic.jsonl` — the 224 organic items judged here (derived from the
  committed `decision-harvest-20261017/harvest-bench.jsonl`; force-added).
- `judge-intern-decision-2b-harvest-organic.json`,
  `judge-intern-decision-4b-harvest-organic.json`,
  `judge-tiny-jev-1.7b-harvest-organic.json` — raw verdicts.
- `judge-comparison.json` — all metrics above + per-item p values for the 23
  wrongs and the test-9 catch matrix.
- `run.log`, `memwatch.log` — run shape (on disk only).
