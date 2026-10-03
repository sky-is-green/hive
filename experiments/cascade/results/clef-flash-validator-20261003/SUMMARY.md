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

Clef-Flash is the better **local** validator: it matches the API judge's train
verdicts (67/70) with zero false accepts and no API spend, and beats Tiny-Jev
on both splits.  On the held-out test the API judge still leads (28 vs 26);
the one wrong answer there is not separated until Clef's 0.80 threshold,
which then rejects 6 correct answers — so the honest operating point remains
0.5 with 1 leak on test.

Artifacts: `clef-flash-judge-eval-{train,test}.json` (per-task p_correct and
the full sweep).  Load: ~5 s warm page cache; the sidecar adapter uses the
text-only path (backbone + joint head + tokenizer), avoiding the released
processor's torchvision dependency.

## Why this matters for the ternary plan

bf16 Clef-Flash is 18.8 GB and ~3.2 s/check on CPU — usable as the harness
validator when the engine is stopped, but not co-resident-friendly.  The
ternary build (PQ2_0 backbone ≈ 2.4 GB + ~200 MB head sidecar) would make it
the always-on local validator; the parity target is the bf16 decisions above.
