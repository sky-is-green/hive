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
| **Clef-Flash-9B ternary r512 (deployed)** | **68/70** | **29/30** | **2 / 1** | **0 / 0** | **~17,900** |

Clef-Flash is the better **local** validator: it matches the API judge's train
verdicts (67/70) with zero false accepts and no API spend, and beats Tiny-Jev
on both splits.  On the held-out test the API judge still leads (28 vs 26);
the one wrong answer there is not separated until Clef's 0.80 threshold,
which then rejects 6 correct answers — so the honest operating point remains
0.5 with 1 leak on test.

The **ternary** row is the deployed `clef-ternary` harness backend on the
merged single-file release (PQ2_0 body + rank-512 corrections, 3.3 GB): the
llama.cpp bridge on CPU + the joint head sidecar in torch fp32 on CPU — the 9B
never enters torch and the GPUs stay free.  It reproduces the offline bridge
benchmark bit-for-bit (max |Δp| 5e-5), beats bf16 Clef and the API judge on
verdict count on both splits, and never false-rejects.  Its two train false
accepts are the same records bf16 gets wrong (`truthfulqa-0005/0014`); the
single test leak is `math-0020`.  **Caveat:** the corrected head's
probabilities are compressed into **[0.57, 0.60]** (one record per split clears
0.60) and the accept cliff is at 0.60, so 0.5 is a very safe operating point
but there is no graded confidence left — recalibrate before trusting any other
threshold.  Speed: ~17.9 s/check (dominated by the recurrent GDN CPU prefill),
so it is the quality lane, not the low-latency one.

Artifacts: `clef-flash-judge-eval-{train,test}.json` (per-task p_correct and
the full sweep, now carrying both the bf16 `clef-flash` and the
`clef-flash-ternary` entries).  Load: ~5 s warm page cache (bf16 sidecar) /
~12 s (ternary, bridge + head); the sidecar adapters use the text-only path
(backbone + joint head + tokenizer), avoiding the released processor's
torchvision dependency.

## Why this matters for the ternary plan

bf16 Clef-Flash is 18.8 GB and ~3.2 s/check on CPU — usable as the harness
validator when the engine is stopped, but not co-resident-friendly.  The
ternary build (PQ2_0 body 3.3 GB + head sidecar) *is* the co-resident lane:
no torch on the 9B, no GPU, 68/70 and 29/30 at 0.5.  It is ~5-6x slower per
check than the bf16 transformer path because the deployed runtime is the
recurrent GDN prefill; if latency becomes binding, the same bridge builds for
HIP and can use the non-display card (one heavy process policy).
