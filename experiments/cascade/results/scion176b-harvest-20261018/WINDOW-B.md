# Window B — the generator's own head, refit out of task (2026-10-18)

The question Window B was run to answer: **is the 176B's own readout a real organic instrument, or
was the 0.803 on organic-66 mostly in-domain?**  Fusion v1 had shown 61% of organic-66 (40/66 items,
7 of the 9 wrongs) sits inside the `176b-own-community` head's train tasks, leaving a 2-wrong clean
slice.  Answer: **it is real.**

## The window

| | |
| --- | --- |
| serve down | 11:36:33 → 12:25:03 (48 min 30 s total) |
| engine load | 37 s |
| embed | 937 judge prompts, 2845 s, **3.0 s/prompt**, 0 retries |
| probe rc | 0, states written (937×2560) |
| restore | `scion-flashnext` healthy in 25 s |
| cross-window check | the 136 rows Window A embedded are **bit-identical** here (mean cosine 1.000000) |

The first attempt died at item 900/937 when the engine froze and the supervisor SIGTERM'd and
auto-restarted it; the probe now rides that out (see `f54c20d`).  The 3.0 s/prompt here versus the
5-7 s/prompt of attempt 1 and Window A's 2.7 s/prompt is page-cache state, not a different workload —
budget ~95 min for a cold window, ~50 min warm.

## The clean head

ridge+Platt on the 2560-d judge states of the **871 harvest train** items, lambda chosen by
`GroupKFold(5)` by `task_id` inside the train split, tested on the 66 organic items.  **The 66 test
items share 0 tasks with the 871 train items** (verified and printed by the script), so this is a
cross-task number, not an in-domain one.

| lambda | grouped-CV AUC (on the 871) | organic-66 AUC | catch@0.5 / CR@0.5 | frontier @ 0 / 3 / 5 / 9 / 14 |
| --- | --- | --- | --- | --- |
| 1 (default) | 0.974 | 0.766 | 2/9 · 1/57 | 2/0 · 3/3 · 4/5 · 5/6 · 6/10 |
| 1000 | 0.987 | 0.819 | 1/9 · 0/57 | 1/0 · 5/3 · 5/3 · 6/7 · 7/14 |
| **10000 (CV-picked)** | **0.989** | **0.832** | 2/9 · 2/57 | 1/0 · 4/3 · 4/3 · 5/8 · 8/13 |
| 100000 | 0.988 | — | — | — |

Per family (cv-picked): gsm8k **0.977** (3 wrongs), humaneval 0.792 (4), math 0.591 (2).

For comparison on the same 66 items: frozen `176b-own-community` **0.803** (61% in-domain), frozen
`176b-own-jbench` 0.671, 0.8B MLP 0.844, raw answer-slot margin 0.603.

**So the clean cross-task number (0.832) is higher than the in-domain one (0.803).**  The Window A
reading was not inflated by task overlap — if anything the overlap was hiding how well the
direction transfers, because the same head fitted on the *community* corpus generalises worse to
harvest than a head fitted on harvest itself does.  Regularisation matters a lot here (0.766 →
0.832 across the grid) and the CV picked it correctly.

## The honest fusion with it

`fusion_v1.py` extended: the 176B signal now covers **all 1,133 train rows** (bench-262 from the port
states, harvest-871 from this window) and all 182 test rows, so the self feature finally gets a
weight learned on organic data instead of one inherited from bench.

| features | test-182 | bench-116 | organic-66 | frontier @ 0 / 3 / 5 / 9 / 14 |
| --- | --- | --- | --- | --- |
| MLP alone | 0.977 | 0.991 | 0.844 | 0/0 · 5/1 · 5/1 · 6/9 · 8/13 |
| L0.5+MLP (best no-self) | 0.978 | 0.991 | 0.852 | 1/0 · 5/1 · 5/1 · 6/9 · 8/12 |
| + self-community | 0.974 | 0.989 | 0.860 | 0/0 · 6/3 · 6/3 · 7/7 · 7/7 |
| **+ self-harvest (Window B)** | **0.988** | **0.999** | **0.897** | **3/0 · 3/0 · 7/5 · 7/5 · 8/12** |
| + self-harvest + self-community | 0.988 | 0.999 | 0.895 | 3/0 · 5/1 · 7/4 · 7/4 · 7/4 |
| + self-jbench (frozen) | 0.980 | 0.996 | 0.803 | 1/0 · 2/1 · 4/4 · 5/9 · 6/12 |

Per family for `+ self-harvest`: gsm8k 0.920 (1/3 caught), humaneval 0.917 (3/4), math 0.909 (0/2),
truthfulqa n/a (0 wrongs); on the **community-clean** slice 0.979 (2/2 caught, 1 correct rejected).

Same-fit-subset ablation (identical rows and base features, one extra column):

| fit rows | organic-66 |
| --- | --- |
| 871 harvest train, base4 | 0.825 |
| 871 harvest train, base4 + self-harvest | **0.889** |
| 1,133 train, base4 + self-harvest | **0.897** |

The head carries the largest coefficient in the fusion (2.69, vs 1.58 for the 0.8B MLP) and the
gain survives being fit on the organic half of the corpus.

## What each instrument catches of the 9 organic wrongs

p < 0.5 means the instrument rejects the answer; caught = at least one instrument rejects it.

| item | 0.8B L0.5 | 0.8B MLP | p_raw | self-community | self-harvest | caught by |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k-0009:hs2 | 0.426 | 0.936 | 0.861 | 0.863 | 0.651 | L0.5 |
| gsm8k-0015:hs2 | 0.009 | 0.000 | 0.765 | 0.266 | 0.422 | all but p_raw |
| gsm8k-0033:hs1 | 0.525 | 1.000 | 0.876 | 0.991 | 0.787 | **none** |
| humaneval-0001:hs1 | 0.760 | 0.877 | 0.674 | **0.341** | **0.185** | self heads only |
| humaneval-0001:hs2 | 0.005 | 0.032 | 0.664 | 0.171 | 0.803 | all but self-harvest |
| humaneval-0015:hs1 | 0.001 | 0.001 | 0.500 | 0.674 | 0.606 | 0.8B heads |
| humaneval-0015:hs2 | 0.449 | 0.058 | 0.568 | 0.952 | 0.801 | 0.8B heads |
| math-0013:hs1 | 0.999 | 0.724 | 0.907 | 1.000 | 0.774 | **none** |
| math-0020:hs2 | 0.003 | 0.016 | 0.866 | 0.064 | 0.856 | all but self-harvest |

The complementarity is the whole argument for the fusion and it is visible row by row: the two self
heads each catch items the 0.8B heads cannot reach and vice-versa.  `humaneval-0001:hs1` — the
"universal miss" that sat above 0.87 on every probability signal in the previous three reports — is
now rejected by **both** self heads (0.341 and 0.185) and by nothing else.  Exactly two items
remain invisible to every instrument measured: `gsm8k-0033:hs1` and `math-0013:hs1`.

Wiring the exported refit head straight in as a feature gave **organic 0.924** — the best number in
the project, and wrong.  The head is fit on the 871 harvest train rows, so on those rows its score
is in-sample (train AUC 0.996) and the fusion was learning to trust the feature from the head's
memorisation.  Refitting it out-of-fold on the train side (`GroupKFold(5)` by `task_id`, same
lambda) gives 0.897 on exactly the same test items.  The honest number is 0.897.

## Where this leaves the gate

- **The 176B's own readout is a real, free instrument and it is the strongest organic signal we
  have.**  0.832 cross-task on its own, 0.897 in the fusion, against 0.852 for the best fusion of
  the 0.8B heads and 0.844 for the single best 0.8B head.  It costs no extra model, no extra VRAM
  and no second body: one engine pass, the same one that produced the answer.
- **The deployable gate is now `L0.5+MLP+p_raw+self-harvest`**, and the frontier moved from
  5/9-at-1-rejection to **7/9 at 5** (8.8% correct-rejection budget), 8/9 at 12.
- **`self-jbench` stays out.**  It is a QA/structured verifier (qa-75 0.979, organic 0.671) and
  every fusion that includes it loses organic AUC (0.897 → 0.809).  It is worth keeping as a
  task-form router input, not as a gate feature.
- Two items are still missed by every instrument: `gsm8k-0033:hs1` and `math-0013:hs1`.  The residual
  is representational, not a weighting problem.

## A leak I introduced and closed

## Is the readout underfit?  Ridge vs small encoders (`probe_head_shapes.py`)

The head above is a plain ridge+Platt needing λ=10000 on 871 items, which smells like an underfit —
and on the 0.8B an MLP had beaten the ridge on organic by a wide margin (0.844 vs 0.688).  So the
same question was asked of the 176B's own states, selecting on the 871 harvest train items under
`GroupKFold(5)` by `task_id` (461 wrongs — the only properly powered measurement available; the
66-item test has 9 wrongs and an AUC SE near 0.08).

| head on the 176B judge state | grouped-CV AUC |
| --- | --- |
| **ridge λ=10000** | **0.9889 ± 0.0042** |
| MLP-64 | 0.9856 ± 0.0050 |
| MLP-128 | 0.9860 ± 0.0051 |
| MLP-256 | 0.9858 ± 0.0052 |
| MLP-512 | 0.9860 ± 0.0052 |

**Every MLP is marginally worse than the tuned ridge, inside one SEM.**  There is no capacity to
win on this representation.

The same sweep on the 0.8B concat (train 1133, 655 wrongs) explains why, and corrects a belief the
head-selection history rested on:

| 0.8B head | grouped-CV AUC | organic-66 |
| --- | --- | --- |
| ridge λ=1 (what was deployed) | 0.8939 ± 0.019 | 0.688 |
| ridge λ=1000 (tuned) | **0.9650 ± 0.003** | — |
| MLP-64 (what was deployed) | 0.9682 ± 0.007 | 0.844 |

Those 0.8B ridges were fit at `ridge_fit`'s default **λ=1**.  Properly regularised they reach
0.9650, and the MLP's remaining advantage is **+0.003 — a wash**.  The "MLP beats ridge" result was
mostly missing regularisation, not capacity.  The deployed MLP is not wrong, but it was not earned,
and no head-shape change is the lever here.

## Caveats

- n = 9 organic wrongs.  The 0.852 → 0.897 step is ~0.4 items and the frontier step (5/1 → 7/5) is
  two catches for four extra rejections.  The direction is consistent across three fit subsets and
  across two feature sets, which is why it is reported as a real effect, but a single item would
  move it by 0.11.
- The refit head saw the same 871 harvest train items as the 0.8B heads (the r2 train split contains
  them).  The two instruments are still independent — different bodies, different hidden states —
  but they are not independent *training sets*, so some of the 0.897 is shared-corpus fitting.
- `humaneval` and `math` per-family numbers rest on 4 and 2 wrongs.
- The 378 bench items get the harvest head's full-fit score: the head never saw a bench item, so
  those are out-of-sample too, but bench-116 is a perturbed-MCQ distribution rather than organic
  free-form, and 0.999 there should not be read as an organic claim.

## Files

- `eval_harvest_head.py` — this analysis (cross-window check, frozen heads, grouped-CV refit,
  per-item dump for the 9 organic wrongs).
- `harvest-head-eval.json` — every number above.
- `harvest-refit-head.json` — the clean head in the standard head format, consumed by `fusion_v1.py`.
- `probe-states-scion176b-harvest.npz` (gitignored) — the 937×2560 states.
- `window.log`, `engine.log`, `server.log` (gitignored) — timings and engine record.
- `../fusion-v1-20261018/FUSION-V1.md` — the no-self fusion this extends.