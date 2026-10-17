# Decision harvest — round 2 (train-side organic top-up)

Round 1 (commit `1ea6231`) left the refit with only 14 train-side organic
wrongs against 418 perturbation negatives — too few to move a 3072-d head, and
the pre-registered organic weighting was a null (commit `15bcf72`). Round 2
grows the **organic** side: more sampled answers per train task, a higher token
cap for the math CoTs, no new perturbations (the 418 verified ones stand), and
cross-corpus dedupe so every run answer is new. Test-side tasks are untouched;
the frozen organic test-66 stays exactly as pre-registered.

Config per run (`--max-neg 0`, thinking off, `--seen-corpus` round 1):

| run | tasks | attempts/task | max_tokens | tag | requests | kept | wrongs | trunc | dup | err | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nonmath | 53 train (gsm8k/humaneval/truthfulqa) | 1 greedy + 6 @ 0.7 | 640 | `r2-` | 371 | 252 | 24 | 7 | 112 | 0 | 1183 s |
| math | 17 train | 1 greedy + 2 @ 0.7 | 1280 | `r2-` | 51 | 43 | 5 | 8 | 0 | 0 | 646 s |
| testmath | 2 test (math-0005/0017, zero coverage) | 1 greedy + 3 @ 0.7 | 1280 | `t2-` | 8 | 4 | 0 | 4 | 0 | 0 | 133 s |

Total: 430 requests, **299 new organic answers (29 wrongs)**, 0 errors, serve
`/health` 200 throughout (start, between runs, end). No re-run interfered with
the 176B serve.

## Yield

| family | new items | new wrongs | wrong rate |
|---|---|---|---|
| gsm8k (26 train tasks) | 179 | 4 | 2.2% |
| humaneval (16) | 55 | 16 | **29.1%** |
| truthfulqa (11) | 18 | 4 | 22.2% |
| math (17) | 43 | 5 | 11.6% |

Wrongs spread over 15 tasks; `humaneval-0019` alone contributed 5. The
code/qa families are the organic-negative engine; gsm8k stays low-yield even
with six samples.

## Merged corpus of record

`harvest_merge.py` concatenates the base (round 1) with the two train runs,
re-runs each item's checker, and refuses duplicates, non-train items, and id
collisions. All **941 labels re-verified, 0 mismatches**.

| | round 1 | round 2 merged |
|---|---|---|
| items | 642 | **937** |
| train organic correct / wrong | 144 / 14 | **410 / 43** |
| train perturbations | 418 | 418 |
| test organic correct / wrong | 57 / 9 | 57 / 9 (frozen) |

Train organic wrongs **14 → 43** (3.1×), organic corrects **144 → 410** (2.8×).
The class mix shifts toward organic answers: organic items are now 453/871 of
the train negatives/positives mass beside the 418 perturbations.

The math token-cap fix partially closed the coverage hole: all 17 train math
tasks now have organic items, but 12/59 math attempts still truncate at 1280
(the hardest CoTs), and the test-math extension yielded 4/8 kept (4 correct, 0
wrong) — a coverage note, not a measurement.

## Artifacts

- `harvest-bench-r2.jsonl` (force-added) — merged 937-item corpus; sha256 in
  `harvest-manifest-r2.json` (`1c853a8f…`).
- `harvest-testmath.jsonl` (force-added) — 4-item test-side extension
  (`math-0005`, `math-0017`), kept out of the training corpus.
- `harvest-manifest-r2.json`, `harvest-testmath-manifest.json` — configs,
  per-split/per-family/per-origin counts, sha256.
- `runs/{nonmath,math,testmath}/harvest-manifest.json` — per-run configs,
  stats, and corpus sha256 (raw run corpora stay on disk as intermediates).
- `experiments/cascade/harvest.py` — now supports `--split`, `--tasks`,
  `--tag-prefix`, `--seen-corpus` (round-2 use).
- `experiments/cascade/harvest_merge.py` — merge + label re-verification +
  extension split, reusable for future rounds.
