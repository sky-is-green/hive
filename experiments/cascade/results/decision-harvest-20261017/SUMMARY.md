# Decision harvest — stage 2 data (organic 176B answers + amplification)

The distribution fix for the trained readout. Stage 1 fit its head on J-Bench,
whose negatives are checker-verified *perturbations* of recorded answers — the
easy half. This stage asks the live 176B (port 8081) for fresh answers to every
one of the 100 benchmark tasks, labels each answer with the task's own checker
(`checkers.check`), keeps the organic wrong answers as gold negatives, and
amplifies correct **train**-side answers with the exact perturbation recipes
J-Bench used (`judge_bench.perturbations`, checker-verified wrong). Test-side
tasks stay **organic-only**, so the harvested test slice is a fresh-distribution
measurement and never synthetic.

Generation: thinking disabled via `chat_template_kwargs` (`enable_thinking:
false`) — the model otherwise spends the budget in `reasoning_content` and
truncated requests return no final `content`. 1 greedy + 2 sampled (temp 0.7)
answers per task, `max_tokens` 640.

## Harvest results

| | count |
|---|---|
| requests to the live 176B | 300 |
| organic kept (checker-labeled) | **224** (201 correct / 23 wrong) |
| verified perturbations (train-side) | 418 |
| truncated at 640 tokens (dropped, never labeled) | 44 |
| exact duplicates across samples (skipped) | 32 |
| request errors | 0 |

Organic answers per bucket (correct / wrong): reasoning 146 / 7, code 38 / 12,
qa 17 / 4. Perturbations: num-flip 243, boxed-edit 75, code-edit 64,
choice-flip B/C/D 36.

Corpus by split (items):

| split | organic correct | organic wrong | perturbed |
|---|---|---|---|
| train (70 tasks) | 144 | 14 | 418 |
| test (30 tasks) | 57 | 9 | — |

Integrity: item ids unique (642/642); every label re-verified by re-running the
task checker over the stored candidate (0 mismatches); sha256 in the manifest.

## Reading

- The organic wrongs are the target distribution: real code bugs
  (`humaneval-*`, 12), boxed/α-numeric math misses (`math-*`), gsm8k slips
  (including a rounding miss 24.3 vs 170/7 under the checker's exact
  tolerance), and a systematic truthfulqa choice failure where greedy and both
  samples pick "D" (wrong). These are exactly the cases stage 1 showed the raw
  verbalizer accepts confidently.
- Truncation is concentrated in the `math-*` boxed solutions: all 27 attempts
  across 9 math tasks hit the cap, leaving those tasks with zero organic
  answers; 33 further tasks kept 1-2 of 3. A higher-cap rerun of the affected
  tasks is the obvious gap-fill if math proves a weak spot after the refit.
- The 32 deduplicated samples are mostly qa (short answers repeat across
  temperatures); dedupe keeps one copy per task.
- Serve shape: health 200 before and after; decode 36-69 t/s with the usual
  file-tier expert reads; 300 requests in 1541 s. No errors, no restarts.

## Artifacts

- `harvest-bench.jsonl` — 642 items, J-Bench schema + `source: "harvest"` and
  per-item `gen` metadata (temperature, completion tokens, ms).
- `harvest-manifest.json` — config, serve info, stats, sha256.
- `harvest.log` — progress log.

## Next (same session)

Refit the frozen-0.8B readout on J-Bench + this corpus (`--bench` repeatable,
by-source test metrics) and evaluate on both the frozen J-Bench test split and
the held-out organic harvest slice; the stage-1 head is applied to the same
corpus for the cross-distribution comparison.
