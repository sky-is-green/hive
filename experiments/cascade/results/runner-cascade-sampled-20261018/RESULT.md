# Runner cascade — sampled arm A (engine default: thinking on)

Integrated pass of the pilot runner after the deterministic verifier and the
repair leg were wired into it (`experiments/cascade/run_cascade.py`, commits
`1bb4627` + `d222478`): 66 bench items, one task per sampled item
(`tasks-organic66.json`), T=0.7, `max_tokens` 1024 with the 2x truncation
ladder, `--judge deterministic --repair auto`, **no** `--api-fallback` — a
reject is flagged and served as-is, never escalated, so this arm isolates
detect + repair.

```sh
.venv/bin/python experiments/cascade/run_cascade.py \
    --tasks experiments/cascade/results/runner-cascade-sampled-20261018/tasks-organic66.json \
    --scion-base http://127.0.0.1:8081/v1 \
    --judge deterministic --repair auto \
    --local-temperature 0.7 --local-max-tokens 1024 \
    --out experiments/cascade/results/runner-cascade-sampled-20261018 \
    --run-name run
```

## Result

Labels are eval-only (the task checkers), computed after the loop:

| | n | |
|---|---|---|
| first local attempts correct | 56/66 | `scion_accuracy` 0.848 |
| served (cascade) correct | **59/66** | `cascade_accuracy` 0.894 |
| detections (repairs attempted) | 3, all `code_exec` | 0 on correct drafts (`false_rejects` 0) |
| repairs cleared | 3/3 | all three scored correct after `canonical()` |
| truncated after the 2x retry | 3 | flagged, not served silently |
| silently accepted wrongs (no detector fired) | 4 | `gsm8k-0009:hs1`, `gsm8k-0033:hg`, `gsm8k-0038:hs2`, `math-0020:hg` |

Cost: one generation per item, one more for each detection (3 repairs, one of
them needing the doubled-budget retry, `generations=2`).

## What the wiring did

- **The three detections are all `code_exec`**: an `IndentationError` at
  module level, a failing checker call, and a draft that contained **no code
  block at all** ("the draft contains no python code block" is itself a
  finding). The repair re-ask — finding quoted, temperature 0 — fixed all
  three, and the canonical fenced block is what was served.
- **`consistency` fired zero times in this arm.** Under the engine's default
  thinking the arithmetic the model writes stays internally consistent even
  when the final answer is wrong: the math blind spot is not
  arithmetic-visible here (in the no-thinking experiment arm it was the
  detector that caught 2/9 wrongs).
- **The truncation rule earned its place.** 11 answers hit the 1024 cap; 8
  answered on the doubled budget, and the 3 that were still cut off
  (`gsm8k-0033:hs1`, `gsm8k-0038:hs1`, `humaneval-0015:hs1`) were flagged
  rather than served as if they were answers. In the product they are the
  items `--api-fallback` exists for.
- **Repairs stay greedy**: `humaneval-0001:hs2` needed the 2x budget on the
  repair too (97 s); the other two were single-shot.
- Latency in this arm is thinking-bound: mean 14.8 s per local answer (p95
  41 s, max 70 s), mean 727 completion tokens; repairs averaged 67 s. The
  no-thinking arm B is the one comparable with the earlier experiment numbers.

## Notes

- `D3-verify` telemetry rows record `outcome=accept|reject`, which the
  summary's `non_ok` counter reads as all-non-ok (`non_ok_rate` 1.0); the
  real reject count is `repairs_attempted` + `flagged_no_fallback` here.
- The runner holds no gold at runtime: the verdict, the repair prompt and the
  gate all read `runtime_spec`, which drops `checker.expect`. The checkers in
  this report are applied post-loop for the label columns only.
