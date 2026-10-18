# Runner cascade — sampled arm B (no thinking, the comparable regime)

The same 66 items and the same command as arm A
(`runner-cascade-sampled-20261018`), with `--local-reasoning-effort none` so
the local answers are written without the engine's default thinking — the
regime the experiment loop measured in `cascade-e2e-sampled-20261018`.

```sh
.venv/bin/python experiments/cascade/run_cascade.py \
    --tasks experiments/cascade/results/runner-cascade-sampled-nothink-20261018/tasks-organic66.json \
    --scion-base http://127.0.0.1:8081/v1 \
    --judge deterministic --repair auto \
    --local-temperature 0.7 --local-max-tokens 1024 \
    --local-reasoning-effort none \
    --out experiments/cascade/results/runner-cascade-sampled-nothink-20261018 \
    --run-name run
```

## Result

Labels are eval-only (the task checkers), computed after the loop:

| | n | |
|---|---|---|
| first local attempts correct | 57/66 | `scion_accuracy` 0.864 |
| served (cascade) correct | **60/66** | `cascade_accuracy` 0.909 |
| detections (repairs attempted) | 4, all `code_exec` | 0 on correct drafts (`false_rejects` 0) |
| repairs cleared | 3/4 | all three served correct |
| still flagged after repair | 1 | `humaneval-0007:hs2` |
| truncated after the 2x retry | 0 | 1024 tokens is enough without thinking |
| silently accepted wrongs | 5 | `gsm8k-0015:hs2`, `gsm8k-0033:hs1`, `gsm8k-0036:hs1`, `math-0023:hg`, `truthfulqa-0015:hg` |

Cost: 66 generations + 4 repairs (one generation each), mean 5.2 s and 237
completion tokens per answer; repairs 6.2 s mean.

## The interesting detection: "no python code block"

Three of the four detections fired because the draft answered a coding task in
prose, with no fenced python block at all — the task's own tests then had
nothing to run. The finding is quoted by the repair prompt, and the greedy
re-ask produced a fenced, passing implementation in all three cases
(`humaneval-0003:hs1`, `humaneval-0015:hs2`; arm A's `humaneval-0007:hs1` had
the same shape).  This is the repair leg earning its keep on a failure mode
accuracy numbers hide: the answer was **absent**, not wrong.

## The one that correctly did not pass

`humaneval-0007:hs2`'s repair is byte-identical to its draft (same
`SyntaxError`, `same_as_draft: true`): the loop flagged it rather than
accepting a second copy of the same breakage — the repeat-verbatim behaviour
the offline repair pass recorded on `humaneval-0015:hs2`.  With
`--api-fallback` set, this is the item that escalates.

## Arm comparison (same tasks, T=0.7)

| arm | first | served | detections | cleared | false rejects | truncated | silent wrongs |
|---|---|---|---|---|---|---|---|
| A thinking (engine default) | 56/66 | 59/66 | 3 (`code_exec`) | 3/3 | 0 | 3 | 4 |
| B no thinking | 57/66 | 60/66 | 4 (`code_exec`) | 3/4 | 0 | 0 | 5 |
| experiment loop (sampled, live) | 58/66 | 62/66 | 6 (consistency 2, code 4) | 4/5 | 0 | 0 | 3 |

Readings:

- The wiring reproduces the experiment's geometry inside the runner: detection
  is precision-first (no false rejects in either arm), the repair clears most
  of what is caught, and the served count beats the raw count.
- Thinking did not help at this sample size (56 vs 57 raw) and cost ~3x the
  tokens, ~4x the wall clock, and 67 s repairs against 6 s.
- `consistency` fired zero times in both runner arms, while the experiment's
  sampled run caught 2/9 with it — different samples and n=9, but the
  arithmetic detector is thin in this regime either way.
- The blind spot is stable across arms: an internally consistent wrong whose
  code parses and whose failure the task's own tests do not see reaches no
  tier.  Arm B's five are in the table above; `gsm8k-0033:hs1` is the known
  universal miss (above 0.87 for every signal measured so far,
  `scion176b-self-organic-20261018/SELF-HEAD.md`).
