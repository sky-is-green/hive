# INCIDENT — uncapped capture polluted by engine crashes (2026-10-03)

Run: `strata-bench-test-20261003` (held-out test split, 3 samples/task, T=0.7,
`max_tokens 3072`, thinking uncapped, **no recovery in the client**).

Result recorded then: 0.833 sample accuracy.  The misses were not model
errors.  Classification of the 15 failed samples (90 total):

| kind | samples | evidence |
|---|---:|---|
| engine restart mid-stream | 8 | `finish_reason=null`, no usage, content empty, partial reasoning (e.g. math-0005/0017, humaneval-0004, truthfulqa-0015) |
| thinking truncation | 2 | `finish_reason=length`, 3,072 tokens, 11-12 KB of reasoning, no answer (humaneval-0015) |
| genuine wrong answers | 5 | GSM8K-0015 ×3, MATH-0020 ×2 at T=0.7 |

Root cause: the long mmap-experts session accumulated **195 GB** of expert
file reads (device log); under sustained generation the MTP verify window
wedged — `verify: timed out at layer N … (#267)` / `an earlier window never
finished on the GPU` — and the engine exited (4 `engine stopped unexpectedly`
events across the session).  The Strata server restarts the engine on the next
request, but the raw streaming client recorded the dead stream as a wrong
answer.

Mitigation shipped with the refit:

- `stream_capture.stream_reliable()` and `run_cascade.ScionTier._post_local()`
  now recover: an `/unload` (fresh engine) then retry, up to 3 times;
  genuine 4xx client errors still raise immediately.
- The re-run `strata-bench-test-capped-20261003` used a thinking cap
  (`reasoning_budget_tokens 1024`) and `max_tokens 2048` and recorded **zero**
  recoveries: 0.889 sample accuracy over the same 90 samples.
- The label runs bounded local generation the same way; the two #267 wedges
  there recovered without losing a label.

Superseded by `strata-bench-test-capped-20261003` for the model card.
