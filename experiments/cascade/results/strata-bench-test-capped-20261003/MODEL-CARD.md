# Benchmark results — Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf

Run with the local hivebench cascade harness; checked programmatically.

## Provenance

| field | value |
|---|---|
| run | `strata-bench-test-capped-20261003` (2026-10-03 02:30:13) |
| model | `<work>/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` |
| model sha256 | `69820c02ec7d0b45…` |
| stack / serving | `strata-125b`, 1 slots, ctx 4096 |
| git | `5cd95a87e3bb` (19 dirty files) |
| task set | `cascade-bench-v1` `c2891dc3ec049fba…`, split `test` |
| decode | T=0.7, max_tokens=2048, 3 samples/task |

## Results

| benchmark | tasks | samples | sample acc | task macro | mean ms | mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 12 | 36 | 0.972 | 0.972 | 9422 | 333 |
| humaneval | 6 | 18 | 0.833 | 0.833 | 24877 | 557 |
| math | 8 | 24 | 0.833 | 0.833 | 22468 | 498 |
| truthfulqa | 4 | 12 | 0.833 | 0.833 | 20050 | 290 |
| **overall** | 30 | 90 | **0.889** | **0.889** | — | — |

`sample acc` pools all samples; `task macro` averages per-task pass rates (equal weight per task).

## Protocol

- System prompt: `You are a precise assistant. Think if you need to, then give the final answer. For code questions return only the requested Python function in a single code block.`
- Temperature > 0 means samples are stochastic; report both pooled and macro accuracy, not a single number.
- Grading: gsm8k: numeric match on the final number; math: normalized LaTeX/\boxed{} match with numeric+sympy fallback; humaneval: sandboxed execution of the canonical test harness; truthfulqa: multiple-choice letter (MC1).

## Caveats

- Checker-based grading undercounts correct answers in unusual formats (especially LaTeX in MATH).
- HumanEval solutions execute the canonical tests in a local sandbox (15 s timeout); no hidden tests.
- TruthfulQA is MC1 only.
- Results are the local generator named above; see the cascade report for escalation economics.
