# Scion 176B: direct Q6_K vs the Q2 bridge (30 tasks, 2026-11-05)

Same protocol for both arms: 30-task test split, 8192 context, `max_tokens=4096`,
`thinking_budget_tokens=3072`, `answer_room` on, temperature 0.1, one sample. One server at a
time on the 2×7900 XT box. Both runs used the `scion-0.2.10` engine.

| Arm | Accuracy | Truncations | Median | p95 |
|---|---:|---:|---:|---:|
| Direct Scion Q6_K | **27/30 (90.0%)** | 0 | 16.3 s | 59.9 s |
| Q2 bridge | **28/30 (93.3%)** | 0 | 12.9 s | 73.2 s |

Every task ended with `finish_reason=stop` on both arms.

| Benchmark | Direct Q6_K | Q2 bridge |
|---|---:|---:|
| GSM8K (12) | 9/12 | 11/12 |
| HumanEval (6) | 6/6 | 5/6 |
| Math (8) | 8/8 | 8/8 |
| TruthfulQA (4) | 4/4 | 4/4 |

## Disagreements

| Task | Direct | Q2 | Note |
|---|---|---|---|
| `gsm8k-0009` | FAIL 480 | PASS 540 | Direct computed `150 + 390` as 480 |
| `gsm8k-0015` | FAIL 41 | PASS 40 | Direct summed the birds as 290 instead of 280 |
| `humanEval-0001` | PASS | FAIL | Direct's code passed; Q2's did not |
| `gsm8k-0038` | FAIL 24500 | FAIL 24500 | Both answered cents; the checker expects 245 dollars |

## Reading

- The one-task difference is within the noise of a 30-task, one-sample, temperature-0.1 run.
  There is no basis to retire the Q2 bridge, and no basis to claim the direct Q6_K container is
  better.
- The old pairing “Q6_K 93.3% vs Q2 76.7%” is **not comparable**: Q6_K ran with a 2048-token cap and
  Q2 with 4096. Do not use it in public material.
- `gsm8k-0038` is a unit/answer-format issue (cents vs dollars), not a quantization difference; both
  arms fail it.
- The Q2 bridge is a Strata-compatible repack of the Q6_K release, **not** an official Scion Q2 and
  not ISTA’s GSQ-RCO Q2_0.

## F16 teacher control

Attempted after the A/B. The F16 body is not a native quantized body, and Strata’s native-pack
verify path requires native dense/head/BF16 projections; the engine starts but rejects prefill with
`blk.0.attn_qkv.weight has no native GGUF blocks`. The F16 control was therefore not run on this box.
It would need either engine work for a packed body in a native expert pack, or the Prism
`qwen4exp` fork on a host with at least 64 GB RAM (this box has ~30 GB). The 62 GB F16 GGUF is kept
for that future run.

## Artifacts

- `experiments/model-matrix/ab-scion-q6k-v0210-4096-tb3072.json`
- `experiments/model-matrix/ab-scion-q2-v0210-4096-tb3072.json`
- `docs/SCION-DIRECT-AB.md` — the runbook
- `experiments/compare_scion_direct_q2.py` — the comparison
