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

## No-corrections control (three tasks, 2026-11-05)

Three arms on `gsm8k-0009`, `0015`, `0038`, same 8192 context, 4096 cap, 3072 thinking budget.
The corrected arm was rerun at the same 5120/1024 VRAM reserve as the no-corrections arm, because the
expert-cache placement changes with the reserve and can change an answer.

| Task | Corrected, 3072 reserve | Corrected, 5120 reserve | No corrections, 5120 | Q2 bridge, 3072 |
|---|---|---|---|---|
| `0009` | FAIL 480 | **PASS 540** | **PASS 540** | **PASS 540** |
| `0015` | FAIL 41 | FAIL 41.4 | FAIL 41 | **PASS 40** |
| `0038` | FAIL 24500 | FAIL 24500 | **PASS 245** | FAIL 24500 |

What this rules out:

- **`0009` is not a correction defect.** The corrected model passes it once the reserve/cache matches;
  the earlier 480 was cache-placement or sampling, not a stable branch error.
- **`0015` is not a correction defect.** It fails with and without corrections; the direct model sums
  the birds as 290 instead of 280. The Q2 bridge gets 40, but it differs in experts, PLE and body, so
  that is not a corrections comparison.
- **`0038` is a units/checker issue.** `24500 cents` and `245 dollars` are the same amount; the
  checker expects the dollar form. The corrected arm answers cents, the no-corrections arm answers
  dollars. This is not a quality difference.

Conclusion: the local control does **not** support “the three failures are the trained corrections.”
Only `0038` changes with the corrections, and both of its answers are mathematically correct. The F16
body control is not needed to explain these three items.

## Unsloth UD-Q2_K_XL of the same FlashNext base (2026-11-05)

Same 30-task suite and task prompts, 8192 context, 4096 cap, 3072 thinking budget. The Unsloth
model is a generic mixed-quant (`IQ2_XXS`/`IQ2_S`/`IQ3_XXS`/`Q2_0` experts, Q8_0/Q6_K body,
`IQ4_NL` PLE) streamed from `/home` by Strata on the same 2×20 GB box.

| Arm | Size | Accuracy | Truncations | Median | p95 |
|---|---:|---:|---:|---:|---:|
| Scion Q6_K | 60.17 GB | 27/30 (90.0%) | 0 | 16.3 s | 59.9 s |
| Q2 bridge | 74.28 GB | 28/30 (93.3%) | 0 | 12.9 s | 73.2 s |
| **Unsloth UD-Q2_K_XL** | **79.6 GB** | **30/30 (100%)** | 0 | 28.6 s | 67.8 s |

Every Unsloth task ended with `finish_reason=stop`.

### Streaming showcase

- 79.6 GB model on 2×20 GB cards: no full-file load; experts are read from the model shards on
  demand (`--mmap-experts`), with the expert cache filling what VRAM allows.
- Display card: **2595 MiB free** with `--vram-reserve-mib 6144` and `--vram-reserve-later-mib 1024`;
  non-display card 876 MiB free (full, by design).
- Average decode **12.09 tok/s**, prompt **44.09 tok/s** across the 30 tasks; total completion
  11,470 tokens.
- File tier per request: 17–79 GB read from storage for 21–119 GB of expert reads; the page cache
  held 83–94% of the decode's reads. Decode waited on disk in every request — this is the streaming
  cost, not a compute limit.

### Reading

The Unsloth quant is **more accurate on this suite by three tasks** than Scion Q6_K and by two than
the Q2 bridge, at 15–20 GB more file size and roughly half the decode speed. Scion's advantage is
size and speed, not absolute accuracy on these 30 tasks; the Q2 bridge is the closest size/accuracy
point to Unsloth.

Caveats: the Unsloth arm ran without MTP/speculation (`--spec 2`, no drafter), while the Scion arms
used spec 4 + MTP; the reserves differ (6144 vs 3072). The 30-task suite is small and cache-sensitive;
the 400-task/PPL numbers on the 176B model card remain the public quality data. Do not turn this into
a “Scion beats Unsloth” claim.
