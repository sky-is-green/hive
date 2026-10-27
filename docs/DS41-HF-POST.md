# DeepSeek-V4.1-Flash-GSQ on one RX 7900 XT 20 GB + 31.7 GiB: 121-127 tok/s prefill, 16 tok/s decode at 6K, 23 s for a 648-token turn

Hi all, I have been running [DeepSeek-V4.1-Flash-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored) on the small end of the hardware scale and thought the numbers were worth posting. Most of the recent reports are 4x/8x or 125+ GiB.

## Hardware
- 1x RX 7900 XT 20 GB (the spare card, the desktop is on the other GPU)
- Ryzen 7 7800X3D, 31.7 GiB DDR4
- Crucial P310 NVMe. The 152 GB GSQ model and both Engram layers are read-only bind mounts.

## Model
- [DeepSeek-V4.1-Flash-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored), 152 GB, 2.25 bpw (42.8% ternary PTQ1_0, 39.2% Q2_0, 18.0% Q3_K; dense Q4_K)
- 40 layers x 384 experts, top-6; Engram L1+L14

## Engine
- [Strata-DSV4.1](https://github.com/Indras-Mirror/Strata-DSV4.1) (HIP), 32K context, KV q8
- 4 GiB RAM arena, 8,192 prompt batch, 1,024 prefill chunk
- `STRATA_SERVE_STEP_MAX=0` (exact layer-major batch for every prompt)
- `STRATA_WAVE_THREADS=16`, `STRATA_WAVE_PARTS=16`, 1.5 GiB prestage, `--pf-b 0`
- D3 at its default; DSpark off

## Results (fixed probes, temperature 0)
| workload | result |
|---|---|
| 6,110-token prefill | 51.7 s, 126.5 tok/s (cold arms 117.8-121.3) |
| 128-token decode at 6.1k | 16.2 tok/s |
| 648-token new turn | 23.3 s |
| 341-token new turn | 24.1 s |
| 73-token new turn | 18.0 s |
| 16-token follow-up on the 6.1k prefix | 7.6 s (6,115 tokens cached) |
| my local agent, 5,734-token prompt | 67 s, 85.6 tok/s; 128 thinking at 17.6 tok/s |
| my local agent tool-result follow-up | 5 s prefix reuse + 139 tokens at 15.1 tok/s |

## What mattered
- With a 4 GiB RAM arena the exact pass path is only about 2.3 tok/s, so the batch/pass crossover moves to about 10-16 tokens. `STRATA_SERVE_STEP_MAX=0` turned 341/648-token tool turns from 129/256 s to 24/23 s, while the 6.1k prefill stayed at about 52 s. That is the part that matters for agent use.
- The original `--prompt-batch 1` prefill was 292 s (19.6 tok/s). 8,192 is a 5.5x gain.
- `--prefill-chunk 1024` is the max that handles the 6.1k prompt here: 2,048 returns 503; 4,096 fails a 17.5 GiB dense graph allocation. At 1,024 the 648-token turn is one dense/expert chunk instead of two.
- The wave reader at 16 threads / 16 parts added 3-7%: 120.8 / 121.3 / 121.2 for 16/8, 16/16 and 32/16. The final confirmation was 126.5.
- `--pf-b 1.43` regressed to 99.3 / 11.2. `STEP_MAX=20` helped the 16-token turn but dropped the 128-token decode to 12.4. D3 off was neutral. I left them off.

## Caveats
- The agent prompt is 5,734 tokens of tool/system content and still costs 67 s. Its routing is wider than the repeated code-review probe. Trimming the agent prompt is the next real lever.
- The direct 6.1k is already at the 4 GiB arena/file tier limit. The 230-244 tok/s reference in [the engine notes](https://github.com/Indras-Mirror/Strata-DSV4.1) needs 128 GiB RAM and a 60-75 GiB arena.
- DSpark on this HIP build fails with `DSpark step: k out of range`. I see related reports in [#72](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/72) and [#28](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/28), so it is off.
- The GSQ model is 2.25 bpw. Quality is a separate trade, and this post is about the speed ceiling of this quant on this box.

## Links
- [Model](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored)
- [Engine](https://github.com/Indras-Mirror/Strata-DSV4.1)
- [24 GB VRAM / 64 GB RAM thread](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/discussions/2)
- [Single DGX Spark thread](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/70)

If anyone is running a 30-32 GiB box, let me know and I am happy to compare configs. I am especially interested in whether a larger RAM arena moves the 5.7k agent prompt, and whether DSpark can be fixed on HIP.
