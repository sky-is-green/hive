# LFM3 family and where its data lives

Source of truth: https://huggingface.co/blog/LiquidAI/lfm3

The LFM3 release is one 2B-class backbone with modal branches. The official
collection is https://huggingface.co/collections/LiquidAI/lfm3.

| model | input | local status |
|---|---|---|
| LFM3-1.7B | text, 256k context | **live** on `:8155`; Q4_K_M now in the library |
| LFM3-1.7B-Base | text base | **downloaded** to `models-hf/LFM3-1.7B-Base`; fine-tune base |
| LFM3-VL-2B | text, images, video | Q8_0 + mmproj downloaded; live on `:8160` with b11960 |
| LFM3-Audio-2B | text, audio, speech | not downloaded |
| LFM3-Omni-2B | text, images, video, audio | not downloaded |
| d1-2B | text, images, video → calibrated decisions | Q8_0 + mmproj downloaded; custom decision API, not chat |
| d1-omni-2B | decisions + audio | not downloaded |

## Role fit

- **mechanics / RAG / cheap chat / simple structured plans** → LFM3-1.7B.
  Hard multi-worker planning belongs to Scion-35B (15/17 vs LFM3 11/17).
- **vision** → LFM3-VL-2B on `:8160` with b11960.
- **judge / router / gate** → d1-2B, but it needs a decision sidecar that speaks
  its `system_one` API; it is not an OpenAI chat endpoint.
- **audio / omni roles** → LFM3-Audio-2B / LFM3-Omni-2B, not mapped yet.

## Engine status

llama.cpp b11960 / v0.8.0 supports LFM3-VL and d1. Both were validated locally
with the Ubuntu Vulkan x64 build and are now managed by Hive's `lfm3` backend:
LFM3-VL-2B on `:8160`, d1-2B on `:8161`. See `LFM3-ENGINE.md`.

The Chat loader hides `architecture=lfm3` library entries until the managed
backend is installed; they remain visible on Models with their upstream links.

The `d1-2B-GGUF` files are downloaded too. d1 is a decision model: its API takes
a state plus a question map and returns calibrated answers in one pass. The
preview now has a d1 decision panel and `/api/decision` adapter; it is not a
chat model and does not belong in the normal chat selector.

## Fine-tuning

Liquid publishes the correct path for LFM3:

- LFM3-1.7B-Base for post-training.
- Cookbook notebooks: https://github.com/Liquid4All/cookbook/tree/main/finetuning/notebooks
- Methods: SFT, DPO, GRPO, CPT — single-GPU friendly.

This is a different recipe from the ternary `--ste --rotate` forensics runner.
The 20-step Qwen3-1.7B smoke validated the runner mechanics; the first real
LFM3 run should use `models-hf/LFM3-1.7B-Base` with Liquid's TRL recipe.

## Local files

The LFM3 GGUF downloads live under `models-p310` and are linked into
`/srv/hive-ai/models/gguf`:

```text
LFM3-1.7B-Q4_K_M.gguf          1.02 GB
LFM3-VL-2B-Q8_0.gguf           1.72 GB
mmproj-LFM3-VL-2B-Q8_0.gguf    0.19 GB
d1-2B-Q8_0.gguf                1.72 GB
mmproj-d1-2B-Q8_0.gguf         0.19 GB
```
