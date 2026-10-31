# Training commands and readiness

## LFM3 path — working now

The LFM3 base is a transformers model, not a ternary GGUF. Use the dedicated
script, not `rmd_kd.py`.

Smoke that completed on 2026-10-30:

```sh
HIP_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
  /home/penis/.unsloth/studio/unsloth_studio/bin/python \
  experiments/lfm3_cpt.py \
  --model-dir /run/media/.../work/models-hf/LFM3-1.7B-Base \
  --corpus /run/media/.../work/corpora/wikitext2-train.txt \
  --out /run/media/.../work/train-smoke/lfm3-cpt-smoke \
  --steps 10 --seq 256 --lr 5e-5 --device cuda:0
```

Result: 10 steps in ~7 s, loss 2.6–3.6, checkpoint and full model saved.
This is a pipeline smoke (CPT on WikiText), not a quality run.

The LFM3 cookbook is the path for real SFT/DPO/GRPO/CPT:
https://github.com/Liquid4All/cookbook/tree/main/finetuning/notebooks

Environment used: transformers **5.22.1**, trl **0.23.1**, torch
**2.11.0+rocm7.2**. The Unsloth env was upgraded to transformers 5.22.1 for
LFM3; Unsloth itself declares `<=5.17.0`, so keep the ternary env separate if
Unsloth is needed.

## Real role SFT run (2026-10-30)

The first real LFM3 fine-tune is complete: 3,266 checker-validated role rows,
615 LoRA steps on the non-display card, held-out role score 0.833 vs 0.653 for
the base checkpoint. Full report: `docs/round5/FINETUNE.md`.

New entry points:

- `experiments/build_role_sft.py` — synthetic role corpus; every emitted pair
  is validated with the role-suite checkers. The real 74 prompts are held out.
- `experiments/lfm3_sft.py` — TRL SFT + LoRA with an explicit completion mask
  (the LFM3 tokenizer returns a `BatchEncoding`, which otherwise makes TRL
  train on the prompt too).
- `experiments/eval_role_sft.py` — in-process HF evaluation on the 74-task role
  suite; no llama.cpp server needed.

```sh
HIP_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
  ~/.unsloth/studio/unsloth_studio/bin/python experiments/lfm3_sft.py \
    --model-dir ../models-hf/LFM3-1.7B-Base \
    --data experiments/data/role-sft-v3/train.jsonl \
    --out ../train-runs/lfm3-role-sft-v3 \
    --epochs 3 --batch 8 --grad-accum 2 --max-length 512 --merge
```

Artifacts: `work/train-runs/lfm3-role-sft-v3/{adapter,merged,*.gguf}`. The GGUF
converts cleanly with llama.cpp b11960 (`conversion/lfm3.py`) and matches the
HF score on the role suite.

## Controller specialist (hyper-specialisation)

The controller is the first one-role adapter. Corpus, wider suite, comparison
and deployment: `docs/round5/CONTROLLER-SPECIALIST.md`.

```sh
python experiments/build_controller_suite.py
python experiments/build_controller_sft.py --out experiments/data/controller-sft-v2
python experiments/build_controller_gap.py --out experiments/data/controller-sft-v3

HIP_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
  ~/.unsloth/studio/unsloth_studio/bin/python experiments/lfm3_sft.py \
    --model-dir ../models-hf/LFM3-1.7B-Base \
    --data experiments/data/controller-sft-v3/train.jsonl \
    --out ../train-runs/controller-sft-base-v3 \
    --epochs 3 --batch 4 --grad-accum 4 --lr 2e-4 --max-length 512 --merge
```

Result: 54/60 on the wider controller suite in HF (57/60 GGUF) vs 50/60 for
Scion-35B, 9/9 on the original controller tasks, 0.25 s median through the GGUF
server. This controller-only checkpoint was live on `:8156` until the
controller+planner promotion below; it remains the rollback artifact.

The private HF repo now hosts the multirole checkpoint:
https://huggingface.co/SkyIsNotGreen/lfm3-1.7b-controller-planner-v1

The packaging script is reusable for the next role:

```sh
python experiments/package_role_specialist.py \
  --run ../train-runs/controller-sft-base-v3 --role controller \
  --repo <new-destination-repo> \
  --eval ../train-runs/controller-sft-base-v3/eval-specialist.json \
  --gguf-eval ../train-runs/controller-sft-base-v3/eval-gguf.json \
  --suite "Hive 60-task controller suite"
```

## Controller + planner multi-role

The first retention test: add planner to the controller specialist and check
whether controller goes down. Full report: `docs/round5/MULTIROLE-CP.md`.

```sh
python experiments/build_planner_suite.py
python experiments/build_planner_sft.py --out experiments/data/planner-sft-v1
python experiments/build_multirole_sft.py --out experiments/data/multirole-cp-v1

HIP_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
  ~/.unsloth/studio/unsloth_studio/bin/python experiments/lfm3_sft.py \
    --model-dir ../train-runs/controller-sft-base-v3/merged \
    --data experiments/data/multirole-cp-v1/train.jsonl \
    --out ../train-runs/multirole-cp-v1 \
    --epochs 2 --batch 4 --grad-accum 4 --lr 1e-4 --max-length 512 --merge
```

Result: controller **59/60** (was 54/60), original 9/9; planner **69/77 HF**
(16/17 original), **66/77 GGUF** (15/17 original). No interference; positive
transfer. The multirole checkpoint is live on `:8156` as
`controller-1.7b-sft`, with Scion as fallback.

Published (private): https://huggingface.co/SkyIsNotGreen/lfm3-1.7b-controller-planner-v1


## Ternary path — Qwen/Scion/Bonsai recipe

The runner exists at:

```text
/run/media/.../work/bonsai2-ternary-forensics/scripts/pilot/rmd_kd.py
```

Smoke that completed earlier:

```sh
HIP_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false \
  ~/.unsloth/studio/unsloth_studio/bin/python \
  scripts/pilot/rmd_kd.py \
  --model-dir /run/media/.../work/models-hf/Qwen3-1.7B \
  --corpus /run/media/.../work/corpora/wikitext2-train.txt \
  --out /run/media/.../work/train-smoke/qwen17-ste-rotate-smoke \
  --ste --rotate --steps 20 --seq 512 --batch 1 --lr 5e-5 \
  --eval-windows 2 --eval-regions 2 \
  --device cuda:0 --teacher-device cuda:1 --seed 1337 \
  --log-every 5 --project-every 10 --save-every 10
```

This is the ternary `--ste --rotate` path, not the LFM3 path. It writes
`student.pt`, eval regions, and `rmd-report.json`.

## What is still missing

1. **Training page integration.** The page builds a manifest and can launch a
   process, but the live Hive service still has no `HIVE_TERNARY_RUNNER` and
   cannot read `/home/penis/.unsloth`. Run these commands as `penis` until the
   runner/env is wired or a shared ROCm venv is installed.
2. **Own-model HF bases.** Scion-35B and Bonsai 27B are local as GGUF only.
   For real fine-tuning, download the trainable HF base:
   - `empero-ai/Qwen3.8-35B-A3B-Distill` (71 GB, 21 shards) — the 35B base.
   - `Qwen/Qwen3.8-27B` — the 27B base.
   The Scion/Bonsai GGUF files are for serving, not training.
3. **Real preference data.** The role corpus is synthetic and checker-validated;
   exported human A/B votes and real role traces are still missing. Those rows
   are needed to move the judge beyond the 27B+ MCQ ceiling.
4. **GPU window.** Training on the non-display card works with DS4.1 resident
   (peak 14.03 GiB). Scion-35B must be stopped if it is resident; after this run
   the live stack was restored with Scion back on the non-display card.
5. **Recipe choice.** LFM3 uses the Liquid/TRL cookbook; Scion/Bonsai ternary
   uses `rmd_kd.py --ste --rotate`. Do not run the ternary recipe on LFM3.
