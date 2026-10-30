#!/usr/bin/env python
"""LFM3 role SFT with TRL + LoRA (the Liquid recipe, not the ternary path).

This is deliberately separate from ``bonsai2-ternary-forensics/.../rmd_kd.py``:
LFM3 is a transformers checkpoint, so ``--ste --rotate`` does not apply.

The recipe mirrors Liquid's LFM3 SFT notebook:

- LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o, in/out projections and w1/w2/w3
- prompt/completion rows with ``chat_template_kwargs={"enable_thinking": false}``
- completion-only loss, bf16, cosine schedule, lr 2e-4

Run on the non-display card only::

    HIP_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
      ~/.unsloth/studio/unsloth_studio/bin/python experiments/lfm3_sft.py \
        --model-dir ../models-hf/LFM3-1.7B-Base \
        --data experiments/data/role-sft/train.jsonl \
        --out ../train-runs/lfm3-role-sft-v1 \
        --epochs 3 --batch 4 --grad-accum 4 --max-length 2048 \
        --merge
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def token_lengths(tokenizer, rows: list[dict], n: int = 200) -> list[int]:
    lengths = []
    for row in rows[:n]:
        text = tokenizer.apply_chat_template(
            row["prompt"] + row["completion"],
            tokenize=False,
            **row.get("chat_template_kwargs", {}),
        )
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    return sorted(lengths)


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, max(0, round(p * (len(values) - 1))))
    return values[index]


def prepare_rows(tokenizer, rows: list[dict], max_length: int) -> tuple[list[dict], int]:
    """Tokenize prompt/completion pairs and build an exact completion mask.

    The LFM3 tokenizer returns a BatchEncoding from apply_chat_template, which
    makes TRL's prompt-length probe compute a mask over the whole sequence. We
    tokenize explicitly and hand TRL ``input_ids`` + ``completion_mask`` with
    ``skip_prepare_dataset`` so only assistant tokens contribute to the loss.
    """
    prepared: list[dict] = []
    skipped = 0
    for row in rows:
        kwargs = row.get("chat_template_kwargs", {})
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True, **kwargs)
        full_text = tokenizer.apply_chat_template(
            row["prompt"] + row["completion"], tokenize=False, **kwargs)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids:
            skipped += 1
            continue
        input_ids = full_ids[:max_length]
        mask = ([0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids)))[:max_length]
        if len(input_ids) < 4 or 1 not in mask:
            skipped += 1
            continue
        prepared.append({"input_ids": input_ids, "completion_mask": mask})
    return prepared, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=0, help="override epochs with max_steps")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="tokenize the corpus and print length stats, then exit")
    parser.add_argument("--limit", type=int, default=0, help="debug: use only N rows")
    args = parser.parse_args(argv)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("json", data_files=args.data, split="train")
    if args.limit:
        raw = raw.select(range(min(args.limit, len(raw))))
    rows = [dict(row) for row in raw]

    lengths = token_lengths(tokenizer, rows, n=min(len(rows), 500))
    stats = {
        "rows": len(rows),
        "tokens": {
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": lengths[-1] if lengths else 0,
            "over_max_length": sum(1 for value in lengths if value > args.max_length),
        },
    }
    print(json.dumps(stats, indent=2), flush=True)
    if args.dry_run:
        return 0

    prepared, skipped = prepare_rows(tokenizer, rows, args.max_length)
    if not prepared:
        raise SystemExit("no usable rows after tokenization")
    print(f"prepared {len(prepared)} rows ({skipped} skipped)", flush=True)
    from datasets import Dataset

    train_dataset = Dataset.from_list(prepared)

    started = time.time()
    print(f"loading {args.model_dir} on {args.device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(args.device)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj", "out_proj", "w1", "w2", "w3",
        ],
        task_type="CAUSAL_LM",
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(out),
        max_steps=args.steps if args.steps > 0 else -1,
        num_train_epochs=args.epochs if args.steps <= 0 else 1.0,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        max_length=args.max_length,
        logging_steps=args.log_steps,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=max(1, args.save_steps),
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        packing=False,
        completion_only_loss=True,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    print("training", flush=True)
    result = trainer.train()
    elapsed = time.time() - started

    adapter_dir = out / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    manifest = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "data": str(Path(args.data).resolve()),
        "rows": len(rows),
        "prepared_rows": len(prepared),
        "skipped_rows": skipped,
        "epochs": args.epochs,
        "steps": args.steps or None,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch * args.grad_accum,
        "lr": args.lr,
        "max_length": args.max_length,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": 0.05},
        "device": args.device,
        "seed": args.seed,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda_devices": [
            {"index": i, "name": torch.cuda.get_device_name(i),
             "total_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)}
            for i in range(torch.cuda.device_count())
        ],
        "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "elapsed_s": round(elapsed, 1),
        "train_loss": result.training_loss,
        "log_history": trainer.state.log_history,
        "token_stats": stats,
        "adapter": str(adapter_dir),
        "merged": None,
    }

    if args.merge:
        merged_dir = out / "merged"
        print(f"merging adapter into {merged_dir}", flush=True)
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        manifest["merged"] = str(merged_dir)

    (out / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("DONE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
