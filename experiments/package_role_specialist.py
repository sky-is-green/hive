#!/usr/bin/env python
"""Package a role-specialist train run for a Hugging Face release.

Builds a single folder with the LoRA adapter, merged model, Q4_K_M GGUF,
evaluation JSONs, the LFM license, a NOTICE, and a generated model card. The
same script is meant to be reused for the next role specialist.

    python experiments/package_role_specialist.py \
      --run ../train-runs/controller-sft-base-v3 \
      --role controller \
      --repo SkyIsNotGreen/lfm3-controller-1.7b-sft \
      --eval ../train-runs/controller-sft-base-v3/eval-specialist.json \
      --out ../train-runs/controller-sft-base-v3/release

Upload is deliberately not automatic; the printed command creates the repo and
uploads the folder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LICENSE = REPO_ROOT.parent / "models-hf" / "LFM3-1.7B-Base" / "LICENSE"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return
        except OSError:
            pass
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        if item.is_file():
            link_or_copy(item, dst / item.relative_to(src))


def read_eval(path: Path | None) -> dict:
    if not path or not path.is_file():
        return {}
    return json.loads(path.read_text())


def card(args, eval_data: dict, gguf_name: str | None,
         gguf_eval: dict | None = None) -> str:
    overall = (eval_data.get("overall") or {})
    by_role = (eval_data.get("by_role") or {})
    score = ""
    if overall:
        score = (f"- HF merged bf16 on the held-out {args.suite}: "
                 f"**{overall.get('accuracy', 0):.3f}** ({overall.get('n', 0)} tasks), "
                 f"median **{overall.get('median_s', 0):.3f} s**.\n")
    if gguf_eval and gguf_eval.get("overall"):
        g = gguf_eval["overall"]
        score += (f"- GGUF Q4_K_M on the same suite: **{g.get('accuracy', 0):.3f}** "
                  f"({g.get('n', 0)} tasks), median **{g.get('median_s', 0):.3f} s**.\n")
    extra = ""
    if args.baseline:
        extra = f"- Baseline for comparison: {args.baseline}\n"
    role_rows = ""
    for role, stats in by_role.items():
        role_rows += (f"| {role} | {stats.get('n', 0)} | "
                      f"{stats.get('accuracy', 0):.3f} | "
                      f"{stats.get('median_s', 0):.3f} s |\n")
    eval_table = ""
    if role_rows:
        eval_table = ("\n\n| role | n | accuracy | median |\n"
                      "|---|---:|---:|---:|\n" + role_rows + "\n")
    gguf = ""
    if gguf_name:
        gguf = (f"GGUF: `gguf/{gguf_name}` (Q4_K_M). Load with llama.cpp:\n\n"
                "```sh\n"
                f"llama-server -m gguf/{gguf_name} --alias {args.alias} "
                "--host 127.0.0.1 --port 8156 -ngl 999 -c 8192 -np 2 "
                "--jinja --reasoning off --no-webui\n"
                "```\n\n"
                "`--reasoning off` is required for this LFM3 template; without it "
                "the GGUF answer shape diverges from the HF model.\n")
    return f"""---
base_model: {args.base_model}
library_name: peft
pipeline_tag: text-generation
license: other
license_name: lfm1.0
license_link: https://huggingface.co/{args.base_model}/blob/main/LICENSE
tags:
- lfm3
- hive
- {args.role}
- lora
- gguf
---

# {args.title}

A {args.role}-specialised LoRA of `{args.base_model}` for the Hive local
control plane. It is trained on one role and one JSON output contract; it is
not a general chat model.

## Evaluation

{score}{extra}{eval_table}The evaluation suite is held out from training. The
model is compared with the resident Scion-35B and with a generalist 1.7B SFT
checkpoint; the specialist is deployed as a canary with Scion as fallback.

## Training

- Base: `{args.base_model}`
- Method: TRL SFT + LoRA (r=16, alpha=32, dropout 0.05; q/k/v/o, in/out, w1/w2/w3)
- Data: synthetic, checker-validated {args.role} tasks; no user data
- Format: LFM3 chat template with `enable_thinking=false`; assistant-only loss
- Hardware: one 20 GB AMD card (ROCm), bf16

## Files

- `adapter/` — PEFT LoRA adapter
- `model/` — merged bf16 transformers model
- `gguf/` — Q4_K_M quantisation
{'- `eval/` — held-out evaluation JSON' if eval_data else ''}

## Usage

Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("{args.repo}", subfolder="model",
                                             dtype="bfloat16", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("{args.repo}", subfolder="model",
                                    trust_remote_code=True)
```

{gguf}
## Limitations

- Narrow role model: it is not a replacement for a 27B+ generalist.
- A few tight six-task schedules still miss a deadline or dependency. Add a
  deterministic assignment/schedule guard in front of it for production use.
- The trained output contract is the Hive {args.role} schema; other prompts are
  out of scope.

## License and attribution

This is a derivative of [`{args.base_model}`](https://huggingface.co/{args.base_model}),
released under the **LFM Open License v1.0** (see `LICENSE`). The base model is
by Liquid AI. Commercial use is subject to the license's revenue threshold;
review the license before commercial deployment. Modified files are the adapter,
merged weights, and GGUF conversion; no changes were made to the base license
text.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--role", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--base-model", default="LiquidAI/LFM3-1.7B-Base")
    parser.add_argument("--alias", default="controller-1.7b-sft")
    parser.add_argument("--eval", default="")
    parser.add_argument("--gguf-eval", default="")
    parser.add_argument("--suite", default="role suite")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--license", default=str(DEFAULT_LICENSE))
    args = parser.parse_args()

    run = Path(args.run).resolve()
    out = Path(args.out).resolve() if args.out else run / "release"
    out.mkdir(parents=True, exist_ok=True)
    args.title = args.title or f"LFM3-1.7B {args.role.title()} Specialist"

    adapter = run / "adapter"
    merged = run / "merged"
    if adapter.is_dir():
        copy_tree(adapter, out / "adapter")
        adapter_cfg = out / "adapter" / "adapter_config.json"
        if adapter_cfg.is_file():
            cfg = json.loads(adapter_cfg.read_text())
            cfg["base_model_name_or_path"] = args.base_model
            adapter_cfg.unlink()  # break the hardlink before editing the release copy
            adapter_cfg.write_text(json.dumps(cfg, indent=2) + "\n")
    if merged.is_dir():
        copy_tree(merged, out / "model")

    ggufs = sorted(run.glob("*-Q4_K_M.gguf"))
    gguf_name = None
    if ggufs:
        gguf_name = ggufs[0].name
        link_or_copy(ggufs[0], out / "gguf" / gguf_name)

    eval_path = Path(args.eval).resolve() if args.eval else None
    eval_data = read_eval(eval_path)
    gguf_eval_path = Path(args.gguf_eval).resolve() if args.gguf_eval else None
    gguf_eval = read_eval(gguf_eval_path)
    if eval_path and eval_path.is_file():
        (out / "eval").mkdir(exist_ok=True)
        link_or_copy(eval_path, out / "eval" / eval_path.name)
        for extra in run.glob("eval-*.json"):
            if extra.resolve() != eval_path:
                link_or_copy(extra, out / "eval" / extra.name)

    license_src = Path(args.license)
    if license_src.is_file():
        link_or_copy(license_src, out / "LICENSE")
    (out / "NOTICE").write_text(
        "This model is a derivative of LiquidAI/LFM3-1.7B-Base.\n"
        "Base model: https://huggingface.co/LiquidAI/LFM3-1.7B-Base\n"
        "License: LFM Open License v1.0 (see LICENSE).\n"
        f"Modifications: LoRA SFT for the Hive {args.role} role, merged weights, "
        "and GGUF Q4_K_M conversion.\n"
    )
    (out / "README.md").write_text(card(args, eval_data, gguf_name, gguf_eval))

    print(f"release folder: {out}")
    for path in sorted(out.iterdir()):
        print(" ", path.name)
    print("\nUpload:")
    print(
        "python - <<'PY'\n"
        "from huggingface_hub import HfApi\n"
        f"api = HfApi()\n"
        f"api.create_repo('{args.repo}', repo_type='model', private=False, exist_ok=True)\n"
        f"api.upload_folder(folder_path='{out}', repo_id='{args.repo}', "
        "commit_message='Add LFM3 controller specialist')\n"
        "PY"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
