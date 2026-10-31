#!/usr/bin/env python
"""Package a named Hive specialist for a private Hugging Face release.

Unlike ``package_role_specialist.py`` (one role), this script takes a release
name and a list of roles, so Conductor/Reader/Auditor/Dispatcher/Clerk keep
their product names and evaluation tables.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LICENSE = REPO.parent / "models-hf" / "LFM3-1.7B-Base" / "LICENSE"


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


def eval_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    overall = data.get("overall") or {}
    role = next(iter((data.get("by_role") or {}).keys()), "")
    return {"role": role or path.stem.replace("eval-", "").replace("-gguf", ""),
            "accuracy": overall.get("accuracy"),
            "n": overall.get("n"),
            "median_s": overall.get("median_s"),
            "gguf": path.stem.endswith("-gguf")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--tagline", required=True)
    parser.add_argument("--roles", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--eval", action="append", default=[])
    parser.add_argument("--gguf", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--hero", default="", help="optional local hero PNG")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    run = Path(args.run).resolve()
    out = Path(args.out).resolve() if args.out else run / f"release-{args.repo.split('/')[-1]}"
    out.mkdir(parents=True, exist_ok=True)
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]

    copy_tree(run / "adapter", out / "adapter")
    copy_tree(run / "merged", out / "model")
    cfg_path = out / "adapter" / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["base_model_name_or_path"] = "LiquidAI/LFM3-1.7B-Base"
    cfg_path.unlink()
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    gguf = Path(args.gguf).resolve() if args.gguf else next(iter(sorted(run.glob("*-Q4_K_M.gguf"))))
    gguf_name = f"{args.repo.split('/')[-1]}-Q4_K_M.gguf"
    link_or_copy(gguf, out / "gguf" / gguf_name)

    evals = [Path(p).resolve() for p in args.eval]
    for path in evals:
        link_or_copy(path, out / "eval" / path.name)
    manifest = json.loads((run / "run-manifest.json").read_text())
    steps = max((r.get("step", 0) for r in manifest.get("log_history", [])), default=0)

    rows = []
    for path in evals:
        item = eval_summary(path)
        rows.append(item)
    eval_table = "\n".join(
        f"| {r['role'] or 'all'} | {'GGUF' if r['gguf'] else 'HF'} | "
        f"{r['accuracy']:.3f} | {r['n']} | {r['median_s']:.3f} s |"
        for r in rows
    )
    tags = "\n".join(f"- {role}" for role in ["lfm3", "hive", *roles, "lora", "gguf"])
    limitations = args.notes or (
        "The model is trained for the Hive role contract; other prompts are out "
        "of scope. Use a deterministic checker or guard in production."
    )
    hero = ""
    if args.hero:
        hero_path = Path(args.hero).resolve()
        link_or_copy(hero_path, out / "assets" / "hero.png")
        hero = ('<p align="center">\n'
                f'  <img src="assets/hero.png" alt="{args.title}" width="360">\n'
                '</p>\n\n')

    readme = f"""---
base_model: LiquidAI/LFM3-1.7B-Base
library_name: transformers
pipeline_tag: text-generation
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM3-1.7B-Base/blob/main/LICENSE
tags:
{tags}
---

# {args.title}

{hero}{args.tagline}

One adapter covers the **{', '.join(roles)}** seat(s) for the Hive local
control plane, built from
[`LiquidAI/LFM3-1.7B-Base`](https://huggingface.co/LiquidAI/LFM3-1.7B-Base).
It is not a general chat model.

## Evaluation

Held-out Hive role suites, greedy decoding:

| role | path | accuracy | n | median |
|---|---|---:|---:|---:|
{eval_table}

## Training

- Base: `LiquidAI/LFM3-1.7B-Base`
- Method: TRL SFT + LoRA (r=16, alpha=32, dropout 0.05)
- Data: synthetic, checker-validated Hive role rows; no user data
- Format: LFM3 chat template with `enable_thinking=false`; assistant-only loss
- Run: {manifest.get('rows')} rows, {manifest.get('epochs')} epochs, {steps} steps
- Peak VRAM: {manifest.get('peak_vram_gib')} GiB on one 20 GB AMD card

## Usage

Merged transformers model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{args.repo}"
model = AutoModelForCausalLM.from_pretrained(
    repo, subfolder="model", dtype="bfloat16", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(
    repo, subfolder="model", trust_remote_code=True)
```

PEFT adapter:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM3-1.7B-Base", dtype="bfloat16", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "{args.repo}", subfolder="adapter")
```

GGUF with llama.cpp:

```sh
llama-server -m gguf/{gguf_name} --alias {args.alias} \\
  --host 127.0.0.1 --port 8156 -ngl 999 -c 8192 -np 2 \\
  --jinja --reasoning off --no-webui
```

`--reasoning off` is required for this LFM3 template.

## Limitations

{limitations}

## License and attribution

This is a derivative of
[`LiquidAI/LFM3-1.7B-Base`](https://huggingface.co/LiquidAI/LFM3-1.7B-Base),
released under the **LFM Open License v1.0** (see `LICENSE`). The base model is
by Liquid AI. Commercial use is subject to the license's revenue threshold.

Modified files: the LoRA adapter, merged weights, and GGUF conversion. See
`NOTICE`.
"""
    (out / "README.md").write_text(readme)
    (out / "NOTICE").write_text(
        "This model is a derivative of LiquidAI/LFM3-1.7B-Base.\n"
        "Base model: https://huggingface.co/LiquidAI/LFM3-1.7B-Base\n"
        "License: LFM Open License v1.0 (see LICENSE).\n"
        f"Modifications: LoRA SFT for the Hive {', '.join(roles)} seat(s), "
        "merged weights, and GGUF Q4_K_M conversion.\n"
    )
    link_or_copy(LICENSE, out / "LICENSE")
    print(f"release folder: {out}")
    print("\nUpload:")
    print(
        "python - <<'PY'\n"
        "from huggingface_hub import HfApi\n"
        f"api=HfApi(); api.create_repo('{args.repo}', repo_type='model', private=True, exist_ok=True)\n"
        f"api.upload_folder(folder_path='{out}', repo_id='{args.repo}', "
        "commit_message='Add Hive specialist')\n"
        "PY"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
