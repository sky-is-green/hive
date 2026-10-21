#!/usr/bin/env bash
# Targeted FP-checkpoint download for an N-layer prefix (W1/W3 local dry runs).
#
# The full Qwen3.8-35B-A3B-Distill checkpoint is 21 shards / ~66 GB, which is
# why $MOE_ARTIFACTS/empero-hf went missing after the rental cleanup.  But
# `qwen35_moe_proxy.load_prefix` skips absent shards with a warning, and
# empero-index.json maps every tensor to its shard -- so a 4-layer prefix only
# needs embedding + layers 0-3 + final norm + lm_head, which is 4 shards / ~13 GB.
#
#   ./fetch_prefix_shards.sh            # 4 layers (the W1/W3 prefix depth)
#   ./fetch_prefix_shards.sh 12         # deeper prefix
#   ./fetch_prefix_shards.sh 4 --plan   # print the plan, download nothing
#
# Re-runnable: curl -C - resumes, so an interrupted fetch just continues.
set -euo pipefail

DEST="${DEST:-<work>/hivebench/artifacts/ternary/moe/empero-hf}"
BASE="${BASE:-https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill/resolve/main}"
INDEX="${INDEX:-<work>/hivebench/artifacts/ternary/moe/empero-index.json}"
LAYERS="${1:-4}"
PLAN="${2:-}"
PY="${PY:-<work>/hivebench/.venv/bin/python}"

# Small files the loader needs regardless of depth: config, tokenizer, index.
META=(config.json tokenizer.json tokenizer_config.json generation_config.json
      model.safetensors.index.json)

# Tensors load_prefix pulls for an N-layer prefix: language-model embed + final
# norm + layers < N, plus lm_head.  Everything else (mtp.*, deeper layers) is
# filtered out, so their shards never have to land on disk.
read -r -a SHARDS < <("$PY" - "$INDEX" "$LAYERS" <<'EOF'
import json, sys
index, n_layers = sys.argv[1], int(sys.argv[2])
weight_map = json.load(open(index))["weight_map"]
need = set()
for key, shard in weight_map.items():
    if key.startswith("model.language_model."):
        stripped = key[len("model.language_model."):]
        if stripped.startswith("layers."):
            if int(stripped.split(".")[1]) < n_layers:
                need.add(shard)
        elif stripped in ("embed_tokens.weight", "norm.weight"):
            need.add(shard)
    elif key == "lm_head.weight":
        need.add(shard)
print(" ".join(sorted(need)))
EOF
)

echo "prefix depth : $LAYERS layers"
echo "destination  : $DEST"
echo "shards       : ${SHARDS[*]}"
if [ "$PLAN" = "--plan" ]; then
  echo "meta         : ${META[*]}"
  echo "(plan only; nothing downloaded)"
  exit 0
fi

mkdir -p "$DEST"
for f in "${META[@]}"; do
  if [ ! -s "$DEST/$f" ]; then
    echo "==> $f"
    curl -sL --fail "$BASE/$f" -o "$DEST/$f" || echo "warn: $f failed"
  fi
done

for s in "${SHARDS[@]}"; do
  echo "==> $s ($(du -h "$DEST/$s" 2>/dev/null | cut -f1 || echo new))"
  curl -L --fail -C - "$BASE/$s" -o "$DEST/$s"
done
echo ALLDONE
du -sh "$DEST"
