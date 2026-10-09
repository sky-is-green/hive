#!/bin/bash
# finish-w4-local.sh - local step-4096 export + validation chain (free).
# Run after the W4 fetch lands: qwen4exp-48l-ptq1_0-corr-step4096-ple.gguf.
#   1. export (--use-temp-file, TMPDIR on disk, PLE table last)
#   2. export_check -> ple_export_verify -> gguf_ref_sweep
# Then generation proof separately:
#   ple-48l-load-proof-any.sh <gguf> 8803 step4096
set -euo pipefail
cd /home/penis/Desktop/work
Q=hivebench/artifacts/ternary/moe/qwen4exp
PY=hivebench/.venv/bin/python
CKPT=$Q/pod/remote-gate4000/qwen4exp-corr-r512-g128-step4000-cur05-deployed.pt
GGUF=$Q/qwen4exp-48l-ptq1_0-corr-step4000-ple.gguf
[ -s "$CKPT" ] || { echo "REFUSING: missing $CKPT"; exit 2; }
mkdir -p tmp-export
df -h /home/penis/Desktop/work | tail -1

echo "=== export $(date -u +%FT%TZ)"
TMPDIR=/home/penis/Desktop/work/tmp-export "$PY" scion/moe/qwen4exp_export.py \
    --model-dir /run/media/penis/9EAC1057AC102BF1/flashnext-export \
    --ple-shard-dir /home/penis/Desktop/work/models/qwen38-flashnext-fp8/shards \
    --layers 48 --experts ptq1_0 --body f16 --use-temp-file --ple q4_0 \
    --branches "$CKPT" --branch-dtype f16 --routers replace \
    --adapter-recipe "Flash-Next cur05 step4000 r512 g128 lloyd, deployed" \
    --out "$GGUF"
sha256sum "$GGUF" | tee "$GGUF.sha256"

echo "=== export_check $(date -u +%FT%TZ)"
"$PY" scion/moe/export_check.py "$GGUF"

echo "=== ple_export_verify $(date -u +%FT%TZ)"
"$PY" scion/moe/ple_export_verify.py "$GGUF" models/qwen38-flashnext-fp8 48

echo "=== gguf_ref_sweep $(date -u +%FT%TZ)"
PYTHONPATH=llama-qwen4exp/gguf-py "$PY" scion/moe/gguf_ref_sweep.py "$GGUF"

echo "=== DONE $(date -u +%FT%TZ)"
