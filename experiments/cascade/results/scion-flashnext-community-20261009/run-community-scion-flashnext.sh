#!/bin/sh
# Scion-FlashNext-176B-A6B community-benchmark arms (GSM8K-300 / MATH-200 /
# HumanEval-164): serve the TERNARY release (PTQ1_0 + corrections) with the
# moe-corr fork, PLE table on CPU/mmap, generate 3 samples/task at the serving
# sampling (non-thinking, 700 cap) plus one think-harder sample/task (thinking,
# 2048 cap), embed judge prompts, fit the in-domain head and evaluate arms
# A/B/C/D on the held-out test tasks.
#   systemd-run --user --unit=scion-flashnext-community /bin/sh <this>
# Mirrors results/jev-readout-20261006/run-community-scion.sh.
set -u
W=<work>/hivebench
PY=<work>/.venv-rocm/bin/python
B=<work>/llama-qwen4exp/build-q4exp-hip/bin/llama-server
M=~/models/scion-strata/Scion-FlashNext-176B-A6B-Q6K.gguf
R=$W/experiments/cascade/results/scion-flashnext-community-20261009
export HIP_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
mkdir -p "$R"
cd "$W" || exit 1

echo "== start scion-flashnext server $(date -Is)"
"$B" -m "$M" -ngl 99 -ot "per_layer_token_embd=CPU" -c 8192 --no-warmup \
    --host 127.0.0.1 --port 8897 \
    --embeddings --pooling last --embd-normalize -1 -fa auto --jinja \
    > ~/models/scion-strata/community-srv.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 300); do
    if curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
if [ "$up" != 1 ]; then echo "server did not start"; kill $SRV 2>/dev/null; exit 1; fi
echo "== server up $(date -Is)"

echo "== gen 3 samples/task $(date -Is)"
"$PY" experiments/cascade/community_eval.py --server http://127.0.0.1:8897 \
    --prefix scion-flashnext --model-label SkyIsNotGreen/Scion-FlashNext-176B-A6B \
    --model-name scion-flashnext --runtime "moe-corr fork (llama.cpp b10721, PTQ1_0+corr)" \
    --tasks experiments/cascade/tasks-community.json --phase gen \
    --out-dir "$R" \
    > "$R/scion-flashnext-gen.log" 2>&1 || echo "FAILED gen"

echo "== think-harder arm $(date -Is)"
"$PY" experiments/cascade/community_eval.py --server http://127.0.0.1:8897 \
    --prefix scion-flashnext --phase hard --out-dir "$R" \
    > "$R/scion-flashnext-hard.log" 2>&1 || echo "FAILED hard"

echo "== embeddings $(date -Is)"
"$PY" experiments/cascade/community_eval.py --server http://127.0.0.1:8897 \
    --prefix scion-flashnext --phase embed --out-dir "$R" \
    > "$R/scion-flashnext-embed.log" 2>&1 || echo "FAILED embed"

echo "== eval $(date -Is)"
"$PY" experiments/cascade/community_eval.py --prefix scion-flashnext \
    --phase eval --out-dir "$R" \
    > "$R/scion-flashnext-eval.log" 2>&1 || echo "FAILED eval"

echo "== stop server $(date -Is)"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "== done $(date -Is)"
