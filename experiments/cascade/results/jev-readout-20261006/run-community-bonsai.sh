#!/bin/sh
# Bonsai community-benchmark, head-only measurement (no baseline re-run: the
# model card already carries GSM8K/MATH/HumanEval baselines).  Serves Bonsai,
# generates 3 samples/task at the card's thinking sampling, embeds judge
# prompts, fits the in-domain head and evaluates arms C (head-gated) / D
# (best-of-3) plus the natural-error catch; arm A comes from the same pool and
# arm B is skipped.
#   systemd-run --user --unit=bonsai-community /bin/sh <this>
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
B=/home/penis/llama.cpp/build/bin/llama-server
M=/home/penis/Downloads/Ternary-Bonsai-2-27B-PQ2_0.gguf
R=$W/experiments/cascade/results/jev-readout-20261006
export HIP_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
cd "$W" || exit 1

echo "== start bonsai server $(date -Is)"
"$B" -m "$M" -ngl 99 -c 8192 --host 127.0.0.1 --port 8897 \
    --embeddings --pooling last --embd-normalize -1 -fa auto --jinja \
    > /tmp/opencode/bonsai-community-srv.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
if [ "$up" != 1 ]; then echo "server did not start"; kill $SRV 2>/dev/null; exit 1; fi

echo "== gen 3 samples/task (thinking) $(date -Is)"
"$PY" experiments/cascade/community_eval.py --server http://127.0.0.1:8897 \
    --prefix bonsai-community --model-label prism-ml/Ternary-Bonsai-2-27B-gguf \
    --model-name bonsai --runtime "moe-corr fork (llama.cpp 0.2.0-dev b10721)" \
    --tasks experiments/cascade/tasks-community.json --phase gen \
    --thinking --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 \
    --max-tokens 4096 --jbench-prefix bonsai27b \
    > "$R/bonsai-community-gen.log" 2>&1 || echo "FAILED gen"

echo "== embeddings $(date -Is)"
"$PY" experiments/cascade/community_eval.py --server http://127.0.0.1:8897 \
    --prefix bonsai-community --phase embed --jbench-prefix bonsai27b \
    > "$R/bonsai-community-embed.log" 2>&1 || echo "FAILED embed"

echo "== eval $(date -Is)"
"$PY" experiments/cascade/community_eval.py --prefix bonsai-community \
    --model-label prism-ml/Ternary-Bonsai-2-27B-gguf --phase eval \
    --jbench-prefix bonsai27b > "$R/bonsai-community-eval.log" 2>&1 \
    || echo "FAILED eval"

echo "== stop server $(date -Is)"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "== done $(date -Is)"
