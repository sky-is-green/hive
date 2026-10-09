#!/bin/bash
# community-resume2.sh - resume the community eval with the harness parallelized across four server slots.
# The first resume run (sequential) is killed before this starts; the completed samples are reused by the
# harness's resume logic.
set -uo pipefail
export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH"
B=/workspace/bench
R=$B/community
COM=/root/community
mkdir -p "$R"
exec >> "$B/community-resume.log" 2>&1
echo "=== community-resume2 start $(date -u)"

cd /workspace/llama-qwen4exp
OURS=/root/models/Scion-FlashNext-176B-A6B-Q6K.gguf
./build/bin/llama-server -m "$OURS" -ngl 99 -c 8192 -np 4 -t 8 --no-warmup \
    --host 127.0.0.1 --port 8897 \
    --embeddings --pooling last --embd-normalize -1 -fa auto --jinja \
    > server2.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 150); do
    if curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
if [ "$up" != 1 ]; then
    echo "SERVER2_FAILED"
    tail -c 1200 server2.log
    exit 1
fi
echo "server up ($(date -u), -np 4)"

PY=python3
CONC=4
echo "== gen 3 samples/task $(date -u)"
$PY "$COM/community_eval.py" --server http://127.0.0.1:8897 --concurrency $CONC \
    --prefix scion-flashnext --model-label SkyIsNotGreen/Scion-FlashNext-176B-A6B \
    --model-name scion-flashnext --runtime "moe-corr fork (CUDA, PTQ1_0+corr)" \
    --tasks "$COM/tasks-community.json" --phase gen --out-dir "$R" \
    > "$R/gen.log" 2>&1 || echo "FAILED gen"

echo "== think-harder arm $(date -u)"
$PY "$COM/community_eval.py" --server http://127.0.0.1:8897 --concurrency $CONC \
    --prefix scion-flashnext --phase hard --out-dir "$R" \
    > "$R/hard.log" 2>&1 || echo "FAILED hard"

echo "== embeddings $(date -u)"
$PY "$COM/community_eval.py" --server http://127.0.0.1:8897 --concurrency $CONC \
    --prefix scion-flashnext --phase embed --out-dir "$R" \
    > "$R/embed.log" 2>&1 || echo "FAILED embed"

echo "== eval $(date -u)"
$PY "$COM/community_eval.py" --prefix scion-flashnext \
    --jbench-prefix scion-flashnext --phase eval --out-dir "$R" \
    > "$R/eval.log" 2>&1 || echo "FAILED eval"

kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
tar czf "$B/community-artifacts.tgz" -C "$R" . 2>/dev/null
echo "COMMUNITY_RESUME_DONE $(date -u)"
