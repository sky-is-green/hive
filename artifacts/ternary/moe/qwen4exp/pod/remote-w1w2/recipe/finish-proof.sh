#!/bin/bash
# finish-proof.sh - apply the embed read-width fix, rebuild llama-server, run the embed + eval phases for the
# 40-task proof (gen + hard already recorded), and save the proof table.
set -uo pipefail
B=/workspace/bench
R=$B/community
COM=/root/community
cd /workspace/llama-qwen4exp
if ! git apply /root/embed-read-width.patch; then echo "PATCH3_APPLY_FAILED"; exit 1; fi
cmake --build build -j"$(nproc)" --target llama-server > /tmp/rebuild2.log 2>&1 && echo REBUILD_OK || { echo REBUILD_FAIL; tail -20 /tmp/rebuild2.log; exit 1; }
pkill -f "[b]uild/bin/llama-server"; sleep 3
setsid nohup ./build/bin/llama-server -m /root/models/Scion-FlashNext-176B-A6B-Q6K.gguf -ngl 99 -c 16384 -np 4 -t 8 --no-warmup \
    --host 127.0.0.1 --port 8897 --embeddings --pooling last --embd-normalize -1 -fa auto --jinja \
    > server6.log 2>&1 < /dev/null &
for i in $(seq 1 150); do curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1 && break; sleep 2; done
echo "server up $(date -u)"; grep -a "n_ctx_slot" server6.log | tail -1
PY=python3
echo "== embed $(date -u)"
$PY "$COM/community_eval.py" --server http://127.0.0.1:8897 --concurrency 4 --prefix scion-flashnext \
    --out-dir "$R" --limit 40 --tasks "$COM/tasks-community.json" --phase embed \
    > "$R/embed.log" 2>&1 || echo "FAILED embed"
echo "== eval $(date -u)"
$PY "$COM/community_eval.py" --prefix scion-flashnext --out-dir "$R" --limit 40 \
    --tasks "$COM/tasks-community.json" --phase eval > "$R/eval.log" 2>&1 || echo "FAILED eval"
cp -f "$R/scion-flashnext-eval.json" "$R/scion-flashnext-eval-proof40.json" 2>/dev/null || true
echo "FINISH_PROOF_DONE $(date -u)"
