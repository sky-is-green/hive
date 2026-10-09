#!/bin/bash
# embed-test.sh - one /embedding request on a fresh single-slot server, to isolate the embed-path abort.
set -uo pipefail
cd /workspace/llama-qwen4exp
pkill -f "[b]uild/bin/llama-server"; sleep 3
setsid nohup ./build/bin/llama-server -m /root/models/Scion-FlashNext-176B-A6B-Q6K.gguf -ngl 99 -c 8192 -np 1 -t 8 --no-warmup \
    --host 127.0.0.1 --port 8897 --embeddings --pooling last --embd-normalize -1 -fa auto --jinja \
    > server3.log 2>&1 < /dev/null &
for i in $(seq 1 150); do curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1 && break; sleep 2; done
echo "server up $(date -u); n_ctx:"; grep -a "n_ctx_slot" server3.log | tail -1
echo "--- apply-template:"
curl -s --max-time 60 http://127.0.0.1:8897/apply-template -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hello"}],"add_generation_prompt":true}' | head -c 160; echo
echo "--- embedding:"
curl -s --max-time 120 http://127.0.0.1:8897/embedding -H 'Content-Type: application/json' \
    -d '{"content":"hello world","embd_normalize":-1}' | head -c 200; echo
echo "--- server alive?"; pgrep -af "[l]lama-server" | head -1
echo "--- server3.log tail:"; tail -8 server3.log
