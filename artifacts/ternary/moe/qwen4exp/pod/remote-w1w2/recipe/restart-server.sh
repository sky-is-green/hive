#!/bin/bash
# restart-server.sh - reload llama-server with a per-slot context that fits the thinking-hard arm
# (16384 / 4 slots = 4096 tokens per sequence).
set -uo pipefail
cd /workspace/llama-qwen4exp
pkill -f "[b]uild/bin/llama-server"
sleep 3
setsid nohup ./build/bin/llama-server -m /root/models/Scion-FlashNext-176B-A6B-Q6K.gguf -ngl 99 -c 16384 -np 4 -t 8 --no-warmup \
    --host 127.0.0.1 --port 8897 --embeddings --pooling last --embd-normalize -1 -fa auto --jinja --cache-ram 0 \
    > server2.log 2>&1 < /dev/null &
for i in $(seq 1 150); do
    if curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1; then echo "SERVER_UP $(date -u)"; exit 0; fi
    sleep 2
done
echo "SERVER_FAILED"; tail -c 1000 server2.log; exit 1
