#!/bin/bash
# embed-test2.sh - embeddings with --pooling none (per-token), to see if the LAST-pool read-back is the abort.
set -uo pipefail
cd /workspace/llama-qwen4exp
pkill -f "[b]uild/bin/llama-server"; sleep 3
setsid nohup ./build/bin/llama-server -m /root/models/Scion-FlashNext-176B-A6B-Q6K.gguf -ngl 99 -c 4096 -np 1 -t 8 --no-warmup \
    --host 127.0.0.1 --port 8897 --embeddings --pooling none --embd-normalize -1 -fa auto --jinja \
    > server4.log 2>&1 < /dev/null &
for i in $(seq 1 150); do curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1 && break; sleep 2; done
echo "server up $(date -u)"; grep -a "pooling" server4.log | head -2
echo "--- embedding (pooling none):"
curl -s --max-time 120 http://127.0.0.1:8897/embedding -H 'Content-Type: application/json' \
    -d '{"content":"hello world","embd_normalize":-1}' > /tmp/emb_none.json 2>&1
python3 - <<'EOF'
import json
try:
    d = json.load(open("/tmp/emb_none.json"))
    e = d["data"][0]["embedding"] if isinstance(d, dict) and "data" in d else d[0]["embedding"]
    print("n_rows:", len(e), "row_len:", len(e[0]) if e and isinstance(e[0], list) else "flat")
    print("first5:", (e[0][:5] if e and isinstance(e[0], list) else e[:5]))
except Exception as ex:
    print("parse:", ex, open("/tmp/emb_none.json").read()[:200])
EOF
echo "--- alive?"; pgrep -af "[l]lama-server" | head -1
echo "--- server4 tail:"; tail -4 server4.log
