#!/bin/bash
# probe2.sh - finish the A6000 measurements on the already-loaded server: 4-concurrent aggregate + embedding.
set -uo pipefail
B=/workspace/bench
cd "$B"
echo "=== probe2 $(date -u)"
t0=$(date +%s.%N)
pids=()
for i in 1 2 3 4; do
    curl -s --max-time 600 http://127.0.0.1:8897/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"model":"m","messages":[{"role":"user","content":"Write a short story about a lighthouse keeper."}],"max_tokens":256,"temperature":0.7}' \
        > "genb$i.json" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
t1=$(date +%s.%N)
python3 - "$t0" "$t1" <<'EOF'
import json, sys
t0, t1 = float(sys.argv[1]), float(sys.argv[2])
tok = 0
for i in range(1, 5):
    try:
        tok += json.load(open("genb%d.json" % i))["usage"]["completion_tokens"]
    except Exception:
        pass
print("conc4: tokens=%d secs=%.1f tok/s=%.1f" % (tok, t1 - t0, tok / max(1e-9, t1 - t0)))
EOF
echo "--- embedding"
curl -s --max-time 120 http://127.0.0.1:8897/embedding -H 'Content-Type: application/json' \
    -d '{"content":"hello world","embd_normalize":-1}' > emb.json 2>&1
python3 - <<'EOF'
import json
try:
    d = json.load(open("emb.json"))
    e = d["data"][0]["embedding"]
    print("embedding: rows=%d row_len=%d" % (len(e), len(e[0]) if e and isinstance(e[0], list) else 0))
except Exception as ex:
    print("embedding parse:", ex)
EOF
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "PROBE2_DONE $(date -u)"
