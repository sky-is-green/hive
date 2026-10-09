#!/bin/bash
# community-probe.sh - runs ON the A6000 pod: full setup (deps, fork+patches, build, model download), smoke,
# then measures single and 4-concurrent generation throughput, embedding, and VRAM. Leaves the pod running.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
B=/workspace/bench
mkdir -p "$B"
cd "$B"
exec >> "$B/probe.log" 2>&1
echo "=== community-probe start $(date -u)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
free -g | head -2

echo "--- deps"
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq rsync ninja-build cmake build-essential git >/dev/null 2>&1
python3 -m pip install -q --break-system-packages "huggingface_hub[hf_xet]" 2>&1 | tail -1

cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
[ -n "$cc" ] || cc=86
echo "CUDA arch: $cc"

REPO=/workspace/llama-qwen4exp
echo "--- clone + build $(date -u)"
rm -rf "$REPO"
git clone -q -b qwen4exp-proto https://github.com/sky-is-green/prism-ml-llama.cpp "$REPO" || { echo CLONE_FAILED; exit 1; }
cd "$REPO"
git apply /root/bench-ple-multiseq.patch || { echo PATCH_FAILED; exit 1; }
git apply /root/server-skip-seq-rm.patch || { echo PATCH2_FAILED; exit 1; }
git apply /root/embed-read-width.patch || { echo PATCH3_FAILED; exit 1; }
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$cc" > cmake.log 2>&1 || { echo CMAKE_FAILED; tail -20 cmake.log; exit 1; }
cmake --build build -j"$(nproc)" --target llama-server llama-completion > build.log 2>&1 || { echo BUILD_FAILED; tail -40 build.log; exit 1; }
echo "build OK $(date -u)"

echo "--- download $(date -u)"
mkdir -p /root/models
hf download SkyIsNotGreen/Scion-FlashNext-176B-A6B Scion-FlashNext-176B-A6B-Q6K.gguf --local-dir /root/models > dl.log 2>&1 \
    || { echo DL_FAILED; tail -8 dl.log; exit 1; }
OURS=/root/models/Scion-FlashNext-176B-A6B-Q6K.gguf
echo "dl OK"

echo "--- smoke $(date -u)"
timeout 900 ./build/bin/llama-completion -m "$OURS" -ngl 99 -c 512 -t 8 --temp 0 -n 8 -no-cnv \
    -p "The capital of France is" > smoke.log 2>&1
grep -q "Paris" smoke.log || { echo SMOKE_FAILED; tail -c 300 smoke.log; exit 1; }
echo "smoke OK"

echo "--- server + measurements $(date -u)"
./build/bin/llama-server -m "$OURS" -ngl 99 -c 16384 -np 4 -t 8 --no-warmup \
    --host 127.0.0.1 --port 8897 \
    --embeddings --pooling last --embd-normalize -1 -fa auto --jinja \
    > server.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 150); do
    if curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
[ "$up" = 1 ] || { echo SERVER_FAILED; tail -c 800 server.log; exit 1; }
echo "server up"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "--- single 256-token generation"
t0=$(date +%s.%N)
curl -s --max-time 900 http://127.0.0.1:8897/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"m","messages":[{"role":"user","content":"Write a short story about a lighthouse keeper."}],"max_tokens":256,"temperature":0.7}' > gen1.json
t1=$(date +%s.%N)
python3 - "$t0" "$t1" gen1.json <<'EOF'
import json, sys
t0, t1 = float(sys.argv[1]), float(sys.argv[2])
d = json.load(open(sys.argv[3]))
n = d["usage"]["completion_tokens"]
print("single: tokens=%d secs=%.1f tok/s=%.1f" % (n, t1 - t0, n / max(1e-9, t1 - t0)))
EOF

echo "--- 4 concurrent 256-token generations"
t0=$(date +%s.%N)
for i in 1 2 3 4; do
    curl -s --max-time 1800 http://127.0.0.1:8897/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"model":"m","messages":[{"role":"user","content":"Write a short story about a lighthouse keeper."}],"max_tokens":256,"temperature":0.7}' > "gen$i.json" &
done
wait
t1=$(date +%s.%N)
python3 - "$t0" "$t1" <<'EOF'
import json, sys
t0, t1 = float(sys.argv[1]), float(sys.argv[2])
tok = 0
for i in range(1, 5):
    try:
        tok += json.load(open("gen%d.json" % i))["usage"]["completion_tokens"]
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
kill $SRV 2>/dev/null
echo "PROBE_DONE $(date -u)"
