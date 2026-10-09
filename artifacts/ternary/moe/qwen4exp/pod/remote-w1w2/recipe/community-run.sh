#!/bin/bash
# community-run.sh - runs ON the pod. Community-protocol window on the TRUE
# release (ternary PTQ1_0 + corrections), fully GPU-resident on an 80 GB card:
#   deps -> clone fork -> CUDA build -> download Q6K + sha -> smoke ->
#   llama-server (--embeddings) -> gen / hard / embed / eval -> artifacts.
# Logs in /workspace/bench/.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
# checkers.py runs candidate code via tempfile; keep it on the system temp dir whatever the launching env had
export TMPDIR=/tmp
B=/workspace/bench
R=$B/community
mkdir -p "$R"
LOG=$B/community-run.log
exec >> "$LOG" 2>&1
echo "=== community-run start $(date -u)"

REPO=/workspace/llama-qwen4exp
MODELS=/root/models
OURS=$MODELS/Scion-FlashNext-176B-A6B-Q6K.gguf
OURS_SHA=236c6ef96f0a5c3bbf660101c51cb1fdcd6f63724d9341a3d5aedc568bd08cec
COM=/root/community
# community_eval.py resolves its defaults from __file__ (REPO/experiments/cascade), which on the pod is /;
# give it the real directory so a phase that forgets --tasks still finds them.
mkdir -p /experiments/cascade
cp -f "$COM/tasks-community.json" /experiments/cascade/tasks-community.json 2>/dev/null || true

echo "--- deps"
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq rsync ninja-build cmake build-essential git >/dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m pip install -q --break-system-packages "huggingface_hub[hf_xet]" numpy 2>&1 | tail -1

cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
[ -n "$cc" ] || cc=80
echo "CUDA arch: $cc"

echo "--- clone fork + CUDA build $(date -u)"
rm -rf "$REPO"
git clone -q -b qwen4exp-proto https://github.com/sky-is-green/prism-ml-llama.cpp "$REPO" || { echo CLONE_FAILED; exit 1; }
git -C "$REPO" apply /root/bench-ple-multiseq.patch || { echo PATCH_FAILED; exit 1; }
git -C "$REPO" apply /root/server-skip-seq-rm.patch || { echo PATCH2_FAILED; exit 1; }
git -C "$REPO" apply /root/embed-read-width.patch || { echo PATCH3_FAILED; exit 1; }
cd "$REPO"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$cc" > cmake.log 2>&1 || { echo CMAKE_FAILED; tail -20 cmake.log; exit 1; }
cmake --build build -j"$(nproc)" --target llama-server llama-completion > build.log 2>&1 || { echo BUILD_FAILED; tail -40 build.log; exit 1; }
echo "build OK $(date -u)"

echo "--- download $(date -u)"
mkdir -p "$MODELS"
dl() { # dl <tag> <hf args...>
    local tag="$1"; shift
    local log="$B/dl-$tag.log"
    if hf download "$@" > "$log" 2>&1; then echo "dl $tag OK"; return 0; fi
    echo "dl $tag xet failed; HTTP retry"
    HF_HUB_DISABLE_XET=1 hf download "$@" >> "$log" 2>&1 && { echo "dl $tag OK (http)"; return 0; }
    echo "DL_${tag^^}_FAILED"; tail -8 "$log"; return 1
}
dl ours SkyIsNotGreen/Scion-FlashNext-176B-A6B Scion-FlashNext-176B-A6B-Q6K.gguf --local-dir "$MODELS" || exit 1
echo "$OURS_SHA  $OURS" | sha256sum -c - >/dev/null 2>&1 && echo "ours sha256 OK" || { echo SHA_FAILED; exit 1; }

echo "--- smoke (full GPU) $(date -u)"
timeout 900 ./build/bin/llama-completion -m "$OURS" -ngl 99 -c 512 -t 8 --temp 0 -n 8 -no-cnv \
    -p "The capital of France is" > smoke.log 2>&1
grep -q "Paris" smoke.log || { echo SMOKE_FAILED; tail -c 300 smoke.log; exit 1; }
echo "smoke OK"

echo "--- server (all resident, embeddings, 4 slots) $(date -u)"
./build/bin/llama-server -m "$OURS" -ngl 99 -c 16384 -np 4 -t 8 --no-warmup \
    --host 127.0.0.1 --port 8897 \
    --embeddings --pooling last --embd-normalize -1 -fa auto --jinja --cache-ram 0 \
    > server.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 150); do
    if curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
if [ "$up" != 1 ]; then
    echo SERVER_FAILED
    echo "--- server.log tail ---"
    tail -c 1200 server.log
    echo "--- end server.log tail ---"
    exit 1
fi
echo "server up $(date -u)"

PY=python3
echo "== gen 3 samples/task $(date -u)"
$PY "$COM/community_eval.py" --server http://127.0.0.1:8897 --concurrency 4 \
    --prefix scion-flashnext --model-label SkyIsNotGreen/Scion-FlashNext-176B-A6B \
    --model-name scion-flashnext --runtime "moe-corr fork (CUDA, PTQ1_0+corr)" \
    --tasks "$COM/tasks-community.json" --phase gen --out-dir "$R" \
    > "$R/gen.log" 2>&1 || echo "FAILED gen"

echo "== think-harder arm $(date -u)"
$PY "$COM/community_eval.py" --server http://127.0.0.1:8897 --concurrency 4 \
    --prefix scion-flashnext --phase hard --out-dir "$R" \
    --tasks "$COM/tasks-community.json" \
    > "$R/hard.log" 2>&1 || echo "FAILED hard"

echo "== embeddings $(date -u)"
$PY "$COM/community_eval.py" --server http://127.0.0.1:8897 --concurrency 4 \
    --prefix scion-flashnext --phase embed --out-dir "$R" \
    --tasks "$COM/tasks-community.json" \
    > "$R/embed.log" 2>&1 || echo "FAILED embed"

echo "== eval $(date -u)"
$PY "$COM/community_eval.py" --prefix scion-flashnext \
    --jbench-prefix scion-flashnext --phase eval --out-dir "$R" \
    --tasks "$COM/tasks-community.json" \
    > "$R/eval.log" 2>&1 || echo "FAILED eval"

kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
tar czf "$B/community-artifacts.tgz" -C "$R" . 2>/dev/null
echo "COMMUNITY_DONE $(date -u)"
