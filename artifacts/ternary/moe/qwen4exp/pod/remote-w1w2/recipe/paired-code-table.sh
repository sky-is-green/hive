#!/bin/bash
# paired-code-table.sh - runs ON the pod: same-pod paired code gate for the
# release and the Q4X160 PLE-table export.
#   deps -> clone + 6 patches -> CUDA build -> release download + sha -> smoke
#   -> serve release: gen 700 / hard 2048 / hard 4096
#   -> table export from uploaded bytes (Q4X160) -> smoke
#   -> serve table: gen 700 / hard 2048 / hard 4096
# No embeddings: the pass labels are computed by the checkers at generation
# time, and the local code_gate_eval.py compares the JSONL.
# Logs + artifacts in /workspace/bench/.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
export TMPDIR=/tmp
export TABLE_TYPE=q4x160
export TABLE_BYTES=${TABLE_BYTES:-/root/q4x160-table.bin}
B=/workspace/bench
P=$B/paired
mkdir -p "$P"
LOG=$B/paired-code-table.log
exec >> "$LOG" 2>&1
echo "=== paired-code-table start $(date -u) type=$TABLE_TYPE"

REPO=/workspace/llama-qwen4exp
MODELS=/root/models
OURS=$MODELS/Scion-FlashNext-176B-A6B-Q6K.gguf
OURS_SHA=236c6ef96f0a5c3bbf660101c51cb1fdcd6f63724d9341a3d5aedc568bd08cec
COM=/root/community

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
for p in bench-ple-multiseq server-skip-seq-rm embed-read-width router-stats kld-gate ple-table; do
    git -C "$REPO" apply "/root/$p.patch" || { echo "PATCH_FAILED ($p)"; exit 1; }
done
cd "$REPO"
./verify-container-support.sh | tail -1
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$cc" > cmake.log 2>&1 || { echo CMAKE_FAILED; tail -20 cmake.log; exit 1; }
cmake --build build -j"$(nproc)" --target llama-server llama-completion > build.log 2>&1 || { echo BUILD_FAILED; tail -40 build.log; exit 1; }
echo "build OK $(date -u)"

# shellcheck source=pod-build-table.sh
source /root/pod-build-table.sh

echo "--- downloads $(date -u)"
mkdir -p "$MODELS"
dl_release || exit 1
echo "$OURS_SHA  $OURS" | sha256sum -c - >/dev/null 2>&1 && echo "release sha256 OK" || { echo SHA_FAILED; exit 1; }

srv_start() { # srv_start <model> <log>
    ./build/bin/llama-server -m "$1" -ngl 99 -c 16384 -np 4 -t 8 --no-warmup \
        --host 127.0.0.1 --port 8897 --embeddings --pooling last --embd-normalize -1 \
        -fa auto --jinja --cache-ram 0 > "$2" 2>&1 &
    SRV=$!
    for i in $(seq 1 150); do
        curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1 && return 0
        sleep 2
    done
    return 1
}
srv_stop() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }

run_arms() { # run_arms <dir> <prefix>
    local d=$1 prefix=$2
    local tasks=$COM/tasks-community-code.json
    mkdir -p "$d" "$d-cap4096"
    python3 "$COM/community_eval.py" --server http://127.0.0.1:8897 --concurrency 4 \
        --prefix "$prefix" --model-label "$prefix" --model-name "$prefix" \
        --runtime "paired gate" --tasks "$tasks" --phase gen --out-dir "$d" \
        > "$d/gen.log" 2>&1 || echo "FAILED gen $prefix"
    python3 "$COM/community_eval.py" --server http://127.0.0.1:8897 --concurrency 4 \
        --prefix "$prefix" --phase hard --out-dir "$d" --tasks "$tasks" \
        > "$d/hard.log" 2>&1 || echo "FAILED hard $prefix"
    python3 "$COM/community_eval.py" --server http://127.0.0.1:8897 --concurrency 4 \
        --prefix "$prefix" --phase hard --out-dir "$d-cap4096" --tasks "$tasks" \
        --hard-max-tokens 4096 > "$d-cap4096/cap4096.log" 2>&1 || echo "FAILED hard4096 $prefix"
}

echo "--- smoke release $(date -u)"
timeout 900 ./build/bin/llama-completion -m "$OURS" -ngl 99 -c 512 -t 8 --temp 0 -n 8 -no-cnv \
    -p "The capital of France is" > "$B/smoke-release.log" 2>&1
grep -q "Paris" "$B/smoke-release.log" || { echo SMOKE_FAILED; tail -c 300 "$B/smoke-release.log"; exit 1; }

echo "--- serve release + arms $(date -u)"
mkdir -p "$P/release" "$P/release-cap4096"
srv_start "$OURS" "$P/server-release.log" || { echo RELEASE_SERVER_FAILED; tail -c 800 "$P/server-release.log"; exit 1; }
run_arms "$P/release" scion-rel
srv_stop
echo "release arms done $(date -u)"

build_table || exit 1

echo "--- serve table + arms $(date -u)"
mkdir -p "$P/table" "$P/table-cap4096"
srv_start "$TABLE" "$P/server-table.log" || { echo TABLE_SERVER_FAILED; tail -c 800 "$P/server-table.log"; exit 1; }
run_arms "$P/table" scion-tbl
srv_stop
echo "table arms done $(date -u)"

ls -la "$P"/release "$P"/table "$P"/release-cap4096 "$P"/table-cap4096
echo "PAIRED_CODE_DONE $(date -u)"
