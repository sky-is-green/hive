#!/bin/bash
# bench-run.sh - runs ON the pod (uploaded to /root). Bench window payload:
#   deps -> clone fork -> CUDA build -> downloads -> smoke -> PPL/HS/WG on
#   our Q6K and ISTA Q2_0 -> SUMMARY.  Logs in /workspace/bench/.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
# CUDA toolkit is installed but not on the default PATH in this image
export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
B=/workspace/bench
mkdir -p "$B"
LOG="$B/run.log"
exec >> "$LOG" 2>&1
echo "=== bench-run start $(date -u)"

REPO=/workspace/llama-qwen4exp
MODELS=/root/models   # overlay disk (200 GB NVMe), not MooseFS (xet writer flaked there)
OURS=$MODELS/Scion-FlashNext-176B-A6B-Q6K.gguf
ISTA=$MODELS/ista/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
WIKI=/workspace/bench/wiki.test.raw
HS=$B/hellaswag_val_full.txt
WG=$B/winogrande-debiased-eval.csv
OURS_SHA=236c6ef96f0a5c3bbf660101c51cb1fdcd6f63724d9341a3d5aedc568bd08cec

echo "--- deps"
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq rsync ninja-build cmake build-essential unzip git >/dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m pip install -q --break-system-packages "huggingface_hub[hf_xet]" 2>&1 | tail -1

# CUDA arch from the booked card (A100=80, H100=90, Blackwell=120)
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
if [ -z "$cc" ]; then
    cc=$(nvidia-smi -q 2>/dev/null | grep -m1 "CUDA Capability" | grep -oE '[0-9]+\.[0-9]+' | head -1 | tr -d '.')
fi
[ -n "$cc" ] || cc=80
echo "CUDA arch: $cc"

echo "--- clone fork + CUDA build $(date -u)"
rm -rf "$REPO"
git clone -q -b qwen4exp-proto https://github.com/sky-is-green/prism-ml-llama.cpp "$REPO" || { echo CLONE_FAILED; exit 1; }
git -C "$REPO" apply /root/bench-ple-multiseq.patch || { echo PATCH_FAILED; exit 1; }
cd "$REPO"
./verify-container-support.sh | tail -1
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$cc" > cmake.log 2>&1 || { echo CMAKE_FAILED; tail -20 cmake.log; exit 1; }
cmake --build build -j"$(nproc)" --target llama-perplexity llama-cli llama-server llama-completion > build.log 2>&1 || { echo BUILD_FAILED; tail -40 build.log; exit 1; }
echo "build OK $(date -u)"
ls -la build/bin/llama-perplexity build/bin/llama-cli build/bin/llama-completion

echo "--- downloads $(date -u)"
mkdir -p "$MODELS/ista"
dl() { # dl <tag> <hf args...> — xet first, classic HTTP fallback
    local tag="$1"; shift
    local log="$B/dl-$tag.log"
    if hf download "$@" > "$log" 2>&1; then echo "dl $tag OK (xet)"; return 0; fi
    echo "dl $tag xet attempt failed; retrying with HF_HUB_DISABLE_XET=1"; tail -3 "$log"
    HF_HUB_DISABLE_XET=1 hf download "$@" >> "$log" 2>&1 && { echo "dl $tag OK (http)"; return 0; }
    echo "DL_${tag^^}_FAILED"; tail -8 "$log"; return 1
}
dl ours SkyIsNotGreen/Scion-FlashNext-176B-A6B Scion-FlashNext-176B-A6B-Q6K.gguf --local-dir "$MODELS" || exit 1
dl ista ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --include "Q2_0/*" --local-dir "$MODELS/ista" || exit 1
curl -sL -o /workspace/bench/wiki.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip || { echo DL_WIKI_FAILED; exit 1; }
unzip -oq /workspace/bench/wiki.zip -d /workspace/bench || { echo UNZIP_FAILED; exit 1; }
WIKI=$(find /workspace/bench -name wiki.test.raw | head -1)
# task data: llama-perplexity reads both from the -f prompt; same files as the
# 35B grid (hellaswag 6-line-per-task text, winogrande debiased CSV)
curl -sL -o "$HS" https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt || { echo DL_HS_FAILED; exit 1; }
curl -sL -o "$WG" https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp/raw/main/winogrande-debiased-eval.csv || { echo DL_WG_FAILED; exit 1; }
echo "ours:  $(stat -c %s "$OURS" 2>/dev/null) bytes"
echo "ista:  $(ls "$MODELS/ista/Q2_0/" 2>/dev/null | tr '\n' ' ')"
echo "wiki:  $WIKI ($(stat -c %s "$WIKI" 2>/dev/null) bytes)"
echo "hs:    $HS ($(stat -c %s "$HS" 2>/dev/null) bytes)"
echo "wg:    $WG ($(stat -c %s "$WG" 2>/dev/null) bytes)"
df -h / "$MODELS" /workspace 2>/dev/null | tail -3
[ -s "$OURS" ] && [ -s "$ISTA" ] && [ -s "$WIKI" ] && [ -s "$HS" ] && [ -s "$WG" ] || { echo MISSING_INPUTS; exit 1; }
sha=$(sha256sum "$OURS" | awk '{print $1}')
[ "$sha" = "$OURS_SHA" ] || { echo "SHA_OURS_FAILED got=$sha"; exit 1; }
echo "ours sha256 OK"

echo "--- CUDA smoke (full GPU) $(date -u)"
# llama-cli applies the chat template (thinking mode false-negatives a short
# check) and, in this fork, rejects -no-cnv (registered for the completion
# example only), so the raw-completion binary is used here.
timeout 900 ./build/bin/llama-completion -m "$OURS" -ngl 99 -c 512 -t 8 --temp 0 -n 8 -no-cnv \
    -p "The capital of France is" > smoke.log 2>&1
echo "smoke rc=$? $(tail -c 200 smoke.log | tr '\r' '\n' | tail -2)"
if ! grep -q "Paris" smoke.log; then
    # diagnostics only: does the model work at all with CPU experts?
    timeout 600 ./build/bin/llama-completion -m "$OURS" -ngl 99 --cpu-moe -c 512 -t 8 --temp 0 -n 4 -no-cnv \
        -p "The capital of France is" > smoke-cpumoe.log 2>&1
    echo "cpu-moe smoke rc=$? $(tail -c 200 smoke-cpumoe.log | tr '\r' '\n' | tail -2)"
    echo "SMOKE_FAILED - aborting (CPU-expert benches would exceed the window budget)"
    exit 1
fi
echo "smoke OK"

echo "--- tiny PPL dry-run (GPU, 4 chunks) $(date -u)"
timeout 900 ./build/bin/llama-perplexity -m "$OURS" -f "$WIKI" -c 512 --chunks 4 -ngl 99 -t 8 \
    > "$B/tiny-ppl.log" 2>&1
tiny_rc=$?
echo "tiny ppl rc=$tiny_rc $(grep -a 'Final estimate' "$B/tiny-ppl.log" | tail -1)"
[ "$tiny_rc" -eq 0 ] || { echo TINY_PPL_FAILED; tail -8 "$B/tiny-ppl.log"; exit 1; }
echo "tiny ppl OK"

bench() { # bench <tag> <model> <first-or-only>
    local tag="$1" model="$2"
    echo "--- $tag ppl $(date -u)"
    timeout 10800 ./build/bin/llama-perplexity -m "$model" -f "$WIKI" -c 512 --chunks 580 \
        -ngl 99 -t 8 > "$B/$tag-ppl.log" 2>&1
    echo "$tag ppl rc=$? $(date -u)"
    echo "--- $tag hellaswag $(date -u)"
    timeout 7200 ./build/bin/llama-perplexity -m "$model" -f "$HS" -c 512 -ngl 99 -t 8 \
        --hellaswag --hellaswag-tasks 400 > "$B/$tag-hellaswag.log" 2>&1
    echo "$tag hellaswag rc=$? $(date -u)"
    echo "--- $tag winogrande $(date -u)"
    timeout 7200 ./build/bin/llama-perplexity -m "$model" -f "$WG" -c 512 -ngl 99 -t 8 \
        --winogrande --winogrande-tasks 400 > "$B/$tag-winogrande.log" 2>&1
    echo "$tag winogrande rc=$? $(date -u)"
    grep -H "Final estimate: PPL" "$B/$tag-ppl.log" | tail -1
    grep -H -E "^\s*400\s" "$B/$tag-hellaswag.log" | tail -1
    grep -H "Final Winogrande score" "$B/$tag-winogrande.log" | tail -1
}

bench ours "$OURS"
bench ista "$ISTA"

echo "=== SUMMARY $(date -u)"
for f in "$B"/ours-*.log "$B"/ista-*.log; do
    echo "== $f"
    grep -aE "Final estimate: PPL|^\s*400\s|Final Winogrande score" "$f" | tail -3
done
echo BENCH_DONE
