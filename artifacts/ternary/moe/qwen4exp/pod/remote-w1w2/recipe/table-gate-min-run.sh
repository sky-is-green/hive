#!/bin/bash
# table-gate-min-run.sh - runs ON the pod: MINIMAL gate for the Q4X160 table
# export (no code arms):
#   deps -> clone fork -> 6 patches -> CUDA build -> release download + sha ->
#   smoke -> KLD(release) -> host preflight -> export (Q4X160) -> KLD(table)
#   -> PPL 580 / HS 400 / WG 400 (table) -> TABLE_GATE_DONE.
# The KLD teacher park and tokens are uploaded to /root/kld-park/.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
export TMPDIR=/tmp
export TABLE_TYPE=q4x160
B=/workspace/bench
mkdir -p "$B"
LOG=$B/table-gate-min-run.log
exec >> "$LOG" 2>&1
echo "=== table-gate-min-run start $(date -u) type=$TABLE_TYPE"

REPO=/workspace/llama-qwen4exp
MODELS=/root/models
OURS=$MODELS/Scion-FlashNext-176B-A6B-Q6K.gguf
OURS_SHA=236c6ef96f0a5c3bbf660101c51cb1fdcd6f63724d9341a3d5aedc568bd08cec
KLD_PARK=/root/kld-park

echo "--- deps"
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq rsync ninja-build cmake build-essential unzip git >/dev/null 2>&1
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
cmake --build build -j"$(nproc)" --target llama-perplexity llama-completion llama-kld \
    > build.log 2>&1 || { echo BUILD_FAILED; tail -40 build.log; exit 1; }
echo "build OK $(date -u)"

# shellcheck source=pod-build-table.sh
source /root/pod-build-table.sh

echo "--- downloads $(date -u)"
mkdir -p "$MODELS"
dl_release || exit 1
echo "$OURS_SHA  $OURS" | sha256sum -c - >/dev/null 2>&1 && echo "release sha256 OK" || { echo SHA_FAILED; exit 1; }
curl -sL -o "$B/wiki.zip" https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip || { echo DL_WIKI_FAILED; exit 1; }
unzip -oq "$B/wiki.zip" -d "$B" || { echo UNZIP_FAILED; exit 1; }
WIKI=$(find "$B" -name wiki.test.raw | head -1)
curl -sL -o "$B/hellaswag_val_full.txt" https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt || { echo DL_HS_FAILED; exit 1; }
curl -sL -o "$B/winogrande-debiased-eval.csv" https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp/raw/main/winogrande-debiased-eval.csv || { echo DL_WG_FAILED; exit 1; }
[ -s "$KLD_PARK/tokens.bin" ] && [ -s "$KLD_PARK/teacher/w0.bin" ] || { echo KLD_PARK_MISSING; exit 1; }
echo "wiki $WIKI; park $(ls "$KLD_PARK/teacher" | wc -l) windows"

echo "--- release smoke (full GPU) $(date -u)"
timeout 900 ./build/bin/llama-completion -m "$OURS" -ngl 99 -c 512 -t 8 --temp 0 -n 8 -no-cnv \
    -p "The capital of France is" > "$B/smoke-release.log" 2>&1
grep -q "Paris" "$B/smoke-release.log" || { echo SMOKE_FAILED; tail -c 300 "$B/smoke-release.log"; exit 1; }
echo "release smoke OK"

echo "--- KLD(release) $(date -u)"
timeout 3600 ./build/bin/llama-kld -m "$OURS" --tokens "$KLD_PARK/tokens.bin" \
    --teacher-dir "$KLD_PARK/teacher" --windows 8 --seq 512 -ngl 99 -t 8 \
    --out "$B/kld-release.json" > "$B/kld-release.log" 2>&1
kld_rc=$?
echo "kld release rc=$kld_rc $(grep -a '^KLD:' "$B/kld-release.log" | tail -1)"
[ "$kld_rc" -eq 0 ] || { echo KLD_RELEASE_FAILED; tail -c 600 "$B/kld-release.log"; exit 1; }

build_table || exit 1

echo "--- KLD(table) $(date -u)"
timeout 3600 ./build/bin/llama-kld -m "$TABLE" --tokens "$KLD_PARK/tokens.bin" \
    --teacher-dir "$KLD_PARK/teacher" --windows 8 --seq 512 -ngl 99 -t 8 \
    --out "$B/kld-table.json" > "$B/kld-table.log" 2>&1
kld_rc=$?
echo "kld table rc=$kld_rc $(grep -a '^KLD:' "$B/kld-table.log" | tail -1)"
[ "$kld_rc" -eq 0 ] || { echo KLD_TABLE_FAILED; tail -c 600 "$B/kld-table.log"; exit 1; }

echo "--- tiny PPL dry-run $(date -u)"
timeout 900 ./build/bin/llama-perplexity -m "$TABLE" -f "$WIKI" -c 512 --chunks 4 -ngl 99 -t 8 \
    > "$B/tiny-ppl-table.log" 2>&1
[ "$?" -eq 0 ] || { echo TINY_PPL_FAILED; tail -8 "$B/tiny-ppl-table.log"; exit 1; }
echo "tiny ppl OK"

echo "--- table PPL 580 $(date -u)"
timeout 10800 ./build/bin/llama-perplexity -m "$TABLE" -f "$WIKI" -c 512 --chunks 580 \
    -ngl 99 -t 8 > "$B/table-ppl.log" 2>&1
echo "ppl rc=$? $(date -u)"

echo "--- table HellaSwag 400 $(date -u)"
timeout 7200 ./build/bin/llama-perplexity -m "$TABLE" -f "$B/hellaswag_val_full.txt" -c 512 -ngl 99 -t 8 \
    --hellaswag --hellaswag-tasks 400 > "$B/table-hellaswag.log" 2>&1
echo "hs rc=$? $(date -u)"

echo "--- table Winogrande 400 $(date -u)"
timeout 7200 ./build/bin/llama-perplexity -m "$TABLE" -f "$B/winogrande-debiased-eval.csv" -c 512 -ngl 99 -t 8 \
    --winogrande --winogrande-tasks 400 > "$B/table-winogrande.log" 2>&1
echo "wg rc=$? $(date -u)"

echo "=== SUMMARY $(date -u)"
grep -a "Final estimate: PPL" "$B/table-ppl.log" | tail -1
grep -aE "^\s*400\s" "$B/table-hellaswag.log" | tail -1
grep -a "Final Winogrande score" "$B/table-winogrande.log" | tail -1
python3 - "$B" <<'EOF'
import json, sys
b = sys.argv[1]
for tag in ("release", "table"):
    d = json.load(open(f"{b}/kld-{tag}.json"))
    print(f"KLD {tag}: mean {d['mean']:.4f} p50 {d['p50']:.4f} p90 {d['p90']:.4f} "
          f"p99 {d['p99']:.4f} p99.9 {d['p99.9']:.4f} max {d['max']:.4f} "
          f"top1 {d['top1_agreement']:.4f}")
EOF
sha256sum "$TABLE" | awk '{print "table sha256 " $1}'
stat -c "table size %s" "$TABLE"
echo "TABLE_GATE_DONE $(date -u)"
