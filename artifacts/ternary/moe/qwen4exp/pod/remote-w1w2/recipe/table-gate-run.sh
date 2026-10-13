#!/bin/bash
# table-gate-run.sh - runs ON the pod: quality gates for the Q3X32 table export
# (all 512 experts intact, PLE table Q4_0 -> Q3X32):
#   deps -> clone fork -> 6 patches -> CUDA build -> release download + sha ->
#   smoke -> KLD(release) -> table export (Q3X32, PTQ1_0 preserved) ->
#   KLD(table) -> PPL 580 / HS 400 / WG 400 (table) ->
#   paired code arms (release + table, gen / hard 2048 / hard 4096) ->
#   TABLE_GATE_DONE.
# The KLD teacher park and tokens are uploaded to /root/kld-park/.
# Logs in /workspace/bench/.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
export TMPDIR=/tmp
B=/workspace/bench
P=$B/paired
mkdir -p "$B" "$P"
LOG=$B/table-gate-run.log
exec >> "$LOG" 2>&1
echo "=== table-gate-run start $(date -u)"

REPO=/workspace/llama-qwen4exp
MODELS=/root/models
OURS=$MODELS/Scion-FlashNext-176B-A6B-Q6K.gguf
OURS_SHA=236c6ef96f0a5c3bbf660101c51cb1fdcd6f63724d9341a3d5aedc568bd08cec
KLD_PARK=/root/kld-park
COM=/root/community

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
cmake --build build -j"$(nproc)" --target llama-perplexity llama-cli llama-server llama-completion llama-kld \
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

# ---- paired code arms: release and table, same pod, same caps ----
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
        --runtime "table gate" --tasks "$tasks" --phase gen --out-dir "$d" \
        > "$d/gen.log" 2>&1 || echo "FAILED gen $prefix"
    python3 "$COM/community_eval.py" --server http://127.0.0.1:8897 --concurrency 4 \
        --prefix "$prefix" --phase hard --out-dir "$d" --tasks "$tasks" \
        > "$d/hard.log" 2>&1 || echo "FAILED hard $prefix"
    python3 "$COM/community_eval.py" --server http://127.0.0.1:8897 --concurrency 4 \
        --prefix "$prefix" --phase hard --out-dir "$d-cap4096" --tasks "$tasks" \
        --hard-max-tokens 4096 > "$d-cap4096/cap4096.log" 2>&1 || echo "FAILED hard4096 $prefix"
}

echo "--- serve release + arms $(date -u)"
mkdir -p "$P/release" "$P/release-cap4096"
srv_start "$OURS" "$P/server-release.log" || { echo RELEASE_SERVER_FAILED; tail -c 800 "$P/server-release.log"; exit 1; }
run_arms "$P/release" scion-rel
srv_stop
echo "release arms done $(date -u)"

echo "--- serve table + arms $(date -u)"
mkdir -p "$P/table" "$P/table-cap4096"
srv_start "$TABLE" "$P/server-table.log" || { echo TABLE_SERVER_FAILED; tail -c 800 "$P/server-table.log"; exit 1; }
run_arms "$P/table" scion-tbl
srv_stop
echo "table arms done $(date -u)"

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
