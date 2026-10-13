#!/bin/bash
# pod-build-table.sh - source me (runs ON the pod).  Downloads the release and
# rewrites the PLE table to Q3X32 (experts preserved); verifies and smokes it.
# Requires: REPO (built fork), B, MODELS, OURS, OURS_SHA, and
# /root/{repack_experts_q2_0.py,ptq1_0.py}.
# Provides: dl_release, build_table.

RELEASE=$MODELS/Scion-FlashNext-176B-A6B-Q6K.gguf
TABLE_TYPE=${TABLE_TYPE:-q3x32}
TABLE=$MODELS/Scion-FlashNext-176B-A6B-${TABLE_TYPE}.gguf

dl_release() {
    local log="$B/dl-ours.log"
    if hf download SkyIsNotGreen/Scion-FlashNext-176B-A6B Scion-FlashNext-176B-A6B-Q6K.gguf \
            --local-dir "$MODELS" > "$log" 2>&1; then echo "dl release OK"; return 0; fi
    echo "dl release xet failed; HTTP retry"
    HF_HUB_DISABLE_XET=1 hf download SkyIsNotGreen/Scion-FlashNext-176B-A6B \
        Scion-FlashNext-176B-A6B-Q6K.gguf --local-dir "$MODELS" >> "$log" 2>&1 \
        && { echo "dl release OK (http)"; return 0; }
    echo DL_OURS_FAILED; tail -8 "$log"; return 1
}

host_preflight() {
    # refuse a pathologically slow host before the long CPU transcode
    python3 - "$RELEASE" "$TABLE_TYPE" <<'EOF' || { echo HOST_TOO_SLOW; return 1; }
import sys, time
from pathlib import Path
sys.path.insert(0, "/root")
import repack_experts_q2_0 as R
idx = R.read_index(Path(sys.argv[1]))
t = next(t for t in idx["infos"] if t.name == R.TABLE_NAME)
with open(sys.argv[1], "rb") as f:
    f.seek(idx["data_start"] + t.offset)
    data = f.read(16 * 1024 * 1024 // 90 * 90)
t0 = time.time()
R.transcode_q4_0_to_table(data, R.TABLE_TARGETS[sys.argv[2]])
dt = time.time() - t0
rate = (len(data) // 90) / dt
print(f"host preflight: {rate/1000:.0f}k rows/s", flush=True)
sys.exit(0 if rate > 80000 else 3)
EOF
}

build_table() {
    mkdir -p /root/scion-codec
    cp -f /root/ptq1_0.py /root/scion-codec/ || { echo CODEC_UPLOAD_FAILED; return 1; }
    local bytes_opt=""
    local tbl_bytes="${TABLE_BYTES:-$B/q4x160-table.bin}"
    if [ -n "${TABLE_BYTES:-}" ] && [ -s "$TABLE_BYTES" ]; then
        echo "--- table export from uploaded bytes ($(stat -c %s "$TABLE_BYTES") B) $(date -u)"
        bytes_opt="--table-bytes $TABLE_BYTES"
    elif [ -n "${TABLE_PARALLEL:-}" ]; then
        echo "--- parallel table transcode ($(nproc) cores) $(date -u)"
        python3 /root/parallel_table_transcode.py "$RELEASE" "$tbl_bytes" "$TABLE_TYPE" \
            "${TABLE_PROCS:-32}" > "$B/table-parallel.log" 2>&1 \
            || { echo TABLE_PARALLEL_FAILED; tail -20 "$B/table-parallel.log"; return 1; }
        bytes_opt="--table-bytes $tbl_bytes"
    else
        echo "--- host preflight $(date -u)"
        host_preflight || return 1
        echo "--- table export ($TABLE_TYPE) $(date -u)"
    fi
    python3 /root/repack_experts_q2_0.py \
        --in "$RELEASE" --out "$TABLE" --preserve-experts --table-type "$TABLE_TYPE" \
        --chunk-mb 32 --scion-moe /root/scion-codec --verify-tensors 4 $bytes_opt \
        > "$B/table-export.log" 2>&1 || { echo TABLE_EXPORT_FAILED; tail -20 "$B/table-export.log"; return 1; }
    ls -la "$RELEASE" "$TABLE"
    echo "--- table smoke $(date -u)"
    timeout 900 "$REPO/build/bin/llama-completion" -m "$TABLE" -ngl 99 -c 512 -t 8 --temp 0 -n 8 -no-cnv \
        -p "The capital of France is" > "$B/smoke-table.log" 2>&1
    grep -q "Paris" "$B/smoke-table.log" || { echo SMOKE_TABLE_FAILED; tail -c 300 "$B/smoke-table.log"; return 1; }
    echo "table smoke OK"
    return 0
}
