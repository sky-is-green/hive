#!/usr/bin/env bash
# Safe, reproducible pure-CPU partner embedding probe.
#
# This is the guard that the first Scion extraction lacked.  It never asks for
# GPU layers, hides every GPU from llama.cpp, refuses to start without RAM
# headroom, samples host/VRAM/176B health during the run, and stops the partner
# if available memory collapses or the 176B serve stays unhealthy.
#
# Usage:
#   run_partner_probe.sh MODEL.gguf BENCH.jsonl NAME OUTDIR [PORT]
#
# Environment overrides: THREADS=8 CTX=2048 MIN_FREE_GIB=12 TIMEOUT=7200
#                        LLAMA_SERVER=... PY=...
set -euo pipefail

if [ "$#" -lt 4 ]; then
    sed -n '2,18p' "$0"
    exit 2
fi

MODEL=$1
BENCH=$2
NAME=$3
OUT=$4
PORT=${5:-8899}
THREADS=${THREADS:-8}
CTX=${CTX:-2048}
MIN_FREE_GIB=${MIN_FREE_GIB:-12}
TIMEOUT=${TIMEOUT:-7200}
BIN=${LLAMA_SERVER:-/home/penis/llama.cpp/build/bin/llama-server}
PY=${PY:-.venv/bin/python}
HEALTH=${SERVE_HEALTH:-http://127.0.0.1:8081/health}
MONITOR=experiments/cascade/monitor_resources.py
PROBE=experiments/cascade/probe_scion.py

for path in "$MODEL" "$BENCH" "$BIN" "$PY" "$MONITOR" "$PROBE"; do
    if [ ! -e "$path" ]; then
        echo "missing: $path" >&2
        exit 1
    fi
done
mkdir -p "$OUT"

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "port $PORT is already listening" >&2
    exit 1
fi
MEM_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
MIN_KB=$((MIN_FREE_GIB * 1024 * 1024))
if [ "$MEM_KB" -lt "$MIN_KB" ]; then
    echo "refusing to start: ${MEM_KB}kB available, need ${MIN_KB}kB" >&2
    exit 1
fi
if ! curl -sf --max-time 3 "$HEALTH" >/dev/null; then
    echo "refusing to start: the 176B serve is not healthy" >&2
    exit 1
fi

# Pure CPU: hide every accelerator before llama.cpp initializes.  --device none
# and --no-op-offload also prevent fused host ops from silently landing on HIP.
export HIP_VISIBLE_DEVICES=
export ROCR_VISIBLE_DEVICES=
export CUDA_VISIBLE_DEVICES=

SERVER_LOG="$OUT/server.log"
PROBE_LOG="$OUT/probe.log"
taskset -c "0-$((THREADS - 1))" nice -n 19 "$BIN" \
    -m "$MODEL" -ngl 0 --device none --no-op-offload \
    -t "$THREADS" -c "$CTX" -np 1 \
    --host 127.0.0.1 --port "$PORT" --embeddings --pooling last \
    --embd-normalize -1 --no-warmup --jinja >"$SERVER_LOG" 2>&1 &
SRV=$!
MON=
GUARD=

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [ -n "${GUARD:-}" ]; then kill "$GUARD" 2>/dev/null || true; fi
    if [ -n "${MON:-}" ]; then
        kill -TERM "$MON" 2>/dev/null || true
        wait "$MON" 2>/dev/null || true
    fi
    kill -TERM "$SRV" 2>/dev/null || true
    for _ in $(seq 1 40); do
        kill -0 "$SRV" 2>/dev/null || break
        sleep 0.5
    done
    kill -KILL "$SRV" 2>/dev/null || true
    exit "$rc"
}
trap cleanup EXIT INT TERM

UP=0
for _ in $(seq 1 180); do
    if curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        UP=1
        break
    fi
    if ! kill -0 "$SRV" 2>/dev/null; then break; fi
    sleep 1
done
if [ "$UP" != 1 ]; then
    echo "partner server failed to start" >&2
    tail -40 "$SERVER_LOG" >&2 || true
    exit 1
fi

"$PY" "$MONITOR" --pid "$SRV" --out "$OUT/stability.csv" --interval 5 \
    --health-url "$HEALTH" --label "$NAME" >"$OUT/stability-monitor.out" 2>&1 &
MON=$!

(
    fails=0
    while kill -0 "$SRV" 2>/dev/null; do
        mem=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
        if [ "$mem" -lt 4194304 ]; then
            echo "$(date -Is) LOW MEM ${mem}kB; stopping partner" 
            kill -TERM "$SRV" 2>/dev/null || true
            break
        fi
        if curl -sf --max-time 2 "$HEALTH" >/dev/null 2>&1; then
            fails=0
        else
            fails=$((fails + 1))
            echo "$(date -Is) 176B health failure $fails"
        fi
        if [ "$fails" -ge 6 ]; then
            echo "$(date -Is) 176B unhealthy for 60s; stopping partner"
            kill -TERM "$SRV" 2>/dev/null || true
            break
        fi
        sleep 10
    done
) >"$OUT/guard.log" 2>&1 &
GUARD=$!

set +e
timeout "$TIMEOUT" "$PY" "$PROBE" \
    --server "http://127.0.0.1:$PORT" --bench "$BENCH" \
    --name "$NAME" --out "$OUT" >"$PROBE_LOG" 2>&1
RC=$?
set -e
echo "probe_rc=$RC"
tail -30 "$PROBE_LOG" || true
exit "$RC"
