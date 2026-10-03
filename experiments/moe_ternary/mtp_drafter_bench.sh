#!/usr/bin/env bash
# mtp_drafter_bench.sh — bench a Scion MTP drafter against a release GGUF.
#
# Baseline (no drafter) vs the drafter through llama-server on the same prompt.
# Handles both the k=1 sidecar and the mtp2 (DSpark) multi-token sidecar, which
# is server-wired via -md auto-detection (proposes k=mtp2.k tokens/step).
#
# Usage:
#   mtp_drafter_bench.sh <model.gguf> <drafter.gguf|-> [port] [n_predict]
#
# Env: HIP_VISIBLE_DEVICES (default 1), FORK (default /home/penis/llama.cpp),
#      PROMPT, N_PREDICT_OVERRIDE via the positional arg.
set -uo pipefail

MODEL=${1:?usage: mtp_drafter_bench.sh <model.gguf> <drafter.gguf|-> [port] [n_predict]}
DRAFTER=${2:?}
PORT=${3:-8137}
N_PREDICT=${4:-160}
PROMPT=${PROMPT:-"The history of science is the study of the development of"}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}
FORK=${FORK:-/home/penis/llama.cpp}
BIN=$FORK/build/bin
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"

[ -x "$BIN/llama-server" ] || { echo "missing $BIN/llama-server (build the fork)"; exit 2; }
[ -f "$MODEL" ] || { echo "missing model $MODEL"; exit 2; }

server_pid() { pgrep -f "llama-server.*--port $PORT" | head -1; }
server_stop() {
    local p; p=$(server_pid)
    [ -n "$p" ] || return 0
    kill "$p" 2>/dev/null
    for _ in $(seq 1 30); do [ -z "$(server_pid)" ] && break; sleep 1; done
    kill -9 "$(server_pid)" 2>/dev/null
}
trap server_stop EXIT INT TERM

server_start() { # $1 extra args, $2 log
    server_stop
    HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES setsid nohup "$BIN/llama-server" \
        -m "$MODEL" -c 2048 -ngl 999 -np 1 --port "$PORT" --host 127.0.0.1 \
        $1 >"$2" 2>&1 &
    local pid=$!
    for _ in $(seq 1 180); do
        grep -q 'listening on' "$2" 2>/dev/null && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}
ms_per_token() { grep 'eval time =' "$1" | grep -v 'prompt eval time' | head -1 \
    | sed -n 's/.*( *\([0-9.]*\) ms per token.*/\1/p'; }
srv_accept()  { grep -oE 'draft acceptance = [0-9.]+' "$1" | grep -oE '[0-9.]+$' | head -1; }
srv_len()     { grep -oE 'mean len = +[0-9.]+' "$1" | grep -oE '[0-9.]+$' | head -1; }
req() { curl -s -m 300 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0.0,\"cache_prompt\":false}" >/dev/null; }

OUT=$(mktemp -d)
echo "== baseline =="
server_start "" "$OUT/base.log" || { echo "baseline server failed"; exit 2; }
req; BASE=$(ms_per_token "$OUT/base.log"); echo "baseline $BASE ms/token"
server_stop

SPEED=1.0; ACC="-"
if [ "$DRAFTER" != "-" ]; then
    # both the k=1 head and the mtp2 (DSpark) sidecar are server-wired via
    # -md + auto-detection (common/speculative.cpp): the multi-token head
    # proposes k=mtp2.k tokens per step and declares n_rs_seq=k
    echo "== drafter $(basename "$DRAFTER") =="
    server_start "-md $DRAFTER" "$OUT/draft.log" || { echo "drafter server failed"; exit 2; }
    req; SPEC=$(ms_per_token "$OUT/draft.log"); ACC=$(srv_accept "$OUT/draft.log")
    LEN=$(srv_len "$OUT/draft.log")
    SPEED=$(awk -v b="$BASE" -v s="$SPEC" 'BEGIN{printf "%.3f",(s>0?b/s:0)}')
    echo "drafter $SPEC ms/token, accept ${ACC:-?}, mean len ${LEN:-?}, speedup ${SPEED}x"
    server_stop
fi

echo "---"
echo "baseline ${BASE} ms/token | drafter speedup ${SPEED}x | acceptance ${ACC}"
echo "(logs in $OUT)"
