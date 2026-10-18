#!/bin/bash
# Short probe: get Bonsai-2 27B onto card 1 (llama.cpp HIP) while the 35B holds card 0, then let the
# two generate at the same time on a few prompts and read the rates out of the engines' own logs.
#
#   bash experiments/cascade/run_bonsai_gpu_probe.sh 2>&1 | tee -a <out>/gpu-probe.log
#
# Why the previous window failed: Bonsai was launched with
#   CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 ROCR_VISIBLE_DEVICES=1
# and its log said "ggml_cuda_init: failed to initialize ROCm: no ROCm-capable device is detected",
# so it silently fell back to CPU at 7 tok/s.  ROCR_VISIBLE_DEVICES renumbers the cards, and
# HIP_VISIBLE_DEVICES is then applied against that already-filtered list, so 1+1 selects nothing.
# Variants are therefore tried one at a time and a variant is accepted only when its own log shows
# the layers actually offloaded — asserting the fix instead of assuming it.
#
# The trap restores scion-flashnext no matter how this exits, and reaches the engines by port as
# well as by PID (an earlier TERM aimed at the wrong PID left the service down).
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work
PY=$W/hivebench/.venv/bin/python
OUT=${WINDOW_OUT:-$W/hivebench/experiments/cascade/results/dual-generation-20261018}
MODEL35B=${MODEL35B:-/home/penis/models/scion-strata/35b-a3b/Scion-35B-A3B-PQ2_0-corr.gguf}
MODEL27B=${MODEL27B:-$W/hivebench/artifacts/ternary/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf}
LLAMA=${LLAMA:-/home/penis/llama.cpp/build/bin/llama-server}
BENCH=${BENCH:-$OUT/organic-66.jsonl}
LIMIT=${LIMIT:-4}
MAXTOK=${MAXTOK:-320}
SERVICE=scion-flashnext
P35=8082
P27=8083
mkdir -p "$OUT"

for f in "$MODEL35B" "$MODEL27B" "$LLAMA" "$BENCH"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

SRV35=""
SRV27=""

vram_free_mib() {   # $1 = card index
    local d=/sys/class/drm/card$1/device
    awk -v t="$(cat $d/mem_info_vram_total 2>/dev/null || echo 0)" \
        -v u="$(cat $d/mem_info_vram_used 2>/dev/null || echo 0)" \
        'BEGIN{printf "%d", (t-u)/1048576}'
}

port_pids() { ss -lntp 2>/dev/null | grep -E ':('"$1"')' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u; }

kill_engines() {
    local p
    for p in $(port_pids "$P35|$P27"); do kill -TERM "$p" 2>/dev/null; done
    for p in "$SRV35" "$SRV27"; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null; done
    sleep 5
    for p in $(port_pids "$P35|$P27"); do kill -9 "$p" 2>/dev/null; done
}

restore() {
    local rc=$?
    trap - EXIT INT TERM
    kill_engines
    echo "== $(date -Is) restarting $SERVICE"
    systemctl --user start "$SERVICE"
    local n
    for n in $(seq 1 120); do
        if curl -s --max-time 5 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"loaded": true'; then
            echo "== $(date -Is) $SERVICE healthy"; break
        fi
        sleep 2
    done
    [ "$n" -ge 120 ] && echo "== $(date -Is) WARNING: $SERVICE not healthy after 240 s"
    echo "== $(date -Is) done rc=$rc"
}
trap restore EXIT INT TERM

wait_health() {   # $1 port, $2 label, $3 tries
    local n
    for n in $(seq 1 "${3:-200}"); do
        local code
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$1/health" || true)
        [ "$code" = "200" ] && return 0
        sleep 2
    done
    return 1
}

# "offloaded N/M layers to GPU" with N > 0 is the proof the layers reached the card; the exact N/M is
# echoed so a partial offload is visible rather than rounded up to success.
offload_ok() {
    local line a
    line=$(grep -oE 'offloaded [0-9]+/[0-9]+ layers to GPU' "$1" 2>/dev/null | tail -1)
    [ -n "$line" ] || return 1
    a=${line#offloaded }; a=${a%%/*}
    [ "$a" -gt 0 ]
}

echo "== $(date -Is) start (pid $$)"
echo "== $(date -Is) VRAM free before: card0 $(vram_free_mib 0) MiB, card1 $(vram_free_mib 1) MiB"
echo "== $(date -Is) stopping $SERVICE"
systemctl --user stop "$SERVICE" || true
sleep 3

# ---- Phase 1: pin Bonsai to card 1, proving it from the log rather than trusting the env ----
# --device is llama.cpp's own selector and takes the names --list-devices prints (ROCm0/ROCm1), so it
# avoids the visibility variables that renumber the cards underneath each other.
V_ENV=("" "" "HIP_VISIBLE_DEVICES=1")
V_ARGS=("-dev ROCm1" "-ts 0,1" "")
BONSAI_OK=0
for i in 0 1 2; do
    log="$OUT/bonsai-try$i.log"
    : > "$log"
    echo "== $(date -Is) Bonsai attempt $((i + 1)): env='${V_ENV[$i]}' args='${V_ARGS[$i]}'"
    # shellcheck disable=SC2086
    env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES ${V_ENV[$i]} \
        "$LLAMA" -m "$MODEL27B" -ngl 999 -c 4096 -np 1 -t 8 \
        --host 127.0.0.1 --port "$P27" --jinja --no-warmup ${V_ARGS[$i]} \
        > "$log" 2>&1 &
    SRV27=$!
    if ! wait_health "$P27" bonsai 200; then
        echo "== $(date -Is) attempt $((i + 1)): no health endpoint"; kill_engines; SRV27=""; continue
    fi
    if offload_ok "$log"; then
        echo "== $(date -Is) attempt $((i + 1)): OFFLOADED, card1 free $(vram_free_mib 1) MiB"
        grep -oE 'offloaded [0-9]+/[0-9]+ layers to GPU' "$log" | tail -1
        BONSAI_OK=1
        break
    fi
    reason=$(grep -oE 'no ROCm-capable device is detected|offloaded [0-9]+/[0-9]+ layers to GPU' "$log" | tail -1)
    echo "== $(date -Is) attempt $((i + 1)): no GPU offload -- ${reason:-nothing about devices in the log}"
    kill_engines
    SRV27=""
done

if [ "$BONSAI_OK" != 1 ]; then
    echo "== $(date -Is) FAIL: Bonsai refused the GPU in all three variants" >&2
    exit 1
fi

# ---- Phase 2: the 35B on card 0, alongside it ----
cat > "$OUT/serve35b.json" <<EOF
{
 "exe": "$W/strata/build-hip/strata-q36",
 "args": ["--model", "$MODEL35B", "--max-context", "4096", "--chunk", "512"],
 "cwd": "$W/strata",
 "tokenizer": "/home/penis/models/scion-strata/pack-35b/tokenizer",
 "model_name": "scion-35b-a3b-pq2_0-corr",
 "log": "$OUT/engine35b.log",
 "lib_dirs": ["/opt/rocm/lib"],
 "port": $P35,
 "backend": "hip",
 "gpu": [0],
 "gpus_asked": true
}
EOF

echo "== $(date -Is) starting the 35B on card 0 (port $P35)"
"$PY" "$W/strata/serve/server.py" --engine strata --config "$OUT/serve35b.json" --port "$P35" \
    > "$OUT/server35b.log" 2>&1 &
SRV35=$!
if ! wait_health "$P35" 35b 200; then
    echo "== $(date -Is) FAIL: the 35B did not come up" >&2
    tail -20 "$OUT/server35b.log" >&2
    exit 1
fi
echo "== $(date -Is) both up — VRAM free: card0 $(vram_free_mib 0) MiB, card1 $(vram_free_mib 1) MiB"

# ---- Phase 3: both churning on the same prompts, at the same time ----
echo "== $(date -Is) dual generation, $LIMIT prompts, both models fired together"
timeout 900 "$PY" "$W/hivebench/experiments/cascade/dual_generation.py" \
    --a "http://127.0.0.1:$P35" --a-name 35b \
    --b "http://127.0.0.1:$P27" --b-name bonsai27b \
    --bench "$BENCH" --limit "$LIMIT" --max-tokens "$MAXTOK" --workers 1 \
    --out "$OUT" \
    --tasks "$W/hivebench/experiments/cascade/tasks-bench.json" \
            "$W/hivebench/experiments/cascade/tasks-community.json" \
    > "$OUT/dual.log" 2>&1
rc=$?

echo "== $(date -Is) rates as the engines reported them"
echo "-- 35B (card 0) --"
grep -oE '\[strata\] done: [0-9]+ tokens in [0-9]+ s \([0-9.]+ tok/s\)' "$OUT/server35b.log" | tail -"$LIMIT"
echo "-- Bonsai-2 27B (card 1) --"
grep -oE 'n_gen = +[0-9]+, tg = +[0-9.]+ t/s' "$OUT/bonsai-try$i.log" | tail -"$LIMIT"
tail -20 "$OUT/dual.log"
exit "$rc"
