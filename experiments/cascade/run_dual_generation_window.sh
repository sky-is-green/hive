#!/bin/bash
# Dual-generation window: 35B on card 0 and Bonsai-2 27B on card 1, churning on the same prompts.
#
#   bash experiments/cascade/run_dual_generation_window.sh 2>&1 | tee -a <out>/window.log
#
# This is the mechanical test for the tiered idea.  The serve is stopped because the 176B occupies
# both cards (layer-split + an 11.76 GiB expert cache), so tier 1 needs both cards to itself.  The
# trap is the safety net: scion-flashnext always comes back.
#
# The guard is not decorative.  An earlier attempt to co-host a second model pushed card1 down to
# 512 MiB free and took 10 GB of swap before it was stopped.  Here both engines abort if either card
# drops below VRAM_FLOOR_MIB free, which is checked on sysfs rather than a tool that may be absent.
set -u
W=<work>
PY=$W/hivebench/.venv/bin/python
OUT=${WINDOW_OUT:-$W/hivebench/experiments/cascade/results/dual-generation-20261018}
MODEL35B=${MODEL35B:-~/models/scion-strata/35b-a3b/Scion-35B-A3B-PQ2_0-corr.gguf}
MODEL27B=${MODEL27B:-$W/hivebench/artifacts/ternary/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf}
LLAMA=${LLAMA:-~/llama.cpp/build/bin/llama-server}
BENCH=${BENCH:-$OUT/organic-66.jsonl}
LIMIT=${LIMIT:-66}
SERVICE=scion-flashnext
P35=8082
P27=8083
VRAM_FLOOR_MIB=${VRAM_FLOOR_MIB:-1536}
mkdir -p "$OUT"

for f in "$MODEL35B" "$MODEL27B" "$LLAMA" "$BENCH"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

SRV35=
SRV27=
GUARD=

vram_free_mib() {   # $1 = card index
    local d=/sys/class/drm/card$1/device
    awk -v t="$(cat $d/mem_info_vram_total 2>/dev/null || echo 0)" \
        -v u="$(cat $d/mem_info_vram_used 2>/dev/null || echo 0)" \
        'BEGIN{printf "%d", (t-u)/1048576}'
}

restore() {
    local rc=$?
    trap - EXIT INT TERM
    [ -n "$GUARD" ] && kill "$GUARD" 2>/dev/null
    for p in "$SRV35" "$SRV27"; do
        [ -n "$p" ] || continue
        kill "$p" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
        kill -9 "$p" 2>/dev/null
    done
    echo "== $(date -Is) restarting $SERVICE"
    systemctl --user start "$SERVICE"
    for n in $(seq 1 120); do
        if curl -s --max-time 5 http://127.0.0.1:8081/health | grep -q '"loaded": true'; then
            echo "== $(date -Is) $SERVICE healthy"; break
        fi
        sleep 2
    done
    [ "$n" -ge 120 ] && echo "== $(date -Is) WARNING: $SERVICE not healthy after 240 s"
    echo "== $(date -Is) done rc=$rc"
}
trap restore EXIT INT TERM

echo "== $(date -Is) start (pid $$)"
echo "== $(date -Is) VRAM free: card0 $(vram_free_mib 0) MiB, card1 $(vram_free_mib 1) MiB"
echo "== $(date -Is) stopping $SERVICE"
systemctl --user stop "$SERVICE" || true
sleep 3

cat > "$OUT/serve35b.json" <<EOF
{
 "exe": "$W/strata/build-hip/strata-q36",
 "args": ["--model", "$MODEL35B", "--max-context", "4096", "--chunk", "512"],
 "cwd": "$W/strata",
 "tokenizer": "~/models/scion-strata/pack-35b/tokenizer",
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

echo "== $(date -Is) starting Bonsai-2 27B on card 1 (port $P27)"
CUDA_VISIBLE_DEVICES=1 ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 \
    "$LLAMA" -m "$MODEL27B" -ngl 999 -c 4096 -np 1 -t 8 \
    --host 127.0.0.1 --port "$P27" --jinja --no-warmup \
    > "$OUT/server27b.log" 2>&1 &
SRV27=$!

for name in 35b:$P35 27b:$P27; do
    label=${name%%:*}; port=${name##*:}
    up=0
    for n in $(seq 1 300); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$port/health" || true)
        if [ "$code" = "200" ]; then up=1; echo "== $(date -Is) $label healthy on $port"; break; fi
        n=$((n+1)); sleep 2
    done
    if [ "$up" != 1 ]; then
        echo "== $(date -Is) FAIL: $label did not come up in 600 s" >&2
        tail -25 "$OUT/server$label.log" 2>/dev/null || tail -25 "$OUT/server${label}b.log" 2>/dev/null
        exit 1
    fi
done

echo "== $(date -Is) both up; VRAM free: card0 $(vram_free_mib 0) MiB, card1 $(vram_free_mib 1) MiB"

(
    while kill -0 "$SRV35" 2>/dev/null && kill -0 "$SRV27" 2>/dev/null; do
        f0=$(vram_free_mib 0); f1=$(vram_free_mib 1)
        echo "$(date -Is) vram_free_mib card0=$f0 card1=$f1" >> "$OUT/vram.log"
        if [ "$f0" -lt "$VRAM_FLOOR_MIB" ] || [ "$f1" -lt "$VRAM_FLOOR_MIB" ]; then
            echo "$(date -Is) VRAM floor breached (card0=$f0 card1=$f1 < $VRAM_FLOOR_MIB); stopping both" \
                >> "$OUT/guard.log"
            kill -TERM "$SRV35" "$SRV27" 2>/dev/null
            break
        fi
        sleep 5
    done
) > /dev/null 2>&1 &
GUARD=$!

START=$(date +%s)
"$PY" "$W/hivebench/experiments/cascade/dual_generation.py" \
    --a "http://127.0.0.1:$P35" --a-name 35b \
    --b "http://127.0.0.1:$P27" --b-name bonsai27b \
    --bench "$BENCH" --limit "$LIMIT" --out "$OUT" \
    --tasks "$W/hivebench/experiments/cascade/tasks-bench.json" \
         "$W/hivebench/experiments/cascade/tasks-community.json" \
    > "$OUT/dual.log" 2>&1
rc=$?
echo "== $(date -Is) dual generation rc=$rc elapsed $(( $(date +%s) - START ))s"
tail -40 "$OUT/dual.log"
exit "$rc"