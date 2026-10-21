#!/bin/bash
# Both-tier window: Scion-35B-A3B on card 0, Ternary-Bonsai-2 27B on card 1, BOTH served by strata.
#
#   bash experiments/cascade/run_dual_strata_window.sh 2>&1 | tee -a <out>/dual-strata.log
#
# Engine choice, and why it is not one binary for both:
#   * the 35B pack is `qwen35moe`, run by the qwen35 engine path (`strata-q36`, worktree `strata`)
#   * the Bonsai pack is a DENSE `qwen35` GGUF, run by `strata-dense` (worktree `strata-bonsai2`,
#     branch `bonsai2-dense`) - `strata-q36` refuses it by name ("architecture is 'qwen35', this
#     engine requires 'qwen4exp'"), because the family enum on the serving branch knows only
#     qwen4exp and qwen35moe.
# Both are the same project, the same serve/server.py and the same HTTP protocol, so the tier-2
# harness hits identical endpoints.
#
# The 176B comes down because it layer-splits across both cards and holds an expert cache on each
# (card0 18.5 GiB, card1 20.1 GiB of 20.5 GiB) - there is no room for a second engine beside it.
# The trap restores it; it reaches the engines by port as well as by PID, because an earlier
# window's TERM landed on a neighbouring PID and left the serve down.
set -u
W=<work>
OUT=${WINDOW_OUT:-$W/hivebench/experiments/cascade/results/dual-generation-20261018}
MODEL35B=${MODEL35B:-~/models/scion-strata/35b-a3b/Scion-35B-A3B-PQ2_0-corr.gguf}
MODEL27B=${MODEL27B:-<home>/Desktop/models/hub/Ternary-Bonsai-2-27B-PQ2_0.gguf}
TOK35B=${TOK35B:-~/models/scion-strata/pack-35b/tokenizer}
TOK27B=${TOK27B:-$W/strata-bonsai2/data/packs/qwen3.8-27b/tokenizer}
Q36=$W/strata/build-hip/strata-q36
DENSE=$W/strata-bonsai2/build-hip/strata-dense
BENCH=${BENCH:-$OUT/organic-66.jsonl}
LIMIT=${LIMIT:-4}
MAXTOK=${MAXTOK:-320}
SERVICE=scion-flashnext
P35=8082
P27=8083
SPY35=$W/strata/.venv/bin/python
[ -x "$SPY35" ] || SPY35=$W/hivebench/.venv/bin/python
SPY27=$W/strata-bonsai2/.venv/bin/python
[ -x "$SPY27" ] || SPY27=$SPY35
mkdir -p "$OUT"

for f in "$MODEL35B" "$MODEL27B" "$Q36" "$DENSE" "$TOK35B/vocab.json" "$TOK27B/vocab.json" "$BENCH"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

SRV35=""
SRV27=""

vram_free_mib() {
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
    sleep 6
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

wait_health() {
    local n code
    for n in $(seq 1 "${2:-300}"); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$1/health" || true)
        [ "$code" = "200" ] && return 0
        sleep 2
    done
    return 1
}

cat > "$OUT/serve27b.json" <<EOF
{
 "exe": "$DENSE",
 "args": ["--native", "$MODEL27B", "--context", "4096", "--no-prefill"],
 "cwd": "$W/strata-bonsai2",
 "tokenizer": "$TOK27B",
 "model_name": "bonsai-2-27b-dense",
 "log": "$OUT/engine27b.log",
 "lib_dirs": [],
 "env": {"STRATA_DENSE_GRAPH": "1"},
 "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1},
 "port": $P27,
 "backend": "hip",
 "gpu": [1],
 "gpus_asked": true
}
EOF

cat > "$OUT/serve35b.json" <<EOF
{
 "exe": "$Q36",
 "args": ["--model", "$MODEL35B", "--max-context", "4096", "--chunk", "512"],
 "cwd": "$W/strata",
 "tokenizer": "$TOK35B",
 "model_name": "scion-35b-a3b-pq2_0-corr",
 "log": "$OUT/engine35b.log",
 "lib_dirs": ["/opt/rocm/lib"],
 "port": $P35,
 "backend": "hip",
 "gpu": [0],
 "gpus_asked": true
}
EOF

echo "== $(date -Is) start (pid $$)"
echo "== $(date -Is) VRAM free before: card0 $(vram_free_mib 0) MiB, card1 $(vram_free_mib 1) MiB"
echo "== $(date -Is) stopping $SERVICE"
systemctl --user stop "$SERVICE" || true
sleep 5

# ---- tier 1: the dense 27B on card 1, started first so its placement is unambiguous ----
echo "== $(date -Is) starting 27B (strata-dense, card 1, port $P27)"
"$SPY27" "$W/strata-bonsai2/serve/server.py" --engine strata --config "$OUT/serve27b.json" --port "$P27" \
    > "$OUT/server27b.log" 2>&1 &
SRV27=$!
if ! wait_health "$P27" 300; then
    echo "== $(date -Is) FAIL: 27B never became healthy" >&2
    tail -25 "$OUT/engine27b.log" "$OUT/server27b.log" >&2
    exit 1
fi
echo "== $(date -Is) 27B healthy; VRAM free: card0 $(vram_free_mib 0) MiB, card1 $(vram_free_mib 1) MiB"

# ---- tier 2: the 35B on card 0 ----
echo "== $(date -Is) starting 35B (strata-q36, card 0, port $P35)"
"$SPY35" "$W/strata/serve/server.py" --engine strata --config "$OUT/serve35b.json" --port "$P35" \
    > "$OUT/server35b.log" 2>&1 &
SRV35=$!
if ! wait_health "$P35" 300; then
    echo "== $(date -Is) FAIL: 35B never became healthy" >&2
    tail -25 "$OUT/engine35b.log" "$OUT/server35b.log" >&2
    exit 1
fi
echo "== $(date -Is) 35B healthy; VRAM free: card0 $(vram_free_mib 0) MiB, card1 $(vram_free_mib 1) MiB"

# ---- both churning on the same prompts at the same time ----
echo "== $(date -Is) dual generation: $LIMIT prompts, both engines fired together"
timeout 2400 "$W/hivebench/.venv/bin/python" "$W/hivebench/experiments/cascade/dual_generation.py" \
    --a "http://127.0.0.1:$P35" --a-name 35b \
    --b "http://127.0.0.1:$P27" --b-name bonsai27b \
    --bench "$BENCH" --limit "$LIMIT" --max-tokens "$MAXTOK" --workers 1 \
    --out "$OUT" \
    --tasks "$W/hivebench/experiments/cascade/tasks-bench.json" \
            "$W/hivebench/experiments/cascade/tasks-community.json" \
    > "$OUT/dual.log" 2>&1
rc=$?

echo "== $(date -Is) rates as the engines reported them"
echo "-- 35B (strata-q36, card 0) --"
grep -oE '\[strata\] done: [0-9]+ tokens in [0-9]+ s \([0-9.]+ tok/s\)' "$OUT/server35b.log" | tail -"$LIMIT"
echo "-- 27B (strata-dense, card 1) --"
grep -oE '\[strata\] done: [0-9]+ tokens in [0-9]+ s \([0-9.]+ tok/s\)' "$OUT/server27b.log" | tail -"$LIMIT"
echo "-- dual summary --"
tail -30 "$OUT/dual.log"
exit "$rc"
