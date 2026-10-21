#!/bin/bash
# Window A/B: the 176B's own judge states for a harvest bench.
#
#   bash experiments/cascade/run_self_head_window.sh 2>&1 | tee -a results/.../window.log
#
# Window B (full 937-item harvest corpus):
#   WINDOW_OUT=.../results/scion176b-harvest-20261018 \
#   WINDOW_BENCH=.../results/decision-harvest-r2-20261017/harvest-bench-r2.jsonl \
#   WINDOW_NAME=scion176b-harvest \
#   bash experiments/cascade/run_self_head_window.sh
#
# Stops scion-flashnext, serves the 176B through the decision-head build (strata-dh, port 8897,
# the established port pattern), embeds the judge prompts with the same prompt the harvest
# extraction used, and restores the service whichever way it ends.  The trap is the safety net:
# the user's serve always comes back.
set -u
W=<work>
PY=$W/hivebench/.venv/bin/python
OUT=${WINDOW_OUT:-$W/hivebench/experiments/cascade/results/scion176b-self-organic-20261018}
BENCH=${WINDOW_BENCH:-$W/hivebench/experiments/cascade/results/partner-bonsai-organic-20261018/bonsai-transfer-bench.jsonl}
NAME=${WINDOW_NAME:-scion176b-transfer}
SERVICE=scion-flashnext
PORT=8897
mkdir -p "$OUT"
echo "== $(date -Is) start (pid $$)"
echo "== $(date -Is) out=$OUT bench=$BENCH name=$NAME"

SRV=
cleanup() {
    echo "== $(date -Is) stopping the decision-head server"
    if [ -n "$SRV" ]; then
        kill "$SRV" 2>/dev/null
        n=0
        while kill -0 "$SRV" 2>/dev/null && [ $n -lt 20 ]; do n=$((n + 1)); sleep 1; done
        kill -9 "$SRV" 2>/dev/null
    fi
    echo "== $(date -Is) restarting $SERVICE"
    systemctl --user start "$SERVICE"
    n=0
    while [ $n -lt 120 ]; do
        if curl -s --max-time 5 http://127.0.0.1:8081/health | grep -q '"loaded": true'; then
            echo "== $(date -Is) $SERVICE healthy"; break
        fi
        n=$((n + 1)); sleep 2
    done
    if [ $n -ge 120 ]; then echo "== $(date -Is) WARNING: $SERVICE not healthy after 240 s"; fi
    echo "== $(date -Is) done"
}
trap cleanup EXIT INT TERM

echo "== $(date -Is) stopping $SERVICE"
systemctl --user stop "$SERVICE" || true
sleep 3

cat > "$OUT/dh-server.json" <<EOF
{
 "exe": "$W/strata-dh/build-hip/strata",
 "args": [
  "--pack", "~/models/scion-strata/pack-q6k",
  "--native", "~/models/scion-strata/Scion-FlashNext-176B-A6B-Q6K.gguf",
  "--ple-gguf", "~/models/scion-strata/Scion-FlashNext-176B-A6B-Q6K.gguf",
  "--embd-gguf", "~/models/scion-strata/scion-embd-bf16.gguf",
  "--expert-profile", "<work>/strata/data/expert-profile.bin",
  "--expert-cache", "auto",
  "--prefill", "8192",
  "--spec", "4",
  "--spec-min-p", "0.5",
  "--mtp", "<work>/strata-data/mtp/rt",
  "--max-context", "4096",
  "--ple-io", "direct",
  "--mmap-experts",
  "--vram-reserve-mib", "3072",
  "--pool-workers", "15",
  "--pcie-frac", "0",
  "--pipeline-windows", "2"
 ],
 "cwd": "$W/strata-dh",
 "tokenizer": "~/models/scion-strata/pack-q6k/tokenizer",
 "model_name": "scion-flashnext-176b-a6b-q6k-dh",
 "log": "$OUT/engine.log",
 "lib_dirs": ["/opt/rocm/lib"],
 "env": {"STRATA_PREFILL_MMQ": "1", "STRATA_PREFILL_RING": "96", "STRATA_IO_THREADS": "32"},
 "port": $PORT,
 "backend": "hip",
 "gpu": [0, 1],
 "gpus_asked": true,
 "layer_split": "auto",
 "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "min_p": 0.05}
}
EOF

echo "== $(date -Is) starting the decision-head server (exe: $W/strata-dh/build-hip/strata)"
"$PY" "$W/strata-dh/serve/server.py" --engine strata --config "$OUT/dh-server.json" --port "$PORT" \
    > "$OUT/server.log" 2>&1 &
SRV=$!

n=0
while [ $n -lt 120 ]; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health" || true)
    if [ "$code" = "200" ]; then echo "== $(date -Is) server up (health 200)"; break; fi
    n=$((n + 1)); sleep 2
done
if [ $n -ge 120 ]; then
    echo "== $(date -Is) FAIL: server did not come up in 240 s"
    exit 1
fi

echo "== $(date -Is) embed: $(wc -l < "$BENCH") judge prompts from $(basename "$BENCH")"
START=$(date +%s)
"$PY" "$W/hivebench/experiments/cascade/probe_scion.py" \
    --server "http://127.0.0.1:$PORT" \
    --bench "$BENCH" \
    --out "$OUT" \
    --name "$NAME"
rc=$?
echo "== $(date -Is) probe rc=$rc elapsed $(( $(date +%s) - START ))s"
ls -lh "$OUT/probe-states-$NAME.npz" 2>/dev/null || true
echo "== $(date -Is) window complete rc=$rc"
exit "$rc"
