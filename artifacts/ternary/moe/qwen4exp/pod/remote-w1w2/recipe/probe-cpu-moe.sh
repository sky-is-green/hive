#!/bin/bash
# probe-cpu-moe.sh - local timing probe of the CPU-MoE serving config.
# Measures one 24-token completion plus the read_bytes delta (the readahead
# amplification trap).  Run only when the desktop can take the I/O pressure.
set -u
Q=/home/penis/Desktop/work/hivebench/artifacts/ternary/moe/qwen4exp
BIN=/home/penis/Desktop/work/llama-qwen4exp/build-q4exp-hip/bin/llama-server
LOG=$Q/pod/remote-w1w2/cpu-moe-probe-server.log
export HIP_VISIBLE_DEVICES=1
rm -f "$LOG"
"$BIN" -m "$Q/qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf" -c 512 -ngl 99 \
    --cpu-moe -ot "per_layer_token_embd.weight=CPU" -fa on \
    --host 127.0.0.1 --port 8802 --temp 0 > "$LOG" 2>&1 &
PID=$!
echo "probe server pid $PID"
ok=0
for i in $(seq 1 360); do
    curl -sf http://127.0.0.1:8802/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$PID" 2>/dev/null || { echo "server died during load"; exit 1; }
    sleep 5
done
[ "$ok" = 1 ] || { echo "health timeout"; kill "$PID"; exit 1; }
echo "healthy after $((i * 5))s"
SP=$(pgrep -f "llama-server.*qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf" | head -1)
r0=$(grep read_bytes /proc/$SP/io | awk '{print $2}')
t0=$(date +%s)
C1=$(curl -s --max-time 3600 -X POST http://127.0.0.1:8802/completion \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "The capital of France is", "n_predict": 24, "temperature": 0}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content",""))')
t1=$(date +%s); r1=$(grep read_bytes /proc/$SP/io | awk '{print $2}')
echo "REQ1 wall=$((t1-t0))s io=$(( (r1-r0)/1000000 ))MB"
echo "REQ1 content: $(printf '%q' "$C1")"
echo "--- server timings"
grep -aE "prompt eval time|eval time|prompt processing" "$LOG" | tail -8
kill "$PID" 2>/dev/null
echo "PROBE DONE"
