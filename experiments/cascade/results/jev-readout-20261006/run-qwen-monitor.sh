#!/bin/sh
# Test 2: non-ternary monitor replication on Qwen3.8-27B IQ4_XS.
# Waits for bonsai-community, sidecar-parity and scion-ctx; serves the model,
# generates 100x4 bench samples, builds context windows, dumps hidden states
# (GPU) and evaluates trace-relative readouts.
#   systemd-run --user --unit=qwen-ctx --collect /bin/sh <this>
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
MQ=/home/penis/Downloads/Qwen3.8-27B-UD-IQ4_XS.gguf
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
B=/home/penis/llama.cpp/build/bin/llama-server
P=/home/penis/llama.cpp/build/bin/test-mtp-probe
R=$W/experiments/cascade/results/jev-readout-20261006
D=$W/experiments/cascade/streams/qwen-bench-monitor-ctx
S=$R/qwen-monitor-samples.jsonl
cd "$W" || exit 1
exec >> "$R/monitor-ctx-qwen.log" 2>&1

echo "== wait for bonsai-community + sidecar-parity + scion-ctx $(date -Is)"
while systemctl --user is-active --quiet bonsai-community.service \
   || systemctl --user is-active --quiet sidecar-parity.service \
   || systemctl --user is-active --quiet scion-ctx.service; do
    sleep 60
done

echo "== start qwen server $(date -Is)"
export HIP_VISIBLE_DEVICES=1
"$B" -m "$MQ" -ngl 99 -c 8192 --host 127.0.0.1 --port 8896 -fa auto \
    > /tmp/opencode/qwen-ctx-srv.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 150); do
    if curl -sf http://127.0.0.1:8896/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
if [ "$up" != 1 ]; then
    echo "qwen server failed"
    kill $SRV 2>/dev/null
    exit 1
fi

echo "== generate 4 samples/task $(date -Is)"
"$PY" experiments/cascade/gen_bench_samples.py --server http://127.0.0.1:8896 \
    --tasks experiments/cascade/tasks-bench.json --out "$S" \
    --samples 4 --temperature 0.7 --max-tokens 700 --thinking \
    || echo "FAILED gen"

echo "== build context windows $(date -Is)"
"$PY" experiments/cascade/build_context_windows.py --base http://127.0.0.1:8896 \
    --samples "$S" --tasks experiments/cascade/tasks-bench.json --out "$D" \
    --thinking || echo "FAILED build"

kill $SRV 2>/dev/null
wait $SRV 2>/dev/null

N=$("$PY" -c "import json;print(len(json.load(open('$D/meta.json'))['order']))" 2>/dev/null)
SEQ=$("$PY" -c "import json;print(json.load(open('$D/meta.json'))['seq'])" 2>/dev/null)
if [ -z "${N:-}" ]; then echo "no windows built"; exit 1; fi
mkdir -p "$D/dump"
rm -f "$D"/dump/*.bin

echo "== hidden dump ($N x $SEQ, GPU) $(date -Is)"
export HIP_VISIBLE_DEVICES=1
"$P" "$MQ" "$D/tokens.bin" "$N" "$SEQ" "$D/dump" 999 || echo "FAILED dump"

echo "== trace-relative eval $(date -Is)"
"$PY" experiments/cascade/monitor_context_eval.py --windows "$D" \
    --out "$R/monitor-context-qwen.json" || echo "FAILED eval"
echo "== done $(date -Is)"
