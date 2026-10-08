#!/bin/sh
# Test 1: generation-context monitor replication on Scion.
# Waits for the bonsai-community unit and the sidecar-parity chain, then
# builds context windows from correction-samples.jsonl (CPU tokenize server),
# dumps per-position hidden states (GPU) and evaluates trace-relative readouts.
#   systemd-run --user --unit=scion-ctx --collect /bin/sh <this>
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
M=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/scion/Scion-35B-A3B-PQ2_0-corr.gguf
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
B=/home/penis/llama.cpp/build/bin/llama-server
P=/home/penis/llama.cpp/build/bin/test-mtp-probe
R=$W/experiments/cascade/results/jev-readout-20261006
D=$W/experiments/cascade/streams/scion-bench-monitor-ctx
cd "$W" || exit 1
exec >> "$R/monitor-ctx-scion.log" 2>&1

echo "== wait for bonsai-community + sidecar-parity $(date -Is)"
while systemctl --user is-active --quiet bonsai-community.service \
   || systemctl --user is-active --quiet sidecar-parity.service; do
    sleep 60
done

echo "== start tokenize server $(date -Is)"
export HIP_VISIBLE_DEVICES=""
"$B" -m "$M" -ngl 0 -c 1024 --host 127.0.0.1 --port 8898 \
    > /tmp/opencode/scion-ctx-tok.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 180); do
    if curl -sf http://127.0.0.1:8898/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
if [ "$up" != 1 ]; then
    echo "tokenize server failed"
    kill $SRV 2>/dev/null
    exit 1
fi

echo "== build context windows $(date -Is)"
"$PY" experiments/cascade/build_context_windows.py --base http://127.0.0.1:8898 \
    --samples "$R/correction-samples.jsonl" \
    --tasks experiments/cascade/tasks-bench.json --out "$D" \
    || echo "FAILED build"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null

N=$("$PY" -c "import json;print(len(json.load(open('$D/meta.json'))['order']))" 2>/dev/null)
SEQ=$("$PY" -c "import json;print(json.load(open('$D/meta.json'))['seq'])" 2>/dev/null)
if [ -z "${N:-}" ]; then echo "no windows built"; exit 1; fi
mkdir -p "$D/dump"
rm -f "$D"/dump/*.bin

echo "== hidden dump ($N x $SEQ, GPU) $(date -Is)"
export HIP_VISIBLE_DEVICES=1
"$P" "$M" "$D/tokens.bin" "$N" "$SEQ" "$D/dump" 999 || echo "FAILED dump"

echo "== trace-relative eval $(date -Is)"
"$PY" experiments/cascade/monitor_context_eval.py --windows "$D" \
    --out "$R/monitor-context-scion.json" || echo "FAILED eval"
echo "== done $(date -Is)"
