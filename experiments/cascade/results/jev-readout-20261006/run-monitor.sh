#!/bin/sh
# L3 monitor dataset: tokenize the correction samples (CPU server), dump
# per-position hidden states with test-mtp-probe (GPU), fit horizon probes.
# Run after correction-run completes:
#   systemd-run --user --unit=monitor-chain /bin/sh <this>
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
T=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/scion
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
B=/home/penis/llama.cpp/build/bin/llama-server
P=/home/penis/llama.cpp/build/bin/test-mtp-probe
R=$W/experiments/cascade/results/jev-readout-20261006
M=$W/experiments/cascade/streams/scion-bench-monitor
cd "$W" || exit 1

echo "== CPU server for tokenize $(date -Is)"
export HIP_VISIBLE_DEVICES=""
"$B" -m "$T/Scion-35B-A3B-PQ2_0-corr.gguf" -ngl 0 -c 1024 \
    --host 127.0.0.1 --port 8898 > /tmp/opencode/scion-srv-tok.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8898/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
if [ "$up" != 1 ]; then echo "tokenize server did not start"; kill $SRV 2>/dev/null; exit 1; fi

echo "== build windows $(date -Is)"
"$PY" experiments/cascade/build_monitor_windows.py --base http://127.0.0.1:8898 \
    --samples "$R/correction-samples.jsonl" --out "$M" \
    > "$R/build-monitor-windows.log" 2>&1 || echo "FAILED build"

kill $SRV 2>/dev/null
wait $SRV 2>/dev/null

N=$("$PY" -c "import json;print(len(json.load(open('$M/meta.json'))['order']))")
SEQ=$("$PY" -c "import json;print(json.load(open('$M/meta.json'))['seq'])")
echo "== hidden dump ($N windows x $SEQ, GPU) $(date -Is)"
export HIP_VISIBLE_DEVICES=1
"$P" "$T/Scion-35B-A3B-PQ2_0-corr.gguf" "$M/tokens.bin" "$N" "$SEQ" "$M/dump" 999 \
    > "$R/test-mtp-probe.log" 2>&1 || echo "FAILED dump"

echo "== monitor probe $(date -Is)"
"$PY" experiments/cascade/monitor_probe.py --windows "$M" \
    > "$R/monitor-probe.log" 2>&1 || echo "FAILED probe"
tail -20 "$R/monitor-probe.log"
echo "== done $(date -Is)"
