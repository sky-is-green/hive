#!/bin/sh
# Bonsai attribution audit: controlled own-vs-other attribution on identical
# candidate text (Bonsai own answers vs Strata recorded answers), plus the
# balanced in-format refit.  Serially after the main bonsai-run unit.
#   systemd-run --user --unit=bonsai-attrib /bin/sh <this>
set -u
W=<work>/hivebench
PY=<work>/.venv-rocm/bin/python
B=~/llama.cpp/build/bin/llama-server
M=~/Downloads/Ternary-Bonsai-2-27B-PQ2_0.gguf
R=$W/experiments/cascade/results/jev-readout-20261006
export HIP_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
cd "$W" || exit 1

echo "== start bonsai server $(date -Is)"
"$B" -m "$M" -ngl 99 -c 8192 --host 127.0.0.1 --port 8897 \
    --embeddings --pooling last --embd-normalize -1 -fa auto --jinja \
    > /tmp/opencode/bonsai-srv-attrib.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
if [ "$up" != 1 ]; then echo "server did not start"; kill $SRV 2>/dev/null; exit 1; fi

echo "== attribution audit $(date -Is)"
"$PY" experiments/cascade/attribution_audit.py --server http://127.0.0.1:8897 \
    --states "$R/probe-states-bonsai27b.npz" \
    --answers "$R/bonsai-answers.json" \
    --out "$R/attribution-bonsai-audit.json" \
    --states-out "$R/attribution-bonsai-states.npz" \
    > "$R/attribution-bonsai-audit.log" 2>&1 || echo "FAILED audit"

echo "== attribution fix $(date -Is)"
"$PY" experiments/cascade/attribution_fix.py \
    --states "$R/attribution-bonsai-states.npz" \
    --audit "$R/attribution-bonsai-audit.json" \
    --out "$R/attribution-bonsai-fix.json" \
    > "$R/attribution-bonsai-fix.log" 2>&1 || echo "FAILED fix"

echo "== stop server $(date -Is)"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "== done $(date -Is)"
