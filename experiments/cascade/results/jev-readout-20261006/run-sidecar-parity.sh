#!/bin/sh
# Sidecar GPU parity: engine-loaded head (test-decision-head) vs the client-side
# JSON on real community prompts.  Waits for the bonsai-community unit to finish
# (its embed phase writes bonsai-community-states.npz), then serves Bonsai on
# the fork and checks one sample per phase through /apply-template.
#   systemd-run --user --unit=sidecar-parity /bin/sh <this>
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
B=/home/penis/llama.cpp/build/bin/llama-server
M=/home/penis/Downloads/Ternary-Bonsai-2-27B-PQ2_0.gguf
R=$W/experiments/cascade/results/jev-readout-20261006
export HIP_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
cd "$W" || exit 1
exec >> "$R/sidecar-parity.log" 2>&1

echo "== wait for bonsai-community unit $(date -Is)"
while systemctl --user is-active --quiet bonsai-community.service; do sleep 60; done
echo "== unit finished $(date -Is)"

if [ ! -f "$R/bonsai-community-states.npz" ]; then
    echo "missing $R/bonsai-community-states.npz"; exit 1
fi

echo "== start parity server $(date -Is)"
"$B" -m "$M" -ngl 99 -c 8192 --host 127.0.0.1 --port 8899 \
    --embeddings --pooling last --embd-normalize -1 -fa auto --jinja \
    > /tmp/opencode/sidecar-parity-srv.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8899/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
if [ "$up" != 1 ]; then echo "server did not start"; kill $SRV 2>/dev/null; exit 1; fi

rc=0
for T in gsm8k-0001 math-0001 humaneval-0001; do
    echo "== parity $T sample 0 $(date -Is)"
    "$PY" experiments/cascade/sidecar_parity.py --server http://127.0.0.1:8899 \
        --model "$M" --sidecar "$R/bonsai-2-27b-decision-head.gguf" \
        --head-json "$R/bonsai-2-27b-decision-head.json" \
        --prefix bonsai-community --task "$T" --sample 0 \
        || rc=1
done

echo "== stop server $(date -Is)"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "== done rc=$rc $(date -Is)"
exit $rc
