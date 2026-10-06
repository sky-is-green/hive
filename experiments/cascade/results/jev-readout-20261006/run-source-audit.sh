#!/bin/sh
# Post-reboot source-swap audit: start the Scion server, embed own vs other
# answers with the frozen head, evaluate natural errors, stop the server.
#   systemd-run --user --unit=source-audit /bin/sh <this>
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
T=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/scion
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
B=/home/penis/llama.cpp/build/bin/llama-server
R=$W/experiments/cascade/results/jev-readout-20261006
export HIP_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
cd "$W" || exit 1

echo "== start scion server $(date -Is)"
"$B" -m "$T/Scion-35B-A3B-PQ2_0-corr.gguf" -ngl 99 -c 8192 \
    --host 127.0.0.1 --port 8897 --embeddings --pooling last \
    --embd-normalize -1 -fa auto > /tmp/opencode/scion-srv-audit.log 2>&1 &
SRV=$!
for i in $(seq 1 90); do
    curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1 && break
    sleep 2
done
curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1 || { echo "server did not start"; kill $SRV 2>/dev/null; exit 1; }

echo "== source audit $(date -Is)"
"$PY" experiments/cascade/source_audit.py --server http://127.0.0.1:8897 \
    2>&1 | tee "$R/source-audit.log"

echo "== stop server $(date -Is)"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "== done $(date -Is)"
