#!/bin/sh
# L1/L2 correction study: start Scion server, generate 4 samples/task, score
# with the exported head, evaluate selection/retry arms. Resumable.
#   systemd-run --user --unit=correction-run /bin/sh <this>
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
T=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/scion
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
B=/home/penis/llama.cpp/build/bin/llama-server
R=$W/experiments/cascade/results/jev-readout-20261006
export HIP_VISIBLE_DEVICES=1
cd "$W" || exit 1

echo "== start scion server $(date -Is)"
"$B" -m "$T/Scion-35B-A3B-PQ2_0-corr.gguf" -ngl 99 -c 8192 \
    --host 127.0.0.1 --port 8897 --embeddings --pooling last \
    --embd-normalize -1 -fa auto > /tmp/opencode/scion-srv-corr.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
if [ "$up" != 1 ]; then echo "server did not start"; kill $SRV 2>/dev/null; exit 1; fi

echo "== correction eval $(date -Is)"
"$PY" experiments/cascade/correction_eval.py --server http://127.0.0.1:8897 \
    --resume --out "$R/correction-eval.json" 2>&1 | tee "$R/correction-eval.log"

echo "== stop server $(date -Is)"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "== done $(date -Is)"
