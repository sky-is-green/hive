#!/bin/sh
# Hinted retry + re-read variants: start Scion server, generate hinted
# retries (each task shown its flagged first answer), score, then embed the
# reasoning-tail re-read variant.  Resumable.
#   systemd-run --user --unit=hinted-run /bin/sh <this>
set -u
W=<work>/hivebench
T=<work>/scion
PY=<work>/.venv-rocm/bin/python
B=~/llama.cpp/build/bin/llama-server
R=$W/experiments/cascade/results/jev-readout-20261006
export HIP_VISIBLE_DEVICES=1
cd "$W" || exit 1

echo "== start scion server $(date -Is)"
"$B" -m "$T/Scion-35B-A3B-PQ2_0-corr.gguf" -ngl 99 -c 8192 \
    --host 127.0.0.1 --port 8897 --embeddings --pooling last \
    --embd-normalize -1 -fa auto > /tmp/opencode/scion-srv-hint.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
if [ "$up" != 1 ]; then echo "server did not start"; kill $SRV 2>/dev/null; exit 1; fi

echo "== hinted retry $(date -Is)"
"$PY" experiments/cascade/hinted_retry.py --server http://127.0.0.1:8897 \
    2>&1 | tee "$R/hinted-retry.log"

echo "== reread variants $(date -Is)"
"$PY" experiments/cascade/reread_variants.py --server http://127.0.0.1:8897 \
    2>&1 | tee "$R/reread-variant.log"

echo "== stop server $(date -Is)"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "== done $(date -Is)"
