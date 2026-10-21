#!/bin/sh
# Bonsai dense test: serve Ternary-Bonsai-2-27B (Q2_0) with the moe-corr fork
# and run the Scion protocol on cascade-bench-v1 -- J-Bench probe, 4 samples
# per task at the card's thinking sampling, hinted retries for wrong-first
# tasks, judge-prompt embeddings in two candidate formats, head fit/eval, and
# the prompted baselines.  Resumable; JSONL checkpoints per phase.
#   systemd-run --user --unit=bonsai-run /bin/sh <this>
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
    > /tmp/opencode/bonsai-srv.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
if [ "$up" != 1 ]; then echo "server did not start"; kill $SRV 2>/dev/null; exit 1; fi

echo "== J-Bench probe $(date -Is)"
"$PY" experiments/cascade/probe_scion.py --server http://127.0.0.1:8897 \
    --name bonsai27b > "$R/probe-bonsai27b.log" 2>&1 || echo "FAILED probe"

echo "== samples (4/task, thinking) $(date -Is)"
"$PY" experiments/cascade/bonsai_eval.py --server http://127.0.0.1:8897 \
    --phase gen --prefix bonsai > "$R/bonsai-gen.log" 2>&1 || echo "FAILED gen"

echo "== hinted retries $(date -Is)"
"$PY" experiments/cascade/bonsai_eval.py --server http://127.0.0.1:8897 \
    --phase hinted --prefix bonsai > "$R/bonsai-hinted.log" 2>&1 || echo "FAILED hinted"

echo "== embeddings (answer + trace) $(date -Is)"
"$PY" experiments/cascade/bonsai_eval.py --server http://127.0.0.1:8897 \
    --phase embed --prefix bonsai > "$R/bonsai-embed.log" 2>&1 || echo "FAILED embed"

echo "== eval $(date -Is)"
"$PY" experiments/cascade/bonsai_eval.py --phase eval --prefix bonsai \
    > "$R/bonsai-eval.log" 2>&1 || echo "FAILED eval"

echo "== export J-Bench head $(date -Is)"
"$PY" experiments/cascade/export_head.py \
    --states "$R/probe-states-bonsai27b.npz" \
    --model prism-ml/Ternary-Bonsai-2-27B-gguf \
    --name bonsai-2-27b-decision-head \
    --runtime "moe-corr fork (llama.cpp 0.2.0-dev b10721), llama-server --embeddings --pooling last --embd-normalize -1" \
    --out "$R/bonsai-2-27b-decision-head.json" \
    > "$R/export-bonsai-head.log" 2>&1 || echo "FAILED export"

echo "== own items $(date -Is)"
"$PY" experiments/cascade/bonsai_eval.py --phase items --prefix bonsai \
    > "$R/bonsai-items.log" 2>&1 || echo "FAILED items"

echo "== prompted J-Bench $(date -Is)"
"$PY" experiments/cascade/readout_judge.py --server-chat http://127.0.0.1:8897 \
    --name bonsai27b-prompted --split all \
    > "$R/readout-bonsai27b-prompted.log" 2>&1 || echo "FAILED prompted bench"

echo "== prompted own answers $(date -Is)"
"$PY" experiments/cascade/readout_judge.py --server-chat http://127.0.0.1:8897 \
    --bench "$R/bonsai-own-items.jsonl" --name bonsai-own-prompted --split all \
    > "$R/readout-bonsai-own-prompted.log" 2>&1 || echo "FAILED prompted own"

echo "== stop server $(date -Is)"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "== done $(date -Is)"
