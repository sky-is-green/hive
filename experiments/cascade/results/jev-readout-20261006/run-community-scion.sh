#!/bin/sh
# Scion community-benchmark arms (GSM8K-300 / MATH-200 / HumanEval-164):
# serve Scion, generate 3 samples/task at the serving sampling (non-thinking)
# plus one think-harder sample/task (thinking, 2048 cap), embed judge prompts,
# fit the in-domain head and evaluate arms A/B/C/D on the held-out test tasks.
#   systemd-run --user --unit=scion-community /bin/sh <this>
set -u
W=<work>/hivebench
T=<work>/scion
PY=<work>/.venv-rocm/bin/python
B=~/llama.cpp/build/bin/llama-server
M=$T/Scion-35B-A3B-PQ2_0-corr.gguf
R=$W/experiments/cascade/results/jev-readout-20261006
export HIP_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
cd "$W" || exit 1

echo "== start scion server $(date -Is)"
"$B" -m "$M" -ngl 99 -c 8192 --host 127.0.0.1 --port 8897 \
    --embeddings --pooling last --embd-normalize -1 -fa auto --jinja \
    > /tmp/opencode/scion-community-srv.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1; then up=1; break; fi
    sleep 2
done
if [ "$up" != 1 ]; then echo "server did not start"; kill $SRV 2>/dev/null; exit 1; fi

echo "== gen 3 samples/task $(date -Is)"
"$PY" experiments/cascade/community_eval.py --server http://127.0.0.1:8897 \
    --prefix scion-community --model-label SkyIsNotGreen/Scion-35B-A3B \
    --model-name scion --runtime "moe-corr fork (llama.cpp 0.2.0-dev b10721)" \
    --tasks experiments/cascade/tasks-community.json --phase gen \
    > "$R/scion-community-gen.log" 2>&1 || echo "FAILED gen"

echo "== think-harder arm $(date -Is)"
"$PY" experiments/cascade/community_eval.py --server http://127.0.0.1:8897 \
    --prefix scion-community --phase hard \
    > "$R/scion-community-hard.log" 2>&1 || echo "FAILED hard"

echo "== embeddings $(date -Is)"
"$PY" experiments/cascade/community_eval.py --server http://127.0.0.1:8897 \
    --prefix scion-community --phase embed \
    > "$R/scion-community-embed.log" 2>&1 || echo "FAILED embed"

echo "== eval $(date -Is)"
"$PY" experiments/cascade/community_eval.py --prefix scion-community \
    --phase eval > "$R/scion-community-eval.log" 2>&1 || echo "FAILED eval"

echo "== stop server $(date -Is)"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "== done $(date -Is)"
