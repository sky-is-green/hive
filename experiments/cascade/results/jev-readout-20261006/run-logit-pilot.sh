#!/bin/sh
# Slot-logit pilot: the model's own p(A)/p(B) at the judge answer slot vs the
# fitted head, on the Scion correction pool.
#   CPU tokenize/template server -> judge prompt files ->
#   batch readout on GPU (hidden states + token probs) -> analysis.
#   systemd-run --user --unit=logit-pilot --collect /bin/sh <this>
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
M=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/scion/Scion-35B-A3B-PQ2_0-corr.gguf
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
B=/home/penis/llama.cpp/build/bin/llama-server
P=/home/penis/llama.cpp/build/bin/test-decision-head
R=$W/experiments/cascade/results/jev-readout-20261006
D=$R/logit-pilot-scion
cd "$W" || exit 1
exec >> "$R/logit-pilot-scion.log" 2>&1

echo "== start tokenize server $(date -Is)"
export HIP_VISIBLE_DEVICES=""
"$B" -m "$M" -ngl 0 -c 2048 --host 127.0.0.1 --port 8898 \
    > /tmp/opencode/scion-logit-tok.log 2>&1 &
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

echo "== build judge prompts $(date -Is)"
rm -rf "$D/prompts"
"$PY" experiments/cascade/make_judge_prompts.py --server http://127.0.0.1:8898 \
    --samples "$R/correction-samples.jsonl" \
    --tasks experiments/cascade/tasks-bench.json \
    --out-dir "$D/prompts" || echo "FAILED prompts"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null

echo "== batch readout (GPU) $(date -Is)"
if [ ! -d "$D/prompts" ] || [ -z "$(ls -A "$D/prompts" 2>/dev/null)" ]; then
    echo "no prompts built"; exit 1
fi
export HIP_VISIBLE_DEVICES=1
"$P" -m "$M" --prompt-dir "$D/prompts" \
    --probs "A, A,B, B,Yes,No,yes,no" \
    --out-dir "$D/out" > "$D/readout.tsv" 2> "$D/readout.err"

echo "== analyze $(date -Is)"
"$PY" experiments/cascade/analyze_logit_pilot.py --dir "$D" \
    --samples "$R/correction-samples.jsonl" \
    --head-json "$R/scion-35b-a3b-decision-head.json" \
    --out "$R/logit-pilot-scion.json" || echo "FAILED analyze"
echo "== done $(date -Is)"
