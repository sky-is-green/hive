#!/bin/sh
# T3: ternary survival — final-layer hidden states for the Clef bodies via the
# fork server embeddings endpoint.  Bodies: QAT (23.25 PPL), h128 GPTQ (120.6),
# f16 reference.  Run detached:
#   systemd-run --user --unit=clef-embed /bin/sh <this>
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
T=/home/penis/Desktop/work/models/clef-flash-ternary
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
B=/home/penis/llama.cpp/build/bin/llama-server
R=$W/experiments/cascade/results/jev-readout-20261006
export HIP_VISIBLE_DEVICES=1
cd "$W" || exit 1

serve_probe() { # <model> <name> [ctx]
    model=$1; name=$2; ctx=${3:-4096}
    echo "== $name $(date -Is)"
    "$B" -m "$model" -ngl 99 -c "$ctx" --host 127.0.0.1 --port 8897 \
        --embeddings --pooling last --embd-normalize -1 -fa auto \
        > "/tmp/opencode/clef-$name-srv.log" 2>&1 &
    SRV=$!
    up=0
    for i in $(seq 1 90); do
        if curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1; then up=1; break; fi
        sleep 2
    done
    if [ "$up" != 1 ]; then
        echo "  server $name did not start"; kill $SRV 2>/dev/null
        return 1
    fi
    "$PY" experiments/cascade/probe_scion.py --server http://127.0.0.1:8897 \
        --name "$name" > "$R/probe-$name.log" 2>&1 || echo "  FAILED probe $name"
    kill $SRV 2>/dev/null
    wait $SRV 2>/dev/null
}

serve_probe "$T/v2/clef-flash-v2-qat-a1.gguf" clef-qat
serve_probe "$T/v2/clef-flash-v2-signs-gptq-lloyd-ao-h128.gguf" clef-h128
serve_probe "$T/clef-flash-f16.gguf" clef-f16 2048
echo "== done $(date -Is)"
