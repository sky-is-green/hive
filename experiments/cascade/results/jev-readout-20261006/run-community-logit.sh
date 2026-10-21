#!/bin/sh
# Slot-logit readout on both community pools (Scion, Bonsai): judge prompts via
# a CPU tokenize server, batch readout on GPU, then the community analysis.
#   systemd-run --user --unit=community-logit --collect /bin/sh <this>
set -u
W=<work>/hivebench
PY=<work>/.venv-rocm/bin/python
B=~/llama.cpp/build/bin/llama-server
P=~/llama.cpp/build/bin/test-decision-head
R=$W/experiments/cascade/results/jev-readout-20261006
cd "$W" || exit 1
exec >> "$R/community-logit.log" 2>&1

run_one() {
    PREFIX=$1
    MODEL=$2
    HEAD=$3
    D=$R/logit-$PREFIX

    echo "== [$PREFIX] prompts $(date -Is)"
    export HIP_VISIBLE_DEVICES=""
    "$B" -m "$MODEL" -ngl 0 -c 2048 --host 127.0.0.1 --port 8898 \
        > "/tmp/opencode/$PREFIX-logit-tok.log" 2>&1 &
    SRV=$!
    up=0
    for i in $(seq 1 240); do
        if curl -sf http://127.0.0.1:8898/health >/dev/null 2>&1; then up=1; break; fi
        sleep 2
    done
    if [ "$up" != 1 ]; then
        echo "tokenize server failed for $PREFIX"
        kill $SRV 2>/dev/null
        return 1
    fi
    rm -rf "$D/prompts"
    "$PY" experiments/cascade/make_judge_prompts.py --server http://127.0.0.1:8898 \
        --samples "$R/$PREFIX-samples.jsonl" \
        --tasks experiments/cascade/tasks-community.json \
        --out-dir "$D/prompts" || echo "FAILED prompts $PREFIX"
    kill $SRV 2>/dev/null
    wait $SRV 2>/dev/null
    sleep 2
    if [ ! -d "$D/prompts" ]; then
        echo "no prompts for $PREFIX"
        return 1
    fi

    echo "== [$PREFIX] readout (GPU) $(date -Is)"
    export HIP_VISIBLE_DEVICES=1
    "$P" -m "$MODEL" --prompt-dir "$D/prompts" \
        --probs "A, A,B, B,Yes,No,yes,no" \
        --out-dir "$D/out" > "$D/readout.tsv" 2> "$D/readout.err"

    echo "== [$PREFIX] analyze $(date -Is)"
    "$PY" experiments/cascade/analyze_logit_community.py --prefix "$PREFIX" \
        --dir "$D" --head-json "$HEAD" --out "$R/logit-$PREFIX.json" \
        || echo "FAILED analyze $PREFIX"
    echo "== [$PREFIX] done $(date -Is)"
}

run_one scion-community \
    <work>/scion/Scion-35B-A3B-PQ2_0-corr.gguf \
    "$R/scion-35b-a3b-decision-head.json"

run_one bonsai-community \
    ~/Downloads/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    "$R/bonsai-2-27b-decision-head.json"

echo "== all done $(date -Is)"
