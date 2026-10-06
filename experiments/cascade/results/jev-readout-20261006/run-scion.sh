#!/bin/sh
# Scion-35B generator session for J-Bench: prompted self-judge, final-layer
# probe, and the generator's own answers (source set + natural errors).
# Assumes the fork server is up on :8897 with --embeddings --pooling last
# --embd-normalize -1.  Run detached:
#   systemd-run --user --unit=scion-session /bin/sh <this>
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
R=$W/experiments/cascade/results/jev-readout-20261006
export HF_HUB_OFFLINE=1
cd "$W" || exit 1

echo "== scion prompted readout $(date -Is)"
"$PY" experiments/cascade/readout_judge.py --server-chat http://127.0.0.1:8897 \
    --name scion35b-prompted --split all \
    > "$R/readout-scion35b-prompted.log" 2>&1 || echo "FAILED readout"

echo "== scion final-layer probe $(date -Is)"
"$PY" experiments/cascade/probe_scion.py --server http://127.0.0.1:8897 \
    --name scion35b > "$R/probe-scion35b.log" 2>&1 || echo "FAILED probe"

echo "== scion answers $(date -Is)"
"$PY" experiments/cascade/gen_answers.py --server http://127.0.0.1:8897 \
    --out "$R/scion-answers.json" > "$R/gen-answers.log" 2>&1 \
    || echo "FAILED answers"

echo "== comparison $(date -Is)"
"$PY" experiments/cascade/compare_readouts.py

echo "== done $(date -Is)"
