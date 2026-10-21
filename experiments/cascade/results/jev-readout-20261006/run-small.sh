#!/bin/sh
# JEV-readout small-model chain: probe + prompted readout on the local
# HF bases (non-display GPU).  T2 size signal; T1c independent-model leg.
# Run detached: systemd-run --user --unit=jev-small /bin/sh <this>
set -u
W=<work>/hivebench
PY=<work>/.venv-rocm/bin/python
R=$W/experiments/cascade/results/jev-readout-20261006
export HIP_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
cd "$W" || exit 1

echo "== 0.8B probe $(date -Is)"
"$PY" experiments/cascade/probe_judge.py --hf Qwen/Qwen3.5-0.8B \
    --name qwen35-0.8b --device cuda:0 --layers 0.5,0.75,1.0 \
    > "$R/probe-qwen35-0.8b.log" 2>&1 || echo "FAILED 0.8B probe"

echo "== 4B probe $(date -Is)"
"$PY" experiments/cascade/probe_judge.py --hf Qwen/Qwen3.5-4B \
    --name qwen35-4b --device cuda:0 --layers 0.5,0.75,1.0 \
    > "$R/probe-qwen35-4b.log" 2>&1 || echo "FAILED 4B probe"

echo "== 4B prompted readout $(date -Is)"
"$PY" experiments/cascade/readout_judge.py --hf Qwen/Qwen3.5-4B \
    --name qwen35-4b --device cuda:0 --split test \
    > "$R/readout-qwen35-4b.log" 2>&1 || echo "FAILED 4B readout"

echo "== done $(date -Is)"
