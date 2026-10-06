#!/bin/sh
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
R=$W/experiments/cascade/results/jev-readout-20261006
export HIP_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
cd "$W" || exit 1
for spec in "intern internlm/Intern-Decision-4B intern-4b cuda:0" \
            "intern internlm/Intern-Decision-2B intern-2b cuda:0" \
            "intern internlm/Intern-Decision-0.8B intern-0.8b cuda:0" \
            "tinyjev lostargon/Tiny-Jev-1.7B tinyjev-1.7b cpu"; do
    set -- $spec
    echo "== $3 ($(date -Is))"
    "$PY" experiments/cascade/score_judges.py --family "$1" --model "$2" \
        --name "$3" --device "$4" --split all > "$R/judge-$3.log" 2>&1 \
        || echo "FAILED $3"
done
echo "== done $(date -Is)"
