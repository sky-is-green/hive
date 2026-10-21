#!/bin/sh
# T2 sweep: Qwen3-1.7B probe + label-budget curves on 0.8B / 4B.
# Run detached: systemd-run --user --unit=jev-sweep /bin/sh <this>
set -u
W=<work>/hivebench
PY=<work>/.venv-rocm/bin/python
R=$W/experiments/cascade/results/jev-readout-20261006
export HIP_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
cd "$W" || exit 1

echo "== Qwen3-1.7B probe $(date -Is)"
"$PY" experiments/cascade/probe_judge.py --hf Qwen/Qwen3-1.7B \
    --name qwen3-1.7b --device cuda:0 --layers 0.5,0.75,1.0 \
    > "$R/probe-qwen3-1.7b.log" 2>&1 || echo "FAILED 1.7B"

for spec in "Qwen/Qwen3.5-0.8B qwen35-0.8b" "Qwen/Qwen3.5-4B qwen35-4b"; do
    set -- $spec
    for n in 20 100 300; do
        echo "== $2 labels $n $(date -Is)"
        "$PY" experiments/cascade/probe_judge.py --hf "$1" --name "$2-l$n" \
            --device cuda:0 --layers 0.75 --labels "$n" \
            > "$R/probe-$2-l$n.log" 2>&1 || echo "FAILED $2-l$n"
    done
done
echo "== done $(date -Is)"
