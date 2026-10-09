#!/bin/sh
# Scion-FlashNext-176B-A6B community-benchmark arms via the STRATA engine.
# The fork's local llama-server is unusable on this box (HIP wedge at full
# model), and Strata has no embeddings route, so: gen (3 samples/task,
# non-thinking, 700 cap) + think-harder (thinking, 2048 cap). The judge-state
# embed/head arms are deferred (need a llama-server /embeddings endpoint).
# Engine: Strata 0.1.37, converted build (Q2_0 experts, Q5_0 table),
# corrections-off — numbers are a lower bound on the release.
#   systemd-run --user --unit=scion-flashnext-community /bin/sh <this>
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
PY=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/.venv-rocm/bin/python
R=$W/experiments/cascade/results/scion-flashnext-community-20261009
export HF_HUB_OFFLINE=1
mkdir -p "$R"
cd "$W" || exit 1

echo "== health $(date -Is)"
curl -sf --max-time 10 http://127.0.0.1:8081/health || { echo "engine not up"; exit 1; }

echo "== gen 3 samples/task $(date -Is)"
"$PY" experiments/cascade/community_eval.py --server http://127.0.0.1:8081 \
    --prefix scion-flashnext --model-label SkyIsNotGreen/Scion-FlashNext-176B-A6B \
    --model-name scion-flashnext --runtime "strata 0.1.37 (converted Q2_0, corrections-off)" \
    --tasks experiments/cascade/tasks-community.json --phase gen \
    --out-dir "$R" \
    > "$R/scion-flashnext-gen.log" 2>&1 || echo "FAILED gen"

echo "== think-harder arm $(date -Is)"
"$PY" experiments/cascade/community_eval.py --server http://127.0.0.1:8081 \
    --prefix scion-flashnext --phase hard --out-dir "$R" \
    > "$R/scion-flashnext-hard.log" 2>&1 || echo "FAILED hard"

echo "== done $(date -Is)"
