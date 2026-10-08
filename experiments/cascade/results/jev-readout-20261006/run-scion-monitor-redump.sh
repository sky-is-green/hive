#!/bin/sh
# Re-dump the Scion monitor states with the embedded correction adapter applied.
# test-mtp-probe previously ran the uncorrected quant, so both the read-back
# (scion-bench-monitor) and generation-context (scion-bench-monitor-ctx) dumps
# were off the served model.  Old artifacts are preserved with -noadapter.
#   systemd-run --user --unit=monitor-redump --collect /bin/sh <this>
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
M=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/scion/Scion-35B-A3B-PQ2_0-corr.gguf
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
P=/home/penis/llama.cpp/build/bin/test-mtp-probe
R=$W/experiments/cascade/results/jev-readout-20261006
RB=$W/experiments/cascade/streams/scion-bench-monitor
CTX=$W/experiments/cascade/streams/scion-bench-monitor-ctx
cd "$W" || exit 1
exec >> "$R/monitor-redump.log" 2>&1

export HIP_VISIBLE_DEVICES=1

echo "== read-back re-dump $(date -Is)"
if [ -d "$RB/dump" ] && [ ! -d "$RB/dump-noadapter" ]; then
    mv "$RB/dump" "$RB/dump-noadapter"
fi
for f in monitor-probe monitor-diag monitor-token-diag monitor-nonlinear; do
    if [ -f "$R/$f.json" ] && [ ! -f "$R/$f-noadapter.json" ]; then
        cp "$R/$f.json" "$R/$f-noadapter.json"
    fi
done
N=$("$PY" -c "import json;print(len(json.load(open('$RB/meta.json'))['order']))")
SEQ=$("$PY" -c "import json;print(json.load(open('$RB/meta.json'))['seq'])")
mkdir -p "$RB/dump"
"$P" "$M" "$RB/tokens.bin" "$N" "$SEQ" "$RB/dump" 999 || echo "FAILED rb dump"

echo "== read-back probes $(date -Is)"
"$PY" experiments/cascade/monitor_probe.py --windows "$RB" \
    > "$R/monitor-probe.log" 2>&1 || echo "FAILED probe"
"$PY" experiments/cascade/monitor_diag.py --windows "$RB" \
    > "$R/monitor-diag.log" 2>&1 || echo "FAILED diag"
"$PY" experiments/cascade/monitor_token_diag.py --windows "$RB" \
    > "$R/monitor-token-diag.log" 2>&1 || echo "FAILED token diag"
"$PY" experiments/cascade/monitor_nonlinear.py --windows "$RB" \
    > "$R/monitor-nonlinear.log" 2>&1 || echo "FAILED nonlinear"

echo "== context re-dump $(date -Is)"
if [ -d "$CTX/dump" ] && [ ! -d "$CTX/dump-noadapter" ]; then
    mv "$CTX/dump" "$CTX/dump-noadapter"
fi
if [ -f "$R/monitor-context-scion.json" ] && [ ! -f "$R/monitor-context-scion-noadapter.json" ]; then
    cp "$R/monitor-context-scion.json" "$R/monitor-context-scion-noadapter.json"
fi
N2=$("$PY" -c "import json;print(len(json.load(open('$CTX/meta.json'))['order']))")
SEQ2=$("$PY" -c "import json;print(json.load(open('$CTX/meta.json'))['seq'])")
mkdir -p "$CTX/dump"
"$P" "$M" "$CTX/tokens.bin" "$N2" "$SEQ2" "$CTX/dump" 999 || echo "FAILED ctx dump"

echo "== context eval $(date -Is)"
"$PY" experiments/cascade/monitor_context_eval.py --windows "$CTX" \
    --out "$R/monitor-context-scion.json" || echo "FAILED ctx eval"
echo "== done $(date -Is)"
