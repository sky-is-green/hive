#!/bin/sh
# L3 monitor correction: re-run all probes after fixing the win_%02d
# string-sort misalignment, plus the non-linear / trajectory retest.
#   systemd-run --user --unit=monitor-fix /bin/sh <this>
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
M=$W/experiments/cascade/streams/scion-bench-monitor
R=$W/experiments/cascade/results/jev-readout-20261006
cd "$W" || exit 1

# preserve the invalid pre-fix artifacts for the record
for f in monitor-probe.json monitor-diag.json monitor-token-diag.json; do
    if [ -f "$R/$f" ] && [ ! -f "$R/${f%.json}-misaligned.json" ]; then
        cp "$R/$f" "$R/${f%.json}-misaligned.json"
    fi
done

echo "== monitor_probe (numeric order) $(date -Is)"
"$PY" experiments/cascade/monitor_probe.py --windows "$M" \
    > "$R/monitor-probe.log" 2>&1 || echo "FAILED probe"
echo "== monitor_diag (numeric order) $(date -Is)"
"$PY" experiments/cascade/monitor_diag.py --windows "$M" \
    > "$R/monitor-diag.log" 2>&1 || echo "FAILED diag"
echo "== monitor_token_diag (numeric order) $(date -Is)"
"$PY" experiments/cascade/monitor_token_diag.py --windows "$M" \
    > "$R/monitor-token-diag.log" 2>&1 || echo "FAILED token diag"
echo "== non-linear / trajectory retest $(date -Is)"
"$PY" experiments/cascade/monitor_nonlinear.py --windows "$M" \
    > "$R/monitor-nonlinear.log" 2>&1 || echo "FAILED nonlinear"
echo "== done $(date -Is)"
