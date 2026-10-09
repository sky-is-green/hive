#!/bin/bash
# pod-bench-run.sh - local driver for the A100 bench window:
#   upload bench-run.sh -> start it detached on the pod -> poll -> fetch the
#   bench logs -> terminate.  Guard: w4bench-spend cap 7.5.
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
WORK=/home/penis/Desktop/work
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
JOBDIR=$Q/pod/remote-bench
RECIPE=$Q/pod/remote-w1w2/recipe
WS=$WORK/scion/moe/pod_watch
KEY=$HOME/.ssh/id_runpod
mkdir -p "$JOBDIR"

POD=$(cat "$JOBDIR/pod-id")
read -r IP PORT < "$JOBDIR/pod-endpoint"
RATE=$(cat "$JOBDIR/pod-rate")
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i $KEY -p $PORT"
SCPOPT="-q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o Compression=no -i $KEY -P $PORT"
SSH="ssh $SSHOPT root@$IP"
SSH_FETCH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -p $PORT"

log() { echo "$(date -u +%FT%TZ) $*"; }
ssh_pod() { $SSH "$@"; }

terminate() {
    systemctl --user stop w4bench-spend >/dev/null 2>&1 || true
    curl -s --max-time 60 -X DELETE https://rest.runpod.io/v1/pods/$POD \
        -H "Authorization: Bearer $(cat "$HOME/.runpod-api-key")" >/dev/null 2>&1 || true
    sleep 5
    if curl -s --max-time 30 https://rest.runpod.io/v1/pods \
        -H "Authorization: Bearer $(cat "$HOME/.runpod-api-key")" | grep -q "\"$POD\""; then
        log "WARN: pod $POD still listed"
    else
        log "pod $POD terminated"
    fi
}
fetch_all() {
    mkdir -p "$JOBDIR/bench"
    rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
        "root@$IP:/workspace/bench/" "$JOBDIR/bench/" >/dev/null 2>&1 || log "WARN: bench rsync failed"
    rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
        "root@$IP:/workspace/llama-qwen4exp/build.log" "$JOBDIR/bench/" >/dev/null 2>&1 || true
    rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
        "root@$IP:/workspace/llama-qwen4exp/smoke.log" "$JOBDIR/bench/" >/dev/null 2>&1 || true
    return 0
}
die() { log "FATAL: $*"; fetch_all; terminate; log "=== bench driver FAILED ==="; exit 1; }

log "=== bench driver start pod=$POD ip=$IP port=$PORT rate=$RATE"

ssh_pod 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; df -h /workspace / | tail -2' || die "ssh probe failed"
ssh_pod 'mkdir -p /workspace/bench /workspace/models' || die "mkdir failed"
scp $SCPOPT "$RECIPE/bench-run.sh" root@$IP:/root/bench-run.sh || die "bench-run upload failed"
scp $SCPOPT "$RECIPE/bench-ple-multiseq.patch" root@$IP:/root/bench-ple-multiseq.patch || die "patch upload failed"

# guards: spend cap 7.5, no puller (logs fetched at stop)
ts=$(date -u +%Y%m%dT%H%M%SZ)
# stop any guard left from a previous window BEFORE rotating its state file,
# so a zombie cycle cannot recreate the old start_balance after the move
systemctl --user stop w4bench-spend >/dev/null 2>&1 || true
systemctl --user reset-failed w4bench-spend >/dev/null 2>&1 || true
[ -e "$JOBDIR/.spend-guard-state" ] && mv "$JOBDIR/.spend-guard-state" "$JOBDIR/.spend-guard-state.prev-$ts"
[ -e "$JOBDIR/pod-watch.log" ] && mv "$JOBDIR/pod-watch.log" "$JOBDIR/pod-watch.log.prev-$ts"
systemd-run --user --unit=w4bench-spend --collect \
    --property=StandardOutput=append:"$JOBDIR/w4bench-spend.log" \
    --property=StandardError=append:"$JOBDIR/w4bench-spend.log" \
    --setenv=PW_POD="$POD" --setenv=PW_SSH_HOST="root@$IP" --setenv=PW_SSH_PORT="$PORT" \
    --setenv=PW_DST="$JOBDIR" --setenv=PW_WINDOW_CAP=7.5 --setenv=PW_RATE_PER_HR="$RATE" \
    --setenv=PW_FETCH_MIN=15 --setenv=PW_MARGIN_MIN=15 --setenv=PW_RESERVE_FLOOR=0.5 \
    --setenv=PW_REMOTE_ART=/workspace/bench \
    bash "$WS/spend_guard.sh" || die "spend guard start failed"
log "spend guard started (cap 7.5, rate $RATE)"

ssh_pod 'setsid nohup bash /root/bench-run.sh >/root/bench-run.out 2>&1 < /dev/null & echo started' || die "bench start failed"
log "bench payload started on the pod"

# poll: BENCH_DONE or a failure marker, up to 5 h
deadline=$(( $(date +%s) + 18000 ))
while :; do
    done_=$(ssh_pod 'grep -c "BENCH_DONE" /workspace/bench/run.log 2>/dev/null' 2>/dev/null | tr -cd '0-9')
    fail=$(ssh_pod 'grep -aE "CLONE_FAILED|PATCH_FAILED|CMAKE_FAILED|BUILD_FAILED|DL_OURS_FAILED|DL_ISTA_FAILED|DL_WIKI_FAILED|DL_HS_FAILED|DL_WG_FAILED|UNZIP_FAILED|MISSING_INPUTS|SHA_OURS_FAILED|SMOKE_FAILED|TINY_PPL_FAILED" /workspace/bench/run.log 2>/dev/null | tail -1' 2>/dev/null)
    if [ "${done_:-0}" -ge 1 ]; then
        log "BENCH_DONE seen"; break
    fi
    if [ -n "$fail" ]; then
        log "REMOTE FAILURE: $fail"; break
    fi
    if grep -q "STOPPING:" "$JOBDIR/pod-watch.log" 2>/dev/null; then
        log "spend guard stopped the pod; exiting poll loop"; break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        log "TIMEOUT waiting for bench"; break
    fi
    tail1=$(ssh_pod 'tail -1 /workspace/bench/run.log 2>/dev/null' 2>/dev/null)
    [ -n "$tail1" ] && log "[remote] $tail1"
    sleep 60
done

log "fetching artifacts"
fetch_all
python3 - "$JOBDIR/bench" <<'EOF' || true
import re, sys
from pathlib import Path
b = Path(sys.argv[1])
res = {}
for f in sorted(b.glob("*.log")):
    t = f.read_text(errors="ignore")
    m = re.findall(r"Final estimate: PPL =\s*([\d.]+)\s*\+/-\s*([\d.]+)", t)
    if m: res[f.name] = ("ppl", m[-1])
    m = re.findall(r"^\s*400\s+([\d.]+)%", t, re.M)
    if m: res[f.name] = ("hellaswag", m[-1])
    m = re.findall(r"Final Winogrande score\(400 tasks\):\s*([\d.]+)\s*\+/-\s*([\d.]+)", t)
    if m: res[f.name] = ("winogrande", m[-1])
print("BENCH RESULTS:")
for k, v in res.items():
    print("  %-28s %-10s %s" % (k, v[0], v[1]))
EOF
terminate
log "=== bench driver DONE ==="
