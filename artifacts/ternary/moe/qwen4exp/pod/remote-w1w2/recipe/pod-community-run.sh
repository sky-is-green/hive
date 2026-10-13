#!/bin/bash
# pod-community-run.sh - local driver for the community-protocol window:
#   upload community-run.sh + community files -> start it detached on the pod
#   -> poll -> fetch artifacts -> terminate.  Guard: scion-community-spend.
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
WORK=/home/penis/Desktop/work
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
JOBDIR=$Q/pod/remote-community
RECIPE=$Q/pod/remote-w1w2/recipe
KEY=$HOME/.ssh/id_runpod
CAP=${CAP:-6.0}          # spend cap for this window (USD)
FLOOR=${FLOOR:-5.0}      # balance floor the guard deletes the pod at
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
    systemctl --user stop scion-community-spend >/dev/null 2>&1 || true
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
    rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
        "root@$IP:/workspace/bench/server.log" "$JOBDIR/bench/" >/dev/null 2>&1 || true
    return 0
}
die() { log "FATAL: $*"; fetch_all; terminate; log "=== community driver FAILED ==="; exit 1; }

log "=== community driver start pod=$POD ip=$IP port=$PORT rate=$RATE cap=$CAP floor=$FLOOR"

ssh_pod 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; df -h /workspace / | tail -2' || die "ssh probe failed"
ssh_pod 'mkdir -p /workspace/bench /workspace/models /root/community' || die "mkdir failed"
scp $SCPOPT "$RECIPE/community-run.sh" root@$IP:/root/community-run.sh || die "payload upload failed"
scp $SCPOPT "$RECIPE/bench-ple-multiseq.patch" root@$IP:/root/bench-ple-multiseq.patch || die "patch upload failed"
scp $SCPOPT "$RECIPE/server-skip-seq-rm.patch" root@$IP:/root/server-skip-seq-rm.patch || die "seq-rm patch upload failed"
scp $SCPOPT "$RECIPE/embed-read-width.patch" root@$IP:/root/embed-read-width.patch || die "embed patch upload failed"
scp $SCPOPT "$RECIPE/router-stats.patch" root@$IP:/root/router-stats.patch || die "router-stats patch upload failed"
scp $SCPOPT "$WORK/hivebench/experiments/cascade/community_eval.py" root@$IP:/root/community/community_eval.py \
    || die "community_eval upload failed"
scp $SCPOPT "$WORK/hivebench/experiments/cascade/tasks-community.json" root@$IP:/root/community/tasks-community.json \
    || die "tasks upload failed"
# community_eval.py imports these three (same directory): without them every phase
# dies with ModuleNotFoundError right after the server starts (first window, 15:36).
for m in checkers probe_judge readout_judge; do
    scp $SCPOPT "$WORK/hivebench/experiments/cascade/$m.py" root@$IP:/root/community/$m.py \
        || die "$m upload failed"
done

# guard: balance floor + elapsed cap (the old pod_watch guard is gone)
systemctl --user stop scion-community-spend >/dev/null 2>&1 || true
systemctl --user reset-failed scion-community-spend >/dev/null 2>&1 || true
systemd-run --user --unit=scion-community-spend --collect \
    --property=StandardOutput=append:"$JOBDIR/community-spend.log" \
    --property=StandardError=append:"$JOBDIR/community-spend.log" \
    bash "$RECIPE/community-guard.sh" "$POD" "$FLOOR" 420 || die "guard start failed"
log "guard started (floor $FLOOR, 420 min cap, window cap \$$CAP)"

ssh_pod 'setsid nohup bash /root/community-run.sh >/root/community-run.out 2>&1 < /dev/null & echo started' || die "payload start failed"
log "payload started on the pod"

deadline=$(( $(date +%s) + 21600 ))
while :; do
    done_=$(ssh_pod 'grep -c "COMMUNITY_DONE" /workspace/bench/community-run.log 2>/dev/null' 2>/dev/null | tr -cd '0-9')
    fail=$(ssh_pod 'grep -aoE "CLONE_FAILED|PATCH_FAILED|PATCH2_FAILED|PATCH3_FAILED|PATCH4_FAILED|CMAKE_FAILED|BUILD_FAILED|DL_OURS_FAILED|SHA_FAILED|SMOKE_FAILED|SERVER_FAILED" /workspace/bench/community-run.log 2>/dev/null | head -1' 2>/dev/null)
    if [ "${done_:-0}" -ge 1 ]; then log "COMMUNITY_DONE seen"; break; fi
    if [ -n "$fail" ]; then log "REMOTE FAILURE: $fail"; break; fi
    if grep -q "deleting pod" "$JOBDIR/community-spend.log" 2>/dev/null; then
        log "guard deleted the pod; exiting poll loop"; break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then log "TIMEOUT waiting for the run"; break; fi
    tail1=$(ssh_pod 'tail -1 /workspace/bench/community-run.log 2>/dev/null' 2>/dev/null)
    [ -n "$tail1" ] && log "[remote] $tail1"
    sleep 60
done

log "fetching artifacts"
fetch_all
python3 - "$JOBDIR/bench/community" <<'EOF' || true
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
f = r / "scion-flashnext-eval.json"
if f.exists():
    d = json.loads(f.read_text())
    print("COMMUNITY EVAL:")
    print(json.dumps(d, indent=1)[:2500])
else:
    print("no eval json fetched yet; files:", sorted(p.name for p in r.glob('*')))
EOF
terminate
log "=== community driver DONE ==="
