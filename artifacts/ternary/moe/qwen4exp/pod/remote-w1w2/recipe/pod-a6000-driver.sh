#!/bin/bash
# pod-a6000-driver.sh - upload the probe + patches, run it on the A6000 pod, fetch the measurements.
# The pod is left RUNNING (the full run can resume on it if the probe passes).
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
WORK=/home/penis/Desktop/work
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
JOBDIR=$Q/pod/remote-a6000
RECIPE=$Q/pod/remote-w1w2/recipe
KEY=$HOME/.ssh/id_runpod
POD=$(cat "$JOBDIR/pod-id")
read -r IP PORT < "$JOBDIR/pod-endpoint"
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i $KEY -p $PORT"
SCPOPT="-q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o Compression=no -i $KEY -P $PORT"
SSH="ssh $SSHOPT root@$IP"
SSH_FETCH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -p $PORT"
log() { echo "$(date -u +%FT%TZ) $*"; }

log "=== a6000 driver start pod=$POD ip=$IP port=$PORT"
$SSH 'nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; free -g | head -2' || { log "ssh failed"; exit 1; }
$SSH 'mkdir -p /workspace/bench /root/models' || { log "mkdir failed"; exit 1; }
scp $SCPOPT "$RECIPE/community-probe.sh" root@$IP:/root/community-probe.sh || { log "upload probe failed"; exit 1; }
scp $SCPOPT "$RECIPE/bench-ple-multiseq.patch" root@$IP:/root/bench-ple-multiseq.patch || { log "upload p1 failed"; exit 1; }
scp $SCPOPT "$RECIPE/server-skip-seq-rm.patch" root@$IP:/root/server-skip-seq-rm.patch || { log "upload p2 failed"; exit 1; }
scp $SCPOPT "$RECIPE/embed-read-width.patch" root@$IP:/root/embed-read-width.patch || { log "upload p3 failed"; exit 1; }
$SSH 'setsid nohup bash /root/community-probe.sh > /root/community-probe.out 2>&1 < /dev/null & echo started' || { log "probe start failed"; exit 1; }
log "probe started on the pod"

deadline=$(( $(date +%s) + 7200 ))
while :; do
    done_=$($SSH 'grep -c "PROBE_DONE" /workspace/bench/probe.log 2>/dev/null' 2>/dev/null | tr -cd '0-9')
    fail=$($SSH 'grep -aoE "CLONE_FAILED|PATCH_FAILED|PATCH2_FAILED|PATCH3_FAILED|CMAKE_FAILED|BUILD_FAILED|DL_FAILED|SMOKE_FAILED|SERVER_FAILED" /workspace/bench/probe.log 2>/dev/null | head -1' 2>/dev/null)
    if [ "${done_:-0}" -ge 1 ]; then log "PROBE_DONE seen"; break; fi
    if [ -n "$fail" ]; then log "REMOTE FAILURE: $fail"; break; fi
    if [ "$(date +%s)" -ge "$deadline" ]; then log "TIMEOUT"; break; fi
    tail1=$($SSH 'tail -1 /workspace/bench/probe.log 2>/dev/null' 2>/dev/null)
    [ -n "$tail1" ] && log "[remote] $tail1"
    sleep 60
done

mkdir -p "$JOBDIR/bench"
for f in probe.log server.log smoke.log build.log dl.log; do
    rsync -a --partial --timeout=300 -e "$SSH_FETCH" "root@$IP:/workspace/bench/$f" "$JOBDIR/bench/" >/dev/null 2>&1 || true
done
log "--- measurements:"
grep -aE "single:|conc4:|embedding:|memory.used|server up|build OK|dl OK|smoke OK" "$JOBDIR/bench/probe.log" 2>/dev/null | tail -12
log "=== a6000 driver DONE (pod left running) ==="
