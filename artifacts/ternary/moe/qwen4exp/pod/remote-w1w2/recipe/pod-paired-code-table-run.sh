#!/bin/bash
# pod-paired-code-table-run.sh - local driver for the same-pod paired code
# window for the Q4X160 table file vs the release:
#   upload (payload, builder, exporter, codec, patches 1-6, community
#   harness, PREBUILT table bytes in 5G chunks) -> start
#   paired-code-table.sh detached -> poll -> fetch -> terminate.
#   No pod-side transcode, so host speed does not matter.
# Guard: scion-pct-spend (floor $0.5, 240 min cap).
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
WORK=<work>
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
CASCADE=$WORK/hivebench/experiments/cascade
JOBDIR=${JOBDIR:-$Q/pod/remote-paired-table}
RECIPE=$Q/pod/remote-w1w2/recipe
KEY=$HOME/.ssh/id_runpod
FLOOR=${FLOOR:-0.5}
CAP_MIN=${CAP_MIN:-240}
TBL=${TBL:-$WORK/tmp-export/q4x160-table.bin}
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
    systemctl --user stop scion-pct-spend >/dev/null 2>&1 || true
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
    mkdir -p "$JOBDIR/bench/paired/release" "$JOBDIR/bench/paired/table" \
             "$JOBDIR/bench/paired/release-cap4096" "$JOBDIR/bench/paired/table-cap4096"
    rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
        "root@$IP:/workspace/bench/paired/release/" "$JOBDIR/bench/paired/release/" >/dev/null 2>&1 || true
    rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
        "root@$IP:/workspace/bench/paired/table/" "$JOBDIR/bench/paired/table/" >/dev/null 2>&1 || true
    rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
        "root@$IP:/workspace/bench/paired/release-cap4096/" "$JOBDIR/bench/paired/release-cap4096/" >/dev/null 2>&1 || true
    rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
        "root@$IP:/workspace/bench/paired/table-cap4096/" "$JOBDIR/bench/paired/table-cap4096/" >/dev/null 2>&1 || true
    for f in paired-code-table.log smoke-release.log smoke-table.log table-export.log; do
        rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
            "root@$IP:/workspace/bench/$f" "$JOBDIR/bench/" >/dev/null 2>&1 || true
    done
    return 0
}
die() { log "FATAL: $*"; fetch_all; terminate; log "=== paired-table driver FAILED ==="; exit 1; }

log "=== paired-table driver start pod=$POD ip=$IP port=$PORT rate=$RATE floor=$FLOOR cap=${CAP_MIN}m"
[ -s "$TBL" ] || die "prebuilt table bytes missing: $TBL"

ssh_pod 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; df -h /workspace /root | tail -3' || die "ssh probe failed"
ssh_pod 'mkdir -p /workspace/bench/paired /root/models /root/community' || die "mkdir failed"

scp $SCPOPT "$RECIPE/paired-code-table.sh" root@$IP:/root/paired-code-table.sh || die "payload upload failed"
scp $SCPOPT "$RECIPE/pod-build-table.sh" root@$IP:/root/pod-build-table.sh || die "builder upload failed"
scp $SCPOPT "$WORK/hivebench/tools/strata-bridge/repack_experts_q2_0.py" root@$IP:/root/repack_experts_q2_0.py || die "exporter upload failed"
scp $SCPOPT "$WORK/scion/moe/ptq1_0.py" root@$IP:/root/ptq1_0.py || die "codec upload failed"
for p in bench-ple-multiseq server-skip-seq-rm embed-read-width router-stats kld-gate ple-table; do
    scp $SCPOPT "$RECIPE/$p.patch" "root@$IP:/root/$p.patch" || die "$p patch upload failed"
done
scp $SCPOPT "$CASCADE/community_eval.py" root@$IP:/root/community/community_eval.py || die "community_eval upload failed"
for m in checkers probe_judge readout_judge; do
    scp $SCPOPT "$CASCADE/$m.py" root@$IP:/root/community/$m.py || die "$m upload failed"
done
scp $SCPOPT "$CASCADE/tasks-community-code.json" root@$IP:/root/community/tasks-community-code.json || die "code tasks upload failed"

systemctl --user stop scion-pct-spend >/dev/null 2>&1 || true
systemctl --user reset-failed scion-pct-spend >/dev/null 2>&1 || true
systemd-run --user --unit=scion-pct-spend --collect \
    --property=StandardOutput=append:"$JOBDIR/pct-spend.log" \
    --property=StandardError=append:"$JOBDIR/pct-spend.log" \
    bash "$RECIPE/community-guard.sh" "$POD" "$FLOOR" "$CAP_MIN" || die "guard start failed"
log "guard started (floor $FLOOR, $CAP_MIN min cap)"

log "uploading prebuilt table bytes ($(du -h "$TBL" | cut -f1)) in 5G chunks"
STAGE=$JOBDIR/upload
rm -rf "$STAGE"; mkdir -p "$STAGE"
split -b 5000M -d --suffix-length=2 "$TBL" "$STAGE/chunk"
for f in "$STAGE"/chunk*; do
    ( scp $SCPOPT "$f" "root@$IP:/root/$(basename "$f")" || touch "$STAGE/.fail.$(basename "$f")" ) &
done
wait
if ls "$STAGE"/.fail.* >/dev/null 2>&1; then die "table bytes chunk upload failed"; fi
ssh_pod 'cd /root && cat chunk* > q4x160-table.bin && rm -f chunk*' || die "concat failed"
remote_size=$(ssh_pod 'stat -c %s /root/q4x160-table.bin' | tr -cd '0-9')
local_size=$(stat -c %s "$TBL")
[ "$remote_size" = "$local_size" ] || die "table bytes size mismatch remote=$remote_size local=$local_size"
rm -rf "$STAGE"
log "table bytes upload OK ($remote_size B)"

ssh_pod 'setsid nohup bash /root/paired-code-table.sh > /root/paired-code-table.out 2>&1 < /dev/null & echo started' || die "payload start failed"
log "payload started on the pod"

deadline=$(( $(date +%s) + CAP_MIN * 60 + 1800 ))
while :; do
    done_=$(ssh_pod 'grep -c "PAIRED_CODE_DONE" /workspace/bench/paired-code-table.log 2>/dev/null' 2>/dev/null | tr -cd '0-9')
    fail=$(ssh_pod 'grep -aoE "CLONE_FAILED|PATCH_FAILED \([a-z-]+\)|CMAKE_FAILED|BUILD_FAILED|DL_OURS_FAILED|SHA_FAILED|SMOKE_FAILED|SMOKE_TABLE_FAILED|RELEASE_SERVER_FAILED|TABLE_SERVER_FAILED|TABLE_EXPORT_FAILED" /workspace/bench/paired-code-table.log 2>/dev/null | head -1' 2>/dev/null)
    if [ "${done_:-0}" -ge 1 ]; then log "PAIRED_CODE_DONE seen"; break; fi
    if [ -n "$fail" ]; then log "REMOTE FAILURE: $fail"; break; fi
    if grep -q "deleting pod" "$JOBDIR/pct-spend.log" 2>/dev/null; then
        log "guard deleted the pod; exiting poll loop"; break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then log "TIMEOUT waiting for the run"; break; fi
    tail1=$(ssh_pod 'tail -1 /workspace/bench/paired-code-table.log 2>/dev/null' 2>/dev/null)
    [ -n "$tail1" ] && log "[remote] $tail1"
    sleep 90
done

log "fetching artifacts"
fetch_all
terminate
log "=== paired-table driver DONE ==="
