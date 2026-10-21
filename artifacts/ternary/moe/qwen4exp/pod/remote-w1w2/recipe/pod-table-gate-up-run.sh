#!/bin/bash
# pod-table-gate-up-run.sh - local driver for the upload-path Q4X160 gate:
#   upload (payload, exporter, codec, patches 1-6, KLD park, PREBUILT table
#   bytes) -> start table-gate-min-run.sh with TABLE_BYTES -> poll -> fetch ->
#   terminate.  No pod-side transcode, so host speed does not matter.
# Guard: scion-tgu-spend (floor $0.5, 150 min cap).
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
WORK=<work>
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
JOBDIR=${JOBDIR:-$Q/pod/remote-gate-q4x160-up}
RECIPE=$Q/pod/remote-w1w2/recipe
KEY=$HOME/.ssh/id_runpod
FLOOR=${FLOOR:-0.5}
PARK=$WORK/scion/moe/artifacts/qwen4exp/kld-park
TBL=${TBL:-$WORK/tmp-export/q4x160-table.bin}
PAR=${PAR:-0}
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
    systemctl --user stop scion-tgu-spend >/dev/null 2>&1 || true
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
    for f in kld-release.json kld-table.json kld-release.log kld-table.log \
             table-ppl.log table-hellaswag.log table-winogrande.log \
             table-export.log table-gate-min-run.log tiny-ppl-table.log \
             smoke-release.log smoke-table.log dl-ours.log; do
        rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
            "root@$IP:/workspace/bench/$f" "$JOBDIR/bench/" >/dev/null 2>&1 || true
    done
    return 0
}
die() { log "FATAL: $*"; fetch_all; terminate; log "=== q4x160-up driver FAILED ==="; exit 1; }

log "=== q4x160-up driver start pod=$POD ip=$IP port=$PORT rate=$RATE floor=$FLOOR par=$PAR"
if [ "$PAR" = 0 ]; then [ -s "$TBL" ] || die "prebuilt table bytes missing: $TBL"; fi

ssh_pod 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; df -h /workspace /root | tail -3' || die "ssh probe failed"
ssh_pod 'mkdir -p /workspace/bench /root/models /root/kld-park' || die "mkdir failed"

scp $SCPOPT "$RECIPE/table-gate-min-run.sh" root@$IP:/root/table-gate-min-run.sh || die "payload upload failed"
scp $SCPOPT "$RECIPE/pod-build-table.sh" root@$IP:/root/pod-build-table.sh || die "builder upload failed"
scp $SCPOPT "$WORK/hivebench/tools/strata-bridge/repack_experts_q2_0.py" root@$IP:/root/repack_experts_q2_0.py || die "exporter upload failed"
scp $SCPOPT "$WORK/hivebench/tools/strata-bridge/parallel_table_transcode.py" root@$IP:/root/parallel_table_transcode.py || die "parallel encoder upload failed"
scp $SCPOPT "$WORK/scion/moe/ptq1_0.py" root@$IP:/root/ptq1_0.py || die "codec upload failed"
for p in bench-ple-multiseq server-skip-seq-rm embed-read-width router-stats kld-gate ple-table; do
    scp $SCPOPT "$RECIPE/$p.patch" "root@$IP:/root/$p.patch" || die "$p patch upload failed"
done

systemctl --user stop scion-tgu-spend >/dev/null 2>&1 || true
systemctl --user reset-failed scion-tgu-spend >/dev/null 2>&1 || true
systemd-run --user --unit=scion-tgu-spend --collect \
    --property=StandardOutput=append:"$JOBDIR/tgu-spend.log" \
    --property=StandardError=append:"$JOBDIR/tgu-spend.log" \
    bash "$RECIPE/community-guard.sh" "$POD" "$FLOOR" 150 || die "guard start failed"
log "guard started (floor $FLOOR, 150 min cap)"

log "uploading KLD teacher park ($(du -sh "$PARK" | cut -f1))"
for i in $(seq 1 3); do
    rsync -a --partial --timeout=1200 -e "$SSH_FETCH" \
        "$PARK/" "root@$IP:/root/kld-park/" && { log "park upload OK"; break; }
    log "park upload attempt $i failed; retrying"
    sleep 10
done
park_ok=$(ssh_pod 'ls /root/kld-park/teacher 2>/dev/null | wc -l' | tr -cd '0-9')
[ "${park_ok:-0}" -ge 8 ] || die "park incomplete on pod ($park_ok windows)"

if [ "$PAR" = 0 ]; then
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
ssh_pod 'setsid nohup env TABLE_BYTES=/root/q4x160-table.bin bash /root/table-gate-min-run.sh > /root/table-gate-min-run.out 2>&1 < /dev/null & echo started' || die "payload start failed"
else
log "parallel mode: the pod transcodes the table across its cores"
ssh_pod 'setsid nohup env TABLE_PARALLEL=1 bash /root/table-gate-min-run.sh > /root/table-gate-min-run.out 2>&1 < /dev/null & echo started' || die "payload start failed"
fi
log "payload started on the pod"

deadline=$(( $(date +%s) + 21600 ))
while :; do
    done_=$(ssh_pod 'grep -c "TABLE_GATE_DONE" /workspace/bench/table-gate-min-run.log 2>/dev/null' 2>/dev/null | tr -cd '0-9')
    fail=$(ssh_pod 'grep -aoE "CLONE_FAILED|PATCH_FAILED \([a-z-]+\)|CMAKE_FAILED|BUILD_FAILED|DL_OURS_FAILED|SHA_FAILED|SMOKE_FAILED|SMOKE_TABLE_FAILED|HOST_TOO_SLOW|KLD_PARK_MISSING|KLD_RELEASE_FAILED|KLD_TABLE_FAILED|TABLE_EXPORT_FAILED|CODEC_UPLOAD_FAILED|TINY_PPL_FAILED|DL_WIKI_FAILED|DL_HS_FAILED|DL_WG_FAILED" /workspace/bench/table-gate-min-run.log 2>/dev/null | head -1' 2>/dev/null)
    if [ "${done_:-0}" -ge 1 ]; then log "TABLE_GATE_DONE seen"; break; fi
    if [ -n "$fail" ]; then log "REMOTE FAILURE: $fail"; break; fi
    if grep -q "deleting pod" "$JOBDIR/tgu-spend.log" 2>/dev/null; then
        log "guard deleted the pod; exiting poll loop"; break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then log "TIMEOUT waiting for the run"; break; fi
    tail1=$(ssh_pod 'tail -1 /workspace/bench/table-gate-min-run.log 2>/dev/null' 2>/dev/null)
    [ -n "$tail1" ] && log "[remote] $tail1"
    sleep 45
done

log "fetching artifacts"
fetch_all
for f in kld-release.json kld-table.json; do
    if [ -s "$JOBDIR/bench/$f" ]; then
        python3 - "$JOBDIR/bench/$f" <<'EOF' || true
import json, sys
d = json.load(open(sys.argv[1]))
print("%s: mean %.4f p50 %.4f p90 %.4f p99.9 %.4f max %.4f top1 %.4f" % (
    sys.argv[1].split("/")[-1], d["mean"], d["p50"], d["p90"], d["p99.9"], d["max"],
    d["top1_agreement"]))
EOF
    fi
done
terminate
log "=== q4x160-up driver DONE ==="
