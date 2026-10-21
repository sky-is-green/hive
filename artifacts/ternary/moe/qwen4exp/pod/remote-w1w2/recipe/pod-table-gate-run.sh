#!/bin/bash
# pod-table-gate-run.sh - local driver for the Q3X32 table gate window:
#   upload (payload, exporter, codec, patches 1-6, KLD park, community files)
#   -> start table-gate-run.sh detached -> poll -> fetch JSON/logs -> terminate.
# Guard: scion-tg-spend (floor $0.5, 330 min window cap).
# Reads $JOBDIR/{pod-id,pod-endpoint,pod-rate}.
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
WORK=<work>
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
CASCADE=$WORK/hivebench/experiments/cascade
JOBDIR=${JOBDIR:-$Q/pod/remote-gate-table}
RECIPE=$Q/pod/remote-w1w2/recipe
KEY=$HOME/.ssh/id_runpod
CAP=${CAP:-3.5}
FLOOR=${FLOOR:-0.5}
PARK=$WORK/scion/moe/artifacts/qwen4exp/kld-park
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
    systemctl --user stop scion-tg-spend >/dev/null 2>&1 || true
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
             table-export.log table-gate-run.log tiny-ppl-table.log \
             smoke-release.log smoke-table.log dl-ours.log; do
        rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
            "root@$IP:/workspace/bench/$f" "$JOBDIR/bench/" >/dev/null 2>&1 || true
    done
    rsync -a --partial --timeout=900 -e "$SSH_FETCH" \
        "root@$IP:/workspace/bench/paired/" "$JOBDIR/bench/paired/" >/dev/null 2>&1 || true
    return 0
}
die() { log "FATAL: $*"; fetch_all; terminate; log "=== table-gate driver FAILED ==="; exit 1; }

log "=== table-gate driver start pod=$POD ip=$IP port=$PORT rate=$RATE cap=$CAP floor=$FLOOR"

ssh_pod 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; df -h /workspace /root | tail -3' || die "ssh probe failed"
ssh_pod 'mkdir -p /workspace/bench /root/models /root/kld-park /root/community' || die "mkdir failed"

scp $SCPOPT "$RECIPE/table-gate-run.sh" root@$IP:/root/table-gate-run.sh || die "payload upload failed"
scp $SCPOPT "$RECIPE/pod-build-table.sh" root@$IP:/root/pod-build-table.sh || die "builder upload failed"
scp $SCPOPT "$WORK/hivebench/tools/strata-bridge/repack_experts_q2_0.py" root@$IP:/root/repack_experts_q2_0.py || die "exporter upload failed"
scp $SCPOPT "$WORK/scion/moe/ptq1_0.py" root@$IP:/root/ptq1_0.py || die "codec upload failed"
for p in bench-ple-multiseq server-skip-seq-rm embed-read-width router-stats kld-gate ple-table; do
    scp $SCPOPT "$RECIPE/$p.patch" "root@$IP:/root/$p.patch" || die "$p patch upload failed"
done
# functional code gate: the 164 HumanEval tasks through the community harness
scp $SCPOPT "$CASCADE/community_eval.py" root@$IP:/root/community/community_eval.py || die "community_eval upload failed"
for m in checkers probe_judge readout_judge; do
    scp $SCPOPT "$CASCADE/$m.py" root@$IP:/root/community/$m.py || die "$m upload failed"
done
scp $SCPOPT "$CASCADE/tasks-community-code.json" root@$IP:/root/community/tasks-community-code.json || die "code tasks upload failed"

log "uploading KLD teacher park ($(du -sh "$PARK" | cut -f1))"
for i in $(seq 1 3); do
    rsync -a --partial --timeout=1200 --info=progress2 -e "$SSH_FETCH" \
        "$PARK/" "root@$IP:/root/kld-park/" && { log "park upload OK"; break; }
    log "park upload attempt $i failed; retrying"
    sleep 10
done
park_ok=$(ssh_pod 'ls /root/kld-park/teacher 2>/dev/null | wc -l' | tr -cd '0-9')
[ "${park_ok:-0}" -ge 8 ] || die "park incomplete on pod ($park_ok windows)"

systemctl --user stop scion-tg-spend >/dev/null 2>&1 || true
systemctl --user reset-failed scion-tg-spend >/dev/null 2>&1 || true
systemd-run --user --unit=scion-tg-spend --collect \
    --property=StandardOutput=append:"$JOBDIR/tg-spend.log" \
    --property=StandardError=append:"$JOBDIR/tg-spend.log" \
    bash "$RECIPE/community-guard.sh" "$POD" "$FLOOR" 330 || die "guard start failed"
log "guard started (floor $FLOOR, 330 min cap, window cap \$$CAP)"

ssh_pod 'setsid nohup bash /root/table-gate-run.sh > /root/table-gate-run.out 2>&1 < /dev/null & echo started' || die "payload start failed"
log "payload started on the pod"

deadline=$(( $(date +%s) + 43200 ))
while :; do
    done_=$(ssh_pod 'grep -c "TABLE_GATE_DONE" /workspace/bench/table-gate-run.log 2>/dev/null' 2>/dev/null | tr -cd '0-9')
    fail=$(ssh_pod 'grep -aoE "CLONE_FAILED|PATCH_FAILED \([a-z-]+\)|CMAKE_FAILED|BUILD_FAILED|DL_OURS_FAILED|SHA_FAILED|SMOKE_FAILED|SMOKE_TABLE_FAILED|KLD_PARK_MISSING|KLD_RELEASE_FAILED|KLD_TABLE_FAILED|TABLE_EXPORT_FAILED|CODEC_UPLOAD_FAILED|TINY_PPL_FAILED|DL_WIKI_FAILED|DL_HS_FAILED|DL_WG_FAILED|RELEASE_SERVER_FAILED|TABLE_SERVER_FAILED" /workspace/bench/table-gate-run.log 2>/dev/null | head -1' 2>/dev/null)
    if [ "${done_:-0}" -ge 1 ]; then log "TABLE_GATE_DONE seen"; break; fi
    if [ -n "$fail" ]; then log "REMOTE FAILURE: $fail"; break; fi
    if grep -q "deleting pod" "$JOBDIR/tg-spend.log" 2>/dev/null; then
        log "guard deleted the pod; exiting poll loop"; break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then log "TIMEOUT waiting for the run"; break; fi
    tail1=$(ssh_pod 'tail -1 /workspace/bench/table-gate-run.log 2>/dev/null' 2>/dev/null)
    [ -n "$tail1" ] && log "[remote] $tail1"
    sleep 60
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
log "=== table-gate driver DONE ==="
