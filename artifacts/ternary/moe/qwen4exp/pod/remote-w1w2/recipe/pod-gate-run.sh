#!/bin/bash
# pod-gate-run.sh - gate-only window driver (local box, detached):
#   upload (scion + deployed ckpt + teacher park) -> setup -> gate-student
#   (GATE_STEP=4000) -> fetch JSON+logs -> delete pod.
# Reads $JOBDIR/{pod-id,pod-endpoint,pod-rate}.  Guard: w4gate-spend cap 3.2,
# stop target = the gate JSON.
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
WORK=<work>
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
JOBDIR="${JOBDIR:-$Q/pod/remote-gate4000}"
WS=$WORK/scion/moe/pod_watch
KEY=$HOME/.ssh/id_runpod
QREQ=/workspace/artifacts/qwen4exp
BUNDLE=$Q/w1w2-upload/scion-flashnext-w1w2.tar.gz
CKPT=$JOBDIR/qwen4exp-corr-r512-g128-step4000-cur05-deployed.pt
TCACHE_TAR=$WORK/tmp-export/w4up/tcache-48l.tar
GATE_STEP="${GATE_STEP:-4000}"
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
    rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
        --include='qwen4exp-eval-*.json' --include='qwen4exp-corr-*.pt' \
        --include='*.log' --include='*.rc' --exclude='*' \
        "root@$IP:$QREQ/" "$JOBDIR/" >/dev/null 2>&1 || log "WARN: artifact rsync failed"
    rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
        "root@$IP:/workspace/logs/" "$JOBDIR/logs/" >/dev/null 2>&1 || log "WARN: log rsync failed"
    return 0
}
die() { log "FATAL: $*"; fetch_all; terminate; log "=== gate driver FAILED ==="; exit 1; }

log "=== gate-only driver start pod=$POD ip=$IP port=$PORT rate=$RATE step=$GATE_STEP"

ssh_pod 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; df -h /workspace | tail -1' || die "ssh probe failed"
if ! ssh_pod 'which rsync >/dev/null 2>&1'; then
    ssh_pod 'apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq rsync >/dev/null 2>&1; which rsync' || die "rsync install failed"
fi
ssh_pod 'mkdir -p /workspace/upload /workspace/artifacts/qwen4exp /workspace/logs' || die "mkdir failed"

# ---- upload (parallel streams; small set: 1.4 GB + 1.75 GB tar + bundle)
log "uploading scion + deployed ckpt + teacher park"
PIDS=()
scp $SCPOPT "$BUNDLE" "root@$IP:/workspace/upload/" & PIDS+=($!)
scp $SCPOPT "$CKPT" "root@$IP:/workspace/upload/" & PIDS+=($!)
scp $SCPOPT "$TCACHE_TAR" "root@$IP:/workspace/upload/" & PIDS+=($!)
bad=0; for p in "${PIDS[@]}"; do wait "$p" || bad=$((bad+1)); done
[ "$bad" -eq 0 ] || die "$bad upload stream(s) failed"

CSHA=$(sha256sum "$CKPT" | awk '{print $1}')
TSHA=$(sha256sum "$TCACHE_TAR" | awk '{print $1}')
log "remote place + sha (ckpt $CSHA tcache $TSHA)"
ssh_pod "Q=$QREQ; cd /workspace/upload && mv qwen4exp-corr-r512-g128-step${GATE_STEP}-cur05-deployed.pt \$Q/qwen4exp-corr-r512-g128-step${GATE_STEP}-cur05-deployed.pt && mv tcache-48l.tar /workspace/upload/tc.tar && tar --no-same-owner -xzf tc.tar -C \$Q && rm -f tc.tar && mkdir -p /workspace/scion && tar --no-same-owner -xzf scion-flashnext-w1w2.tar.gz -C /workspace/scion --strip-components=1 && sha256sum \$Q/qwen4exp-corr-r512-g128-step${GATE_STEP}-cur05-deployed.pt && ls \$Q/tcache-48l | wc -l && cat \$Q/tcache-48l/meta.json" > "$JOBDIR/gate-remote-shas.txt" 2>&1 || die "remote placement failed"
RSHA=$(grep -oE '^[0-9a-f]{64}' "$JOBDIR/gate-remote-shas.txt" | head -1)
[ "$RSHA" = "$CSHA" ] || die "remote ckpt sha '$RSHA' != $CSHA"
log "remote sha OK: ckpt ${CSHA:0:12}"

# ---- guards (cap 3.2; stop target = the gate JSON)
ts=$(date -u +%Y%m%dT%H%M%SZ)
for f in "$JOBDIR/.spend-guard-state" "$JOBDIR"/.stage-watch-*; do
    [ -e "$f" ] && mv "$f" "$f.prev-$ts"
done
systemctl --user reset-failed w4gate-spend w4gate-pull w4gate-watch >/dev/null 2>&1 || true
systemd-run --user --unit=w4gate-spend --collect \
    --property=StandardOutput=append:"$JOBDIR/w4gate-spend.log" \
    --property=StandardError=append:"$JOBDIR/w4gate-spend.log" \
    --setenv=PW_POD="$POD" --setenv=PW_SSH_HOST="root@$IP" --setenv=PW_SSH_PORT="$PORT" \
    --setenv=PW_DST="$JOBDIR" --setenv=PW_WINDOW_CAP=3.2 --setenv=PW_RATE_PER_HR="$RATE" \
    --setenv=PW_FETCH_MIN=10 --setenv=PW_MARGIN_MIN=10 --setenv=PW_RESERVE_FLOOR=0.5 \
    --setenv=PW_CKPTS="qwen4exp-eval-kld-48l-step${GATE_STEP}.json" \
    --setenv=PW_KILL_SIBLING="pod_watch/puller.sh" \
    bash "$WS/spend_guard.sh" || die "spend guard start failed"
systemd-run --user --unit=w4gate-pull --collect \
    --property=StandardOutput=append:"$JOBDIR/w4gate-pull.log" \
    --property=StandardError=append:"$JOBDIR/w4gate-pull.log" \
    --setenv=PW_POD="$POD" --setenv=PW_SSH_HOST="root@$IP" --setenv=PW_SSH_PORT="$PORT" \
    --setenv=PW_DST="$JOBDIR" --setenv=PW_VERIFY=1 \
    bash "$WS/puller.sh" >/dev/null 2>&1 || log "WARN: puller start failed"
log "guards started (cap 3.2)"

# ---- setup
log "launching setup"
ssh_pod 'cd /workspace/scion && bash moe/pod_watch/stage-exec.sh setup bash moe/box-run-flashnext.sh setup' || die "setup launch failed"
for i in $(seq 1 60); do
    rc=$(ssh_pod 'cat /workspace/logs/setup.rc 2>/dev/null' | tr -cd '0-9')
    [ -n "$rc" ] && break
    grep -q "STOPPING" "$JOBDIR/w4gate-spend.log" 2>/dev/null && die "spend guard stopped during setup"
    sleep 60
done
[ "${rc:-}" = "0" ] || die "setup rc=${rc:-none}"
log "setup done"

# ---- gate-student (reads the uploaded teacher park)
systemctl --user reset-failed w4gate-watch >/dev/null 2>&1 || true
systemd-run --user --unit=w4gate-watch --collect \
    --property=StandardOutput=append:"$JOBDIR/w4gate-watch.log" \
    --property=StandardError=append:"$JOBDIR/w4gate-watch.log" \
    --setenv=PW_POD="$POD" --setenv=PW_SSH_HOST="root@$IP" --setenv=PW_SSH_PORT="$PORT" \
    --setenv=PW_DST="$JOBDIR" --setenv=PW_STAGE=gate-student \
    --setenv=PW_LOG_IDLE_S=2700 --setenv=PW_STEP_IDLE_S=2700 \
    --setenv=PW_KILL_PATTERN="qwen4exp_eval.py" \
    bash "$WS/stage_watchdog.sh" >/dev/null 2>&1 || log "WARN: watchdog failed"
log "launching gate-student GATE_STEP=$GATE_STEP"
ssh_pod "cd /workspace/scion && bash moe/pod_watch/stage-exec.sh gate-student env GATE_STEP=$GATE_STEP bash moe/box-run-flashnext.sh gate-student" || die "gate launch failed"
for i in $(seq 1 240); do
    rc=$(ssh_pod 'cat /workspace/logs/gate-student.rc 2>/dev/null' | tr -cd '0-9')
    [ -n "$rc" ] && break
    grep -q "STOPPING" "$JOBDIR/w4gate-spend.log" 2>/dev/null && { log "spend guard stopped the pod"; break; }
    grep -q "STALL:" "$JOBDIR/w4gate-watch.log" 2>/dev/null && die "gate-student stalled"
    sleep 60
done
[ "${rc:-}" = "0" ] || log "WARN: gate rc=${rc:-none}"

# ---- fetch + verify + terminate
log "fetching artifacts"
fetch_all
GATEJSON=$JOBDIR/qwen4exp-eval-kld-48l-step${GATE_STEP}.json
[ -s "$GATEJSON" ] || die "gate JSON missing locally"
python3 - "$GATEJSON" "$GATE_STEP" <<'EOF' || true
import json, sys
d = json.load(open(sys.argv[1]))
step = sys.argv[2] if len(sys.argv) > 2 else "?"
print("GATE step%s: n_tokens %s mean %.6f p50 %.4f p99 %.4f p99.9 %.4f max %.4f" % (
    step, d["n_tokens"], d["mean"], d["p50"], d["p99"], d["p99.9"], d["max"]))
EOF
terminate
log "=== gate-only driver DONE ==="
