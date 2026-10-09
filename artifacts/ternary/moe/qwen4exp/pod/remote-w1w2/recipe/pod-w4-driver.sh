#!/bin/bash
# pod-w4-driver.sh - W4 window driver (runs on the local box, detached).
#
#   upload (parallel chunks) -> setup -> train-resume (3000->4096) ->
#   deploy-checkpoint -> gate-student (GATE_STEP=4096) -> fetch -> terminate
#
# Reads $W4DIR/{pod-id,pod-endpoint,pod-rate}.  Guards: spend_guard cap $9 is
# started FIRST (systemd unit w4-spend; stops when the gate JSON is fetched +
# sha-verified), puller unit w4-pull, stage watchdog units w4-watch-train /
# w4-watch-gate.  The puller keeps local copies every 5 min.
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
WORK=/home/penis/Desktop/work
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
W4DIR=$Q/pod/remote-w4
RECIPE=$Q/pod/remote-w1w2/recipe
WS=$WORK/scion/moe/pod_watch
SD=$WORK/tmp-export/w4up
KEY=$HOME/.ssh/id_runpod
QREQ=/workspace/artifacts/qwen4exp
mkdir -p "$W4DIR" "$W4DIR/logs"

POD=$(cat "$W4DIR/pod-id")
read -r IP PORT < "$W4DIR/pod-endpoint"
RATE=$(cat "$W4DIR/pod-rate")
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i $KEY -p $PORT"
SCPOPT="-q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o Compression=no -i $KEY -P $PORT"
SSH="ssh $SSHOPT root@$IP"
SSH_FETCH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -p $PORT"

log() { echo "$(date -u +%FT%TZ) $*"; }
ssh_pod() { $SSH "$@"; }

pod_exists() {
    curl -s --max-time 30 https://rest.runpod.io/v1/pods \
        -H "Authorization: Bearer $(cat "$HOME/.runpod-api-key")" \
        | grep -q "\"$POD\""
}

terminate() {
    curl -s --max-time 60 -X DELETE https://rest.runpod.io/v1/pods/$POD \
        -H "Authorization: Bearer $(cat "$HOME/.runpod-api-key")" >/dev/null 2>&1 || true
    sleep 5
    if pod_exists; then
        log "WARN: pod $POD still listed after DELETE"
    else
        log "pod $POD terminated (not listed)"
    fi
}

fetch_all() {
    rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
        --include='qwen4exp-corr-*.pt' --include='qwen4exp-eval-*.json' \
        --include='*.log' --include='*.rc' --exclude='*' \
        "root@$IP:$QREQ/" "$W4DIR/" >/dev/null 2>&1 || log "WARN: artifact rsync failed"
    rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
        "root@$IP:/workspace/logs/" "$W4DIR/logs/" >/dev/null 2>&1 || log "WARN: log rsync failed"
    return 0
}

die() {
    log "FATAL: $*"
    fetch_all
    terminate
    log "=== pod-w4-driver FAILED ==="
    exit 1
}

# ---------------------------------------------------------------- staging
EXP_CACHE_SHA=43a391d1668e8ad5aa0c372cbf006db4aba125b0fff913e2596c5b636119a490
EXP_CKPT_SHA=4b14f32b166ba2daffdb3199be54a80e0a79ca7e8730fe78eeb362a739ae29fb
EXP_REF_SHA=57ee452af5ad9f644037086ae79f70b9df46b8b3283c749ccff77285032debf3
EXP_MIX_SHA=e59c7d93316e8c95cecb53c99c9507b4e28a1c0ca8ecb29126ee1626af45f26d
EXP_BUNDLE_SHA=ae237378a38de649a3446d22f4ae0d8a7d434c985630ceff1865b57f6da9a999
EXP_TCACHE_SHA=b0804e2ab35f18ab718fca2f62d9bed94b8c54c7542d92d18634ae02a72561ba

stage_chunks() {
    [ -s "$SD/cache.05" ] && [ -s "$SD/ckpt.00" ] && [ -s "$SD/tc.00" ] && return 0
    log "staging upload chunks"
    rm -rf "$SD"; mkdir -p "$SD"
    tar -C "$Q/pod/remote-w1w2" -czf "$SD/tcache-48l.tar" tcache-48l || return 1
    split -b 2G -d -a 2 "$Q/prefix-top512-tail64.pt" "$SD/cache." || return 1
    split -b 2G -d -a 2 "$Q/pod/remote-p2b/qwen4exp-corr-r512-g128-step3000-cur05.pt" "$SD/ckpt." || return 1
    split -b 2G -d -a 2 "$SD/tcache-48l.tar" "$SD/tc." || return 1
    cp "$Q/w1w2-upload/scion-flashnext-w1w2.tar.gz" "$SD/"
    cp "$Q/eval-ref-w2.pt" "$Q/pod/curric-combo.jsonl" "$SD/"
}

verify_local_chunks() {
    log "verifying local chunk concatenations"
    [ "$(cat "$SD"/cache.* | sha256sum | awk '{print $1}')" = "$EXP_CACHE_SHA" ] || return 1
    [ "$(cat "$SD"/ckpt.* | sha256sum | awk '{print $1}')" = "$EXP_CKPT_SHA" ] || return 1
    [ "$(cat "$SD"/tc.* | sha256sum | awk '{print $1}')" = "$EXP_TCACHE_SHA" ] || return 1
    return 0
}

# ---------------------------------------------------------------- upload
upload() {
    log "ssh probe + disk/rsync check"
    ssh_pod 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; df -h /workspace | tail -1' || die "ssh probe failed"
    # rsync is needed by the guards; install if the image lacks it
    if ! ssh_pod 'which rsync >/dev/null 2>&1'; then
        log "installing rsync on the pod"
        ssh_pod 'apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq rsync >/dev/null 2>&1; which rsync' || die "rsync install failed"
    fi
    ssh_pod 'mkdir -p /workspace/upload /workspace/artifacts/qwen4exp /workspace/logs' || die "mkdir failed"

    # Reuse path: a previous driver run may have staged everything already
    # (driver restart after a local bug).  Verify remotely and skip upload.
    if ssh_pod 'test -s /workspace/scion/moe/box-run-flashnext.sh && test -s /workspace/artifacts/qwen4exp/tcache-48l/meta.json'; then
        local rs cks rrs rcs
        rs=$(ssh_pod "sha256sum $QREQ/prefix-top512-tail64.pt" 2>/dev/null | awk '{print $1}')
        cks=$(ssh_pod "sha256sum $QREQ/qwen4exp-corr-r512-g128-step3000-cur05.pt" 2>/dev/null | awk '{print $1}')
        rrs=$(ssh_pod "sha256sum $QREQ/eval-ref-w2.pt" 2>/dev/null | awk '{print $1}')
        rcs=$(ssh_pod "sha256sum $QREQ/curric-combo.jsonl" 2>/dev/null | awk '{print $1}')
        if [ "$rs" = "$EXP_CACHE_SHA" ] && [ "$cks" = "$EXP_CKPT_SHA" ] && \
           [ "$rrs" = "$EXP_REF_SHA" ] && [ "$rcs" = "$EXP_MIX_SHA" ]; then
            log "remote inputs already staged + sha-verified; skipping upload"
            return 0
        fi
        log "remote inputs present but shas differ; re-uploading"
    fi

    stage_chunks || die "local staging failed"
    verify_local_chunks || die "local chunk sha mismatch (staging corrupt)"

    log "parallel upload of $(du -sh "$SD" | awk '{print $1}') in $(ls "$SD" | grep -cE '^(cache|ckpt|tc)\.') chunks"
    local pids=() f
    for f in "$SD"/cache.* "$SD"/ckpt.* "$SD"/tc.* "$SD/scion-flashnext-w1w2.tar.gz" "$SD/eval-ref-w2.pt" "$SD/curric-combo.jsonl"; do
        scp $SCPOPT "$f" "root@$IP:/workspace/upload/" 2>>"$W4DIR/w4-upload.err" &
        pids+=($!)
    done
    local bad=0 p
    for p in "${pids[@]}"; do wait "$p" || bad=$((bad+1)); done
    [ "$bad" -eq 0 ] || die "$bad upload stream(s) failed"

    log "remote concat + sha verify"
    ssh_pod 'cd /workspace/upload && cat cache.0* > f_cache && rm -f cache.0* && cat ckpt.0* > f_ckpt && rm -f ckpt.0* && cat tc.0* > f_tc && rm -f tc.0* && sha256sum f_cache f_ckpt f_tc' > "$W4DIR/w4-remote-shas.txt" 2>&1 || die "remote concat failed"
    local rc_sha ck_sha tc_sha
    rc_sha=$(awk '$2=="f_cache"{print $1}' "$W4DIR/w4-remote-shas.txt")
    ck_sha=$(awk '$2=="f_ckpt"{print $1}' "$W4DIR/w4-remote-shas.txt")
    tc_sha=$(awk '$2=="f_tc"{print $1}' "$W4DIR/w4-remote-shas.txt")
    [ "$rc_sha" = "$EXP_CACHE_SHA" ] || die "remote cache sha $rc_sha != $EXP_CACHE_SHA"
    [ "$ck_sha" = "$EXP_CKPT_SHA" ] || die "remote ckpt sha $ck_sha != $EXP_CKPT_SHA"
    [ "$tc_sha" = "$EXP_TCACHE_SHA" ] || die "remote tcache sha $tc_sha != $EXP_TCACHE_SHA"
    log "remote shas OK: cache ${rc_sha:0:12} ckpt ${ck_sha:0:12} tcache ${tc_sha:0:12}"

    ssh_pod "Q=$QREQ; mv /workspace/upload/f_cache \$Q/prefix-top512-tail64.pt && mv /workspace/upload/f_ckpt \$Q/qwen4exp-corr-r512-g128-step3000-cur05.pt && mv /workspace/upload/f_tc /workspace/upload/tc.tar && tar --no-same-owner -xzf /workspace/upload/tc.tar -C \$Q && rm -f /workspace/upload/tc.tar && mv /workspace/upload/eval-ref-w2.pt \$Q/ && mv /workspace/upload/curric-combo.jsonl \$Q/ && mkdir -p /workspace/scion && tar --no-same-owner -xzf /workspace/upload/scion-flashnext-w1w2.tar.gz -C /workspace/scion --strip-components=1 && ls -la \$Q && ls \$Q/tcache-48l | head -3 && cat \$Q/tcache-48l/meta.json" || die "remote placement failed"
    log "upload + placement done"
}

# ---------------------------------------------------------------- guards
start_guards() {
    local ts
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    for f in "$W4DIR/.spend-guard-state" "$W4DIR"/.stage-watch-*; do
        [ -e "$f" ] && mv "$f" "$f.prev-$ts"
    done
    systemctl --user reset-failed w4-spend w4-pull w4-watch-train w4-watch-gate >/dev/null 2>&1 || true
    systemd-run --user --unit=w4-spend --collect \
        --property=StandardOutput=append:"$W4DIR/w4-spend.log" \
        --property=StandardError=append:"$W4DIR/w4-spend.log" \
        --setenv=PW_POD="$POD" --setenv=PW_SSH_HOST="root@$IP" --setenv=PW_SSH_PORT="$PORT" \
        --setenv=PW_DST="$W4DIR" --setenv=PW_WINDOW_CAP=9 --setenv=PW_RATE_PER_HR="$RATE" \
        --setenv=PW_FETCH_MIN=15 --setenv=PW_MARGIN_MIN=10 --setenv=PW_RESERVE_FLOOR=0.5 \
        --setenv=PW_CKPTS=qwen4exp-eval-kld-48l-step4096.json \
        --setenv=PW_KILL_SIBLING=pod_watch/puller.sh \
        bash "$WS/spend_guard.sh" || die "spend_guard start failed"
    systemd-run --user --unit=w4-pull --collect \
        --property=StandardOutput=append:"$W4DIR/w4-pull.log" \
        --property=StandardError=append:"$W4DIR/w4-pull.log" \
        --setenv=PW_POD="$POD" --setenv=PW_SSH_HOST="root@$IP" --setenv=PW_SSH_PORT="$PORT" \
        --setenv=PW_DST="$W4DIR" --setenv=PW_VERIFY=1 \
        bash "$WS/puller.sh" || log "WARN: puller start failed"
    log "guards started (cap 9, rate $RATE, stop target = gate JSON)"
}

start_stage_watch() { # start_stage_watch <unit> <stage> <logidle> <stepidle> <pattern>
    local unit=$1 stage=$2 li=$3 si=$4 pat=$5
    systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
    systemd-run --user --unit="$unit" --collect \
        --property=StandardOutput=append:"$W4DIR/$unit.log" \
        --property=StandardError=append:"$W4DIR/$unit.log" \
        --setenv=PW_POD="$POD" --setenv=PW_SSH_HOST="root@$IP" --setenv=PW_SSH_PORT="$PORT" \
        --setenv=PW_DST="$W4DIR" --setenv=PW_STAGE="$stage" \
        --setenv=PW_LOG_IDLE_S="$li" --setenv=PW_STEP_IDLE_S="$si" \
        --setenv=PW_KILL_PATTERN="$pat" \
        bash "$WS/stage_watchdog.sh" >/dev/null 2>&1 || log "WARN: watchdog $unit failed to start"
}

# ---------------------------------------------------------------- waiting
# wait_rc <stage> <timeout_s> [watch_unit]
# RC global; return 0 = rc known, 2 = pod gone/guard stop, 3 = watchdog stall,
# 1 = timeout
wait_rc() {
    local stage=$1 tmo=$2 watch=${3:-} deadline cycle=0 fails=0 rc last
    deadline=$(( $(date +%s) + tmo ))
    while :; do
        rc=$(ssh_pod "cat /workspace/logs/$stage.rc 2>/dev/null" 2>/dev/null | tr -cd '0-9')
        if [ -n "$rc" ]; then RC=$rc; return 0; fi
        if grep -q "STOPPING" "$W4DIR/w4-spend.log" 2>/dev/null; then
            log "spend guard stopped the pod"; return 2
        fi
        if [ -n "$watch" ] && grep -q "STALL:" "$W4DIR/$watch.log" 2>/dev/null; then
            log "stage watchdog $watch stalled $stage"; return 3
        fi
        if ! ssh_pod true 2>/dev/null; then
            fails=$((fails+1))
            if [ "$fails" -ge 5 ] && ! pod_exists; then
                log "pod gone (ssh down, not in API)"; return 2
            fi
        else
            fails=0
        fi
        cycle=$((cycle+1))
        if [ $((cycle % 5)) -eq 0 ]; then
            last=$(ssh_pod "tail -n 2 /workspace/logs/$stage.log" 2>/dev/null | tail -1)
            [ -n "$last" ] && log "[$stage] $last"
            if [ "$stage" = "train-resume" ]; then
                local nan
                nan=$(ssh_pod "tail -n 400 /workspace/logs/$stage.log" 2>/dev/null | grep -cE 'lm nan|kd nan|total nan')
                if [ "${nan:-0}" -gt 0 ]; then
                    log "NaN detected in $stage; killing stage"
                    local pid
                    pid=$(ssh_pod "cat /workspace/logs/$stage.pid" 2>/dev/null | tr -cd '0-9')
                    [ -n "$pid" ] && ssh_pod "kill -TERM -$pid; kill -TERM $pid" >/dev/null 2>&1
                    return 3
                fi
            fi
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            log "WAIT TIMEOUT for $stage (${tmo}s)"; return 1
        fi
        sleep 60
    done
}

run_stage() { # run_stage <stage> <cmd...>
    local stage=$1; shift
    log "launching stage $stage"
    ssh_pod "cd /workspace/scion && bash moe/pod_watch/stage-exec.sh $stage $*" || die "stage $stage launch failed"
}

# ================================================================ main
log "=== pod-w4-driver start pod=$POD ip=$IP port=$PORT rate=$RATE"

upload
start_guards

# 1. setup (~20 min: pinned transformers + kernels + 173 GiB teacher)
run_stage setup bash moe/box-run-flashnext.sh setup
wait_rc setup 3600
WSET=$?
if [ "$WSET" -ne 0 ]; then
    die "setup did not complete (wait_rc=$WSET)"
fi
[ "$RC" = "0" ] || die "setup rc=$RC"
log "setup done"

# 2. train-resume (~6 h: load/compact ~45 min + 1096 steps)
start_stage_watch w4-watch-train train-resume 2700 2700 "qwen4exp_proxy.py train"
run_stage train-resume bash moe/box-run-flashnext.sh train-resume
wait_rc train-resume 28800 w4-watch-train
WRC=$?
if [ "$WRC" -eq 3 ]; then
    fetch_all; die "train-resume stalled/NaN"
elif [ "$WRC" -eq 1 ]; then
    fetch_all; die "train-resume timeout"
elif [ "$WRC" -eq 2 ]; then
    log "pod gone during train-resume; verifying local artifacts"
    fetch_all
    terminate
    log "=== pod-w4-driver ENDED EARLY (pod gone) ==="
    exit 2
fi
if [ "${RC:-}" != "0" ]; then
    fetch_all; die "train-resume rc=$RC"
fi
log "train-resume done rc=0"

# 3. deploy-checkpoint (raw 4096 -> deployed)
log "deploy-checkpoint"
# the exporter imports `gguf`; the pod image lacks it (W4 failure)
ssh_pod 'python3 -m pip install -q --break-system-packages gguf >/dev/null 2>&1 || pip install -q --break-system-packages gguf >/dev/null 2>&1; python3 -c "import gguf; print(\"gguf\", gguf.__version__ if hasattr(gguf,\"__version__\") else \"ok\")"' || log "WARN: gguf install probe failed"
ssh_pod 'cd /workspace/scion && bash moe/box-run-flashnext.sh deploy-checkpoint' 2>&1 | tee "$W4DIR/w4-deploy.log"
ssh_pod "ls -la $QREQ/qwen4exp-corr-r512-g128-step4096-cur05-deployed.pt" || die "deployed checkpoint missing"

# 4. gate-student on the uploaded teacher park (GATE_STEP=4096)
start_stage_watch w4-watch-gate gate-student 2400 3600 "qwen4exp_eval.py"
run_stage gate-student env GATE_STEP=4096 bash moe/box-run-flashnext.sh gate-student
wait_rc gate-student 14400 w4-watch-gate
GWR=$?
if [ "$GWR" -eq 2 ]; then
    log "pod gone during gate (guard stop?); verifying local artifacts"
elif [ "$GWR" -eq 1 ] || [ "$GWR" -eq 3 ]; then
    fetch_all; die "gate-student timeout/stall"
elif [ "${RC:-}" != "0" ]; then
    fetch_all; die "gate-student rc=$RC"
fi
log "gate-student done"

# 5. fetch + verify + terminate
log "fetching artifacts"
fetch_all

GATEJSON=$W4DIR/qwen4exp-eval-kld-48l-step4096.json
RAW=$W4DIR/qwen4exp-corr-r512-g128-step4096-cur05-4096.pt
DEP=$W4DIR/qwen4exp-corr-r512-g128-step4096-cur05-deployed.pt
for f in "$GATEJSON" "$RAW" "$DEP"; do
    [ -s "$f" ] || die "artifact missing locally: $f"
done
python3 - "$GATEJSON" <<'EOF' || true
import json, sys
d = json.load(open(sys.argv[1]))
print("GATE: n_tokens %(n_tokens)s mean %(mean).6f p50 %(p50).4f p99 %(p99).4f p99.9 %(p99.9).4f max %(max).4f" % d)
EOF
for f in "$RAW" "$DEP"; do
    if ssh_pod true 2>/dev/null; then
        r=$(ssh_pod "sha256sum $QREQ/$(basename "$f")" | awk '{print $1}')
        l=$(sha256sum "$f" | awk '{print $1}')
        if [ -n "$r" ] && [ "$r" != "$l" ]; then
            die "sha mismatch $(basename "$f"): remote ${r:0:12} local ${l:0:12}"
        fi
        log "sha OK $(basename "$f") ${l:0:12}"
    else
        log "pod unreachable; local $(basename "$f") sha $(sha256sum "$f" | awk '{print $1}' | cut -c1-12)"
    fi
done

terminate
log "=== pod-w4-driver DONE ==="
