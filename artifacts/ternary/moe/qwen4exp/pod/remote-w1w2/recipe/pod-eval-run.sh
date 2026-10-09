#!/bin/bash
# pod-eval-run.sh - eval window driver (local box, detached):
#   apt/build fork CPU-only -> parallel chunk upload -> sha -> cap_eval
#   (server backend) -> fetch JSON+log -> delete pod.
# Reads $EVAL_JOBDIR/{pod-id,pod-endpoint,eval-gguf,eval-name,eval-log,eval-sha}.
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
WORK=/home/penis/Desktop/work
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
EVAL_JOBDIR="${EVAL_JOBDIR:-$Q/pod/remote-eval}"
STAGE="${EVAL_STAGE:-/run/media/penis/30CE2C97CE2C577E/storage/evalstage}"
SRC_TAR="${EVAL_SRC_TAR:-$Q/pod/llama-qwen4exp-src.tar.gz}"
KEY=$HOME/.ssh/id_runpod
mkdir -p "$EVAL_JOBDIR"

POD=$(cat "$EVAL_JOBDIR/pod-id")
read -r IP PORT < "$EVAL_JOBDIR/pod-endpoint"
GGUF=$(cat "$EVAL_JOBDIR/eval-gguf")
NAME=$(cat "$EVAL_JOBDIR/eval-name")
LOGTAG=$(cat "$EVAL_JOBDIR/eval-log")
SHA=$(cat "$EVAL_JOBDIR/eval-sha")
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -i $KEY -p $PORT"
SCPOPT="-q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o Compression=no -i $KEY -P $PORT"
SSH="ssh $SSHOPT root@$IP"

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
die() { log "FATAL: $*"; terminate; log "=== eval driver FAILED ==="; exit 1; }

log "=== eval driver start pod=$POD ip=$IP port=$PORT model=$NAME"

ssh_pod 'nproc; free -g | head -2; df -h /workspace /root 2>/dev/null | tail -2' || die "ssh probe failed"

# 1. upload source + cap_eval, build CPU-only FIRST (binary verified before
#    the 65 GB upload).
log "uploading fork source + cap_eval"
scp $SCPOPT "$SRC_TAR" root@$IP:/root/ || die "source upload failed"
scp $SCPOPT "$WORK/scion/moe/cap_eval.py" "$WORK/scion/moe/cap_tasks.json" root@$IP:/root/ || die "cap_eval upload failed"
ssh_pod 'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq rsync ninja-build cmake build-essential >/dev/null 2>&1; cd /root && tar -xzf llama-qwen4exp-src.tar.gz && ls llama-qwen4exp/src/models/qwen4exp.cpp llama-qwen4exp/common/build-info.cpp.in' || die "source extract failed"
log "building fork CPU-only"
ssh_pod 'cd /root/llama-qwen4exp && rm -rf build && (nohup bash -c "cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && cmake --build build -j\$(nproc) --target llama-server" > /root/build.log 2>&1 &); echo build-started'
for i in $(seq 1 90); do
    if ssh_pod 'test -x /root/llama-qwen4exp/build/bin/llama-server'; then
        log "build OK after ~$((i*20))s"; break
    fi
    if [ "$i" -ge 3 ] && ! ssh_pod 'pgrep -f "cmake --build" >/dev/null || pgrep -x ninja >/dev/null || pgrep -x cc1plus >/dev/null || pgrep -x g++ >/dev/null || pgrep -f "cmake -B" >/dev/null'; then
        ssh_pod 'tail -30 /root/build.log'
        die "build process ended without a binary"
    fi
    sleep 20
done
ssh_pod 'test -x /root/llama-qwen4exp/build/bin/llama-server' || die "no llama-server after wait"
log "build verified"

# 2. parallel chunk upload
FREE=$(df -Pk "$(dirname "$STAGE")" | awk 'NR==2{print $4}')
NEED=$(( $(stat -c %s "$GGUF") / 1024 + 5242880 ))
[ "${FREE:-0}" -gt "$NEED" ] || die "not enough local disk for chunk staging (free ${FREE}kB need ${NEED}kB)"
rm -rf "$STAGE"; mkdir -p "$STAGE"
log "splitting $NAME ($(du -h "$GGUF" | cut -f1))"
split -b 7G -d --suffix-length=1 "$GGUF" "$STAGE/chunk"
PIDS=()
for f in "$STAGE"/chunk*; do
    scp $SCPOPT "$f" root@$IP:/workspace/ 2>>"$EVAL_JOBDIR/eval-upload.err" &
    PIDS+=($!)
done
bad=0
for p in "${PIDS[@]}"; do wait "$p" || bad=$((bad+1)); done
rm -f "$STAGE"/chunk*
[ "$bad" -eq 0 ] || die "$bad upload stream(s) failed"

log "remote concat + sha"
ssh_pod "cd /workspace && rm -f '$NAME' && for f in \$(ls chunk* | sort); do cat \"\$f\" >> '$NAME' && rm -f \"\$f\"; done; sha256sum '$NAME'" > "$EVAL_JOBDIR/eval-remote-sha.txt" 2>&1
RSHA=$(ssh_pod "sha256sum /workspace/$NAME" | awk '{print $1}')
log "remote sha $RSHA (expect $SHA)"
[ "$RSHA" = "$SHA" ] || die "sha mismatch remote $RSHA != $SHA"

# 3. cap_eval (CPU, server backend)
log "cap_eval start"
if ! ssh_pod "cd /root && python3 cap_eval.py --model /workspace/$NAME --backend server --server-bin /root/llama-qwen4exp/build/bin/llama-server --server-args \"-c 512 -t 16\" --server-log /root/$LOGTAG-server.log --out /root/$LOGTAG.json --timeout 1800" 2>&1 | tail -30; then
    log "WARN: cap_eval returned nonzero"
fi
scp $SCPOPT root@$IP:/root/$LOGTAG.json "$Q/$LOGTAG.json" && log "fetched JSON"
scp $SCPOPT root@$IP:/root/$LOGTAG-server.log "$Q/$LOGTAG-server.log" && log "fetched server log"
python3 - "$Q/$LOGTAG.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("RESULT: ok %d / %d = %.3f" % (d["ok"], d["n"], d["accuracy"]))
print("harness errors:", sum(1 for r in d["results"] if r.get("error")))
for c, v in d["by_category"].items():
    print("  %-12s %d/%d" % (c, v["ok"], v["n"]))
EOF

log "deleting pod"
terminate
log "=== eval driver DONE ==="
