#!/bin/bash
# watch-community-resume.sh - local watcher for community-resume.sh on the pod:
#   poll the resume log -> fetch artifacts -> print the eval table -> terminate.
# The scion-community-driver unit is stopped for this path (it would terminate
# the pod as soon as it sees the first run's COMMUNITY_DONE marker).
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
WORK=/home/penis/Desktop/work
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
JOBDIR=$Q/pod/remote-community
KEY=$HOME/.ssh/id_runpod
POD=$(cat "$JOBDIR/pod-id")
read -r IP PORT < "$JOBDIR/pod-endpoint"
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i $KEY -p $PORT"
SSH_FETCH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -p $PORT"
log() { echo "$(date -u +%FT%TZ) $*"; }
sshp() { ssh $SSHOPT root@$IP "$@"; }

terminate() {
    systemctl --user stop scion-community-spend >/dev/null 2>&1 || true
    curl -s --max-time 60 -X DELETE https://rest.runpod.io/v1/pods/$POD \
        -H "Authorization: Bearer $(cat "$HOME/.runpod-api-key")" >/dev/null 2>&1 || true
    log "pod $POD terminated"
}

log "=== resume watcher start pod=$POD ip=$IP port=$PORT"
while :; do
    done_=$(sshp 'grep -c "COMMUNITY_RESUME_DONE" /workspace/bench/community-resume.log 2>/dev/null' 2>/dev/null | tr -cd '0-9')
    srv_fail=$(sshp 'grep -ac "SERVER2_FAILED" /workspace/bench/community-resume.log 2>/dev/null' 2>/dev/null | tr -cd '0-9')
    if [ "${done_:-0}" -ge 1 ]; then log "RESUME_DONE seen"; break; fi
    if [ "${srv_fail:-0}" -ge 1 ]; then log "REMOTE FAILURE: SERVER2_FAILED"; break; fi
    if grep -q "deleting pod" "$JOBDIR/community-spend.log" 2>/dev/null; then
        log "guard deleted the pod; exiting poll loop"; exit 0
    fi
    tail1=$(sshp 'tail -1 /workspace/bench/community-resume.log 2>/dev/null' 2>/dev/null)
    [ -n "$tail1" ] && log "[remote] $tail1"
    sleep 120
done

log "fetching artifacts"
mkdir -p "$JOBDIR/bench/community"
rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
    "root@$IP:/workspace/bench/community/" "$JOBDIR/bench/community/" >/dev/null 2>&1 || log "WARN: community rsync failed"
for f in community-run.log community-resume.log server2.log; do
    rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
        "root@$IP:/workspace/bench/$f" "$JOBDIR/bench/" >/dev/null 2>&1 || true
done
rsync -a --partial --timeout=600 -e "$SSH_FETCH" \
    "root@$IP:/workspace/llama-qwen4exp/server2.log" "$JOBDIR/bench/" >/dev/null 2>&1 || true
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
log "=== resume watcher DONE ==="
