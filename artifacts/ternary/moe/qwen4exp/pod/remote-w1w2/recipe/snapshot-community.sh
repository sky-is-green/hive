#!/bin/bash
# snapshot-community.sh - fetch the pod's community artifacts every 4 minutes so a provider-side pod
# loss cannot take the samples with it (the 2026-10-09 16:15 loss did exactly that).
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
WORK=<work>
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp/pod
JOBDIR=${SNAP_JOBDIR:-$Q/remote-community}
mkdir -p "$JOBDIR/snapshots"
while :; do
    POD=$(cat "$JOBDIR/pod-id" 2>/dev/null)
    if [ -z "$POD" ]; then sleep 60; continue; fi
    # a pod id in the file that no longer exists: stop
    if ! curl -s --max-time 20 "${RUNPOD_URL:-https://rest.runpod.io/v1/pods}" \
         -H "Authorization: Bearer $(cat "$HOME/.runpod-api-key")" | grep -q "\"$POD\""; then
        echo "$(date -u +%FT%TZ) pod $POD is gone; snapshotter stops" >> "$JOBDIR/snapshot.log"
        exit 0
    fi
    if [ -f "$JOBDIR/pod-endpoint" ]; then
        read -r IP PORT < "$JOBDIR/pod-endpoint" || { sleep 60; continue; }
        rsync -a --partial --timeout=180 \
            -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $HOME/.ssh/id_runpod -p $PORT" \
            "root@$IP:/workspace/bench/community/" "$JOBDIR/snapshots/" \
            >> "$JOBDIR/snapshot.log" 2>&1 || true
        echo "$(date -u +%FT%TZ) snapshot attempt ($(du -sh "$JOBDIR/snapshots" 2>/dev/null | cut -f1))" >> "$JOBDIR/snapshot.log"
    fi
    sleep 240
done
