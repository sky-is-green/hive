#!/bin/bash
# pod-table-gate-book.sh - book the Q3X32 table-gate window (RTX A6000 by
# default), wait for the endpoint, launch pod-table-gate-run.sh as unit
# scion-tg-driver.  Override GPUS="NVIDIA L40S" for the fallback.
set -u
WORK=<work>
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
JOBDIR=${JOBDIR:-$Q/pod/remote-gate-table}
RECIPE=$Q/pod/remote-w1w2/recipe
GPUS=${GPUS:-"NVIDIA RTX A6000,NVIDIA L40S,NVIDIA L40"}
DRIVER=${DRIVER:-pod-table-gate-run.sh}
UNIT=${UNIT:-scion-tg-driver}
mkdir -p "$JOBDIR"
LOG=$JOBDIR/tg-book.log
exec >> "$LOG" 2>&1
log() { echo "$(date -u +%FT%TZ) $*"; }

log "=== table-gate book start gpus=$GPUS"
K=$(cat "$HOME/.runpod-api-key")
PUB=$(cat "$HOME/.ssh/id_runpod.pub")
PAYLOAD=$(mktemp)
GPUS="$GPUS" python3 - "$PUB" > "$PAYLOAD" <<'EOF'
import json, os, sys
pub = sys.argv[1].strip()
gpus = [g.strip() for g in os.environ.get("GPUS", "NVIDIA RTX A6000").split(",") if g.strip()]
p = {"name": "flashnext-table-gate",
     "imageName": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
     "cloudType": "SECURE", "computeType": "GPU", "gpuCount": 1,
     "gpuTypeIds": gpus,
     "gpuTypePriority": "custom",
     "dataCenterIds": ["US-TX-4","US-TX-3","US-TX-1","US-GA-2","US-GA-1","US-NC-1","US-IL-1","US-WA-1","US-CA-2",
                       "US-DE-1","US-MD-1","CA-MTL-1","CA-MTL-2","EU-RO-1","EU-SE-1","EU-CZ-1","EU-NL-1","EU-FR-1",
                       "EUR-IS-1","EUR-IS-2","EUR-IS-3","EUR-NO-1","AP-JP-1","AP-IN-1","CA-MTL-3","OC-AU-1"],
     "dataCenterPriority": "custom",
     "minRAMPerGPU": 96, "minVCPUPerGPU": 16,
     "containerDiskInGb": 200,
     "ports": ["22/tcp"], "interruptible": False,
     "env": {"SSH_PUBLIC_KEY": pub, "PUBLIC_KEY": pub}}
print(json.dumps(p))
EOF
RESP=$(curl -s --max-time 90 -X POST https://rest.runpod.io/v1/pods \
    -H "Authorization: Bearer $K" -H 'Content-Type: application/json' -d @"$PAYLOAD")
rm -f "$PAYLOAD"
echo "$RESP" > "$JOBDIR/booking-response.json"
PID=$(echo "$RESP" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("id",""))
except Exception: print("")')
if [ -z "$PID" ]; then
    log "no pod id: $(echo "$RESP" | head -c 300)"; exit 2
fi
RATE=$(echo "$RESP" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("costPerHr", 0))
except Exception: print(0)')
log "booked id=$PID rate=$RATE"
echo "$PID" > "$JOBDIR/pod-id"
echo "$RATE" > "$JOBDIR/pod-rate"

( sleep 43200
  curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$PID \
      -H "Authorization: Bearer $K" >/dev/null 2>&1
  echo "$(date -u +%FT%TZ) dead-man fired $PID" >> "$LOG" ) &
disown

for i in $(seq 1 75); do
    R=$(curl -s --max-time 20 -X POST https://api.runpod.io/graphql \
        -H "Content-Type: application/json" -H "Authorization: Bearer $K" \
        -d "{\"query\":\"query { pod(input: {podId: \\\"$PID\\\"}) { id runtime { ports { ip publicPort privatePort type } } } }\"}")
    EP=$(echo "$R" | python3 -c '
import json, sys
try:
    rt = json.load(sys.stdin)["data"]["pod"].get("runtime") or {}
    ssh = [q for q in (rt.get("ports") or []) if q.get("privatePort") == 22]
    if ssh: print(ssh[0]["ip"], ssh[0]["publicPort"])
except Exception: pass' 2>/dev/null)
    if [ -n "$EP" ]; then
        echo "$EP" > "$JOBDIR/pod-endpoint"
        log "ENDPOINT $EP after ~$((i*20))s"
        systemctl --user reset-failed "$UNIT" >/dev/null 2>&1 || true
        if systemd-run --user --unit="$UNIT" --collect \
            --setenv="JOBDIR=$JOBDIR" \
            --setenv="PAR=${PAR:-0}" \
            --property=StandardOutput=append:"$JOBDIR/tg-driver.log" \
            --property=StandardError=append:"$JOBDIR/tg-driver.log" \
            bash "$RECIPE/$DRIVER"; then
            log "table-gate driver launched as unit $UNIT"
        else
            log "systemd-run failed; setsid fallback"
            setsid nohup bash "$RECIPE/$DRIVER" \
                >> "$JOBDIR/tg-driver.log" 2>&1 < /dev/null &
            disown
        fi
        exit 0
    fi
    [ $((i % 6)) -eq 0 ] && log "waiting for endpoint... $((i*20))s"
    sleep 20
done
log "$PID stuck without endpoint after 25 min; deleting"
curl -s --max-time 60 -X DELETE https://rest.runpod.io/v1/pods/$PID \
    -H "Authorization: Bearer $K" >/dev/null
exit 2
