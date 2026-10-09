#!/bin/bash
# pod-a6000-book.sh - book the cheapest fitting card (RTX A6000 48GB, $0.33/h), wait for the endpoint, launch
# the probe driver. The pod is NOT terminated by the driver: it stays for the full run if the probe passes.
set -u
WORK=/home/penis/Desktop/work
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
JOBDIR=$Q/pod/remote-a6000
RECIPE=$Q/pod/remote-w1w2/recipe
mkdir -p "$JOBDIR"
LOG=$JOBDIR/a6000-book.log
exec >> "$LOG" 2>&1
log() { echo "$(date -u +%FT%TZ) $*"; }

log "=== a6000 book start"
K=$(cat "$HOME/.runpod-api-key")
PUB=$(cat "$HOME/.ssh/id_runpod.pub")
PAYLOAD=$(mktemp)
python3 - "$PUB" > "$PAYLOAD" <<'EOF'
import json, sys
pub = sys.argv[1].strip()
p = {"name": "flashnext-a6000",
     "imageName": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
     "cloudType": "SECURE", "computeType": "GPU", "gpuCount": 1,
     "gpuTypeIds": ["NVIDIA RTX A6000"],
     "gpuTypePriority": "custom",
     "dataCenterIds": ["US-TX-4","US-TX-3","US-TX-1","US-GA-2","US-GA-1","US-NC-1","US-IL-1","US-WA-1","US-CA-2",
                       "US-DE-1","US-MD-1","CA-MTL-1","CA-MTL-2","EU-RO-1","EU-SE-1","EU-CZ-1","EU-NL-1","EU-FR-1",
                       "EUR-IS-1","EUR-IS-2","EUR-IS-3","EUR-NO-1","AP-JP-1","AP-IN-1","CA-MTL-3","OC-AU-1"],
     "dataCenterPriority": "custom",
     "minRAMPerGPU": 48, "minVCPUPerGPU": 8,
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
if [ -z "$PID" ]; then log "no pod id: $(echo "$RESP" | head -c 300)"; exit 2; fi
RATE=$(echo "$RESP" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("costPerHr", 0))
except Exception: print(0)')
log "booked id=$PID rate=$RATE"
echo "$PID" > "$JOBDIR/pod-id"
echo "$RATE" > "$JOBDIR/pod-rate"

( sleep 28800
  curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$PID \
      -H "Authorization: Bearer $K" >/dev/null 2>&1
  echo "$(date -u +%FT%TZ) dead-man fired $PID" >> "$LOG" ) &
disown

for i in $(seq 1 60); do
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
        systemctl --user reset-failed scion-a6000-driver >/dev/null 2>&1 || true
        systemd-run --user --unit=scion-a6000-driver --collect \
            --property=StandardOutput=append:"$JOBDIR/a6000-driver.log" \
            --property=StandardError=append:"$JOBDIR/a6000-driver.log" \
            bash "$RECIPE/pod-a6000-driver.sh" \
            && log "probe driver launched as unit scion-a6000-driver" \
            || log "systemd-run failed"
        exit 0
    fi
    [ $((i % 6)) -eq 0 ] && log "waiting for endpoint... $((i*20))s"
    sleep 20
done
log "$PID stuck without endpoint after 20 min; deleting"
curl -s --max-time 60 -X DELETE https://rest.runpod.io/v1/pods/$PID -H "Authorization: Bearer $K" >/dev/null
exit 2
