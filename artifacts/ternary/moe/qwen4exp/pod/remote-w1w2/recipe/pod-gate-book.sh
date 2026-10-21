#!/bin/bash
# pod-gate-book.sh - book the step-4000 gate-only pod and launch pod-gate-run.sh
# as the systemd user unit w4gate-driver.  Writes $JOBDIR state + dead-man 4.5 h.
set -u
WORK=<work>
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
JOBDIR=$Q/pod/remote-gate4000
RECIPE=$Q/pod/remote-w1w2/recipe
mkdir -p "$JOBDIR"
LOG=$JOBDIR/gate-book.log
exec >> "$LOG" 2>&1
log() { echo "$(date -u +%FT%TZ) $*"; }

log "=== pod-gate-book start"
K=$(cat "$HOME/.runpod-api-key")
PUB=$(cat "$HOME/.ssh/id_runpod.pub")
PAYLOAD=$(mktemp)
python3 - "$PUB" > "$PAYLOAD" <<'EOF'
import json, sys
pub = sys.argv[1].strip()
p = {"name": "flashnext-w4-gate",
     "imageName": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
     "cloudType": "SECURE", "computeType": "GPU", "gpuCount": 1,
     "gpuTypeIds": ["NVIDIA RTX 6000 Ada Generation", "NVIDIA L40", "NVIDIA L40S"],
     "gpuTypePriority": "custom",
     "dataCenterIds": ["US-KS-2", "US-TX-4", "US-GA-2"],
     "dataCenterPriority": "custom",
     "minRAMPerGPU": 180, "minVCPUPerGPU": 16,
     "containerDiskInGb": 100, "volumeInGb": 300, "volumeMountPath": "/workspace",
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

# 4.5 h dead-man backstop (guard cap 3.2 is the primary stop)
( sleep 16200
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
        echo "$PID" > /tmp/opencode/pod-id
        echo "$EP" > /tmp/opencode/pod-endpoint
        log "ENDPOINT $EP after ~$((i*20))s"
        systemctl --user reset-failed w4gate-driver >/dev/null 2>&1 || true
        if systemd-run --user --unit=w4gate-driver --collect \
            --property=StandardOutput=append:"$JOBDIR/gate-driver.log" \
            --property=StandardError=append:"$JOBDIR/gate-driver.log" \
            bash "$RECIPE/pod-gate-run.sh"; then
            log "gate driver launched as unit w4gate-driver"
        else
            log "systemd-run failed; setsid fallback"
            setsid nohup bash "$RECIPE/pod-gate-run.sh" \
                >> "$JOBDIR/gate-driver.log" 2>&1 < /dev/null &
            disown
        fi
        exit 0
    fi
    sleep 20
done
log "$PID stuck without endpoint after 25 min; deleting"
curl -s --max-time 60 -X DELETE https://rest.runpod.io/v1/pods/$PID \
    -H "Authorization: Bearer $K" >/dev/null
exit 2
