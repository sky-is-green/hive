#!/bin/bash
# pod-w4-book.sh - W4 booking: SECURE [RTX 6000 Ada, L40, L40S], pinned
# [US-KS-2, US-TX-4, US-GA-2], volume 300GB, then wait <=25 min for the SSH
# endpoint and launch pod-w4-driver.sh as a systemd user unit (w4-driver).
#
# Writes $W4DIR/{pod-id,pod-rate,pod-endpoint,booking-response.json} and
# logs to $W4DIR/w4-book.log.  A 13 h dead-man (disowned) is the absolute
# backstop; the spend guard (cap $9) is the primary stop.
#
# Run detached:
#   setsid nohup bash pod-w4-book.sh >/dev/null 2>&1 & disown
set -u
WORK=/home/penis/Desktop/work
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
W4DIR=$Q/pod/remote-w4
RECIPE=$Q/pod/remote-w1w2/recipe
mkdir -p "$W4DIR"
LOG=$W4DIR/w4-book.log
exec >> "$LOG" 2>&1
log() { echo "$(date -u +%FT%TZ) $*"; }

log "=== pod-w4-book start"
K=$(cat "$HOME/.runpod-api-key")
PUB=$(cat "$HOME/.ssh/id_runpod.pub")
PAYLOAD=$(mktemp)
python3 - "$PUB" > "$PAYLOAD" <<'EOF'
import json, sys
pub = sys.argv[1].strip()
p = {"name": "flashnext-w4",
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
echo "$RESP" > "$W4DIR/booking-response.json"
PID=$(echo "$RESP" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("id",""))
except Exception: print("")')
if [ -z "$PID" ]; then
    log "no pod id: $(echo "$RESP" | head -c 300)"
    exit 2
fi
RATE=$(echo "$RESP" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("costPerHr", 0))
except Exception: print(0)')
log "booked id=$PID rate=$RATE"
echo "$PID" > "$W4DIR/pod-id"
echo "$RATE" > "$W4DIR/pod-rate"

# 13 h dead-man backstop (the spend guard cap is the real stop)
( sleep 46800
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
        echo "$EP" > "$W4DIR/pod-endpoint"
        echo "$PID" > /tmp/opencode/pod-id
        echo "$EP" > /tmp/opencode/pod-endpoint
        log "ENDPOINT $EP after ~$((i*20))s"
        systemctl --user reset-failed w4-driver >/dev/null 2>&1 || true
        if systemd-run --user --unit=w4-driver --collect \
            --property=StandardOutput=append:"$W4DIR/w4-driver.log" \
            --property=StandardError=append:"$W4DIR/w4-driver.log" \
            bash "$RECIPE/pod-w4-driver.sh"; then
            log "driver launched as unit w4-driver"
        else
            log "systemd-run failed; setsid fallback"
            setsid nohup bash "$RECIPE/pod-w4-driver.sh" \
                >> "$W4DIR/w4-driver.log" 2>&1 < /dev/null &
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
