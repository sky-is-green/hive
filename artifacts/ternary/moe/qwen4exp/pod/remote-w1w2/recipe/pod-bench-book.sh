#!/bin/bash
# pod-bench-book.sh - book the A100 80GB bench pod (full-GPU), wait for the
# endpoint, launch pod-bench-run.sh as the systemd user unit w4bench-driver.
set -u
WORK=/home/penis/Desktop/work
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
JOBDIR=$Q/pod/remote-bench
RECIPE=$Q/pod/remote-w1w2/recipe
mkdir -p "$JOBDIR"
LOG=$JOBDIR/bench-book.log
exec >> "$LOG" 2>&1
log() { echo "$(date -u +%FT%TZ) $*"; }

log "=== pod-bench-book start"
K=$(cat "$HOME/.runpod-api-key")
PUB=$(cat "$HOME/.ssh/id_runpod.pub")
PAYLOAD=$(mktemp)
python3 - "$PUB" > "$PAYLOAD" <<'EOF'
import json, sys
pub = sys.argv[1].strip()
p = {"name": "flashnext-bench",
     "imageName": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
     "cloudType": "SECURE", "computeType": "GPU", "gpuCount": 1,
     "gpuTypeIds": ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe",
                    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                    "NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3"],
     "gpuTypePriority": "custom",
     "dataCenterIds": [
                        "US-TX-4",
                        "US-TX-3",
                        "US-TX-1",
                        "US-GA-2",
                        "US-GA-1",
                        "US-NC-1",
                        "US-IL-1",
                        "US-WA-1",
                        "US-CA-2",
                        "US-DE-1",
                        "US-MD-1",
                        "CA-MTL-1",
                        "CA-MTL-2",
                        "EU-RO-1",
                        "EU-SE-1",
                        "EU-CZ-1",
                        "EU-NL-1",
                        "EU-FR-1",
                        "EUR-IS-1",
                        "EUR-IS-2",
                        "EUR-IS-3",
                        "EUR-NO-1",
                        "AP-JP-1",
                        "AP-IN-1",
                        "CA-MTL-3",
                        "OC-AU-1"],
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

# 6 h dead-man backstop (spend guard cap 7.5 is the primary stop)
( sleep 21600
  curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$PID \
      -H "Authorization: Bearer $K" >/dev/null 2>&1
  echo "$(date -u +%FT%TZ) dead-man fired $PID" >> "$LOG" ) &
disown

for i in $(seq 1 15); do
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
        systemctl --user reset-failed w4bench-driver >/dev/null 2>&1 || true
        if systemd-run --user --unit=w4bench-driver --collect \
            --property=StandardOutput=append:"$JOBDIR/bench-driver.log" \
            --property=StandardError=append:"$JOBDIR/bench-driver.log" \
            bash "$RECIPE/pod-bench-run.sh"; then
            log "bench driver launched as unit w4bench-driver"
        else
            log "systemd-run failed; setsid fallback"
            setsid nohup bash "$RECIPE/pod-bench-run.sh" \
                >> "$JOBDIR/bench-driver.log" 2>&1 < /dev/null &
            disown
        fi
        exit 0
    fi
    sleep 20
done
log "$PID stuck without endpoint after 5 min; deleting"
curl -s --max-time 60 -X DELETE https://rest.runpod.io/v1/pods/$PID \
    -H "Authorization: Bearer $K" >/dev/null
exit 2
