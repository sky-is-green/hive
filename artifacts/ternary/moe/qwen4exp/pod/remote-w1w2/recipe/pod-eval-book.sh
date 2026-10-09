#!/bin/bash
# pod-eval-book.sh - book one eval pod (build-first, parallel chunk upload,
# cap_eval via the server backend), wait for the SSH endpoint, launch
# pod-eval-run.sh as a systemd user unit (w4eval-driver).
#
# Env:
#   EVAL_GGUF  local GGUF to evaluate (default: 48l step4096 PLE f16)
#   EVAL_NAME  remote filename (default: basename)
#   EVAL_JOBDIR local state dir (default: $Q/pod/remote-eval)
#   EVAL_LOG   cap_eval log basename (default: cap-eval-48l-step4096-ple)
# Run detached:
#   setsid nohup bash pod-eval-book.sh >/dev/null 2>&1 & disown
set -u
WORK=/home/penis/Desktop/work
Q=$WORK/hivebench/artifacts/ternary/moe/qwen4exp
RECIPE=$Q/pod/remote-w1w2/recipe
EVAL_GGUF="${EVAL_GGUF:-$Q/qwen4exp-48l-ptq1_0-corr-step4096-ple.gguf}"
EVAL_JOBDIR="${EVAL_JOBDIR:-$Q/pod/remote-eval}"
EVAL_NAME="${EVAL_NAME:-$(basename "$EVAL_GGUF")}"
EVAL_LOG="${EVAL_LOG:-cap-eval-48l-step4096-ple}"
mkdir -p "$EVAL_JOBDIR"
LOG=$EVAL_JOBDIR/eval-book.log
exec >> "$LOG" 2>&1
log() { echo "$(date -u +%FT%TZ) $*"; }

[ -s "$EVAL_GGUF" ] || { log "REFUSING: $EVAL_GGUF missing"; exit 2; }
EVAL_SHA=$(sha256sum "$EVAL_GGUF" | awk '{print $1}')
echo "$EVAL_GGUF" > "$EVAL_JOBDIR/eval-gguf"
echo "$EVAL_NAME" > "$EVAL_JOBDIR/eval-name"
echo "$EVAL_LOG" > "$EVAL_JOBDIR/eval-log"
echo "$EVAL_SHA" > "$EVAL_JOBDIR/eval-sha"
log "eval pod for $EVAL_GGUF sha=${EVAL_SHA:0:12}"

K=$(cat "$HOME/.runpod-api-key")
PUB=$(cat "$HOME/.ssh/id_runpod.pub")
PAYLOAD=$(mktemp)
python3 - "$PUB" > "$PAYLOAD" <<'EOF'
import json, sys
pub = sys.argv[1].strip()
p = {"name": "flashnext-w4-eval",
     "imageName": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
     "cloudType": "SECURE", "computeType": "GPU", "gpuCount": 1,
     "gpuTypeIds": ["NVIDIA RTX 6000 Ada Generation", "NVIDIA L40", "NVIDIA L40S"],
     "gpuTypePriority": "custom",
     "dataCenterIds": ["US-KS-2", "US-TX-4", "US-GA-2"],
     "dataCenterPriority": "custom",
     "minRAMPerGPU": 180, "minVCPUPerGPU": 16,
     "containerDiskInGb": 100, "volumeInGb": 100, "volumeMountPath": "/workspace",
     "ports": ["22/tcp"], "interruptible": False,
     "env": {"SSH_PUBLIC_KEY": pub, "PUBLIC_KEY": pub}}
print(json.dumps(p))
EOF
RESP=$(curl -s --max-time 90 -X POST https://rest.runpod.io/v1/pods \
    -H "Authorization: Bearer $K" -H 'Content-Type: application/json' -d @"$PAYLOAD")
rm -f "$PAYLOAD"
echo "$RESP" > "$EVAL_JOBDIR/booking-response.json"
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
echo "$PID" > "$EVAL_JOBDIR/pod-id"
echo "$RATE" > "$EVAL_JOBDIR/pod-rate"

# 4 h dead-man backstop (build 10m + upload 25m + eval 60m + margin)
( sleep 14400
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
        echo "$EP" > "$EVAL_JOBDIR/pod-endpoint"
        echo "$PID" > /tmp/opencode/pod-id
        echo "$EP" > /tmp/opencode/pod-endpoint
        log "ENDPOINT $EP after ~$((i*20))s"
        systemctl --user reset-failed w4eval-driver >/dev/null 2>&1 || true
        if systemd-run --user --unit=w4eval-driver --collect \
            --property=StandardOutput=append:"$EVAL_JOBDIR/eval-driver.log" \
            --property=StandardError=append:"$EVAL_JOBDIR/eval-driver.log" \
            --setenv=EVAL_JOBDIR="$EVAL_JOBDIR" \
            --setenv=EVAL_STAGE="${EVAL_STAGE:-/run/media/penis/30CE2C97CE2C577E/storage/evalstage}" \
            bash "$RECIPE/pod-eval-run.sh"; then
            log "eval driver launched as unit w4eval-driver"
        else
            log "systemd-run failed; setsid fallback"
            setsid nohup bash "$RECIPE/pod-eval-run.sh" \
                >> "$EVAL_JOBDIR/eval-driver.log" 2>&1 < /dev/null &
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
