#!/bin/bash
# pod-one.sh - single targeted booking (US-KS-2 first), 25-min endpoint wait.
# On success writes /tmp/opencode/pod-id and pod-endpoint and runs pod-cap-run2.sh.
set -u
K=$(cat $HOME/.runpod-api-key)
PUB=$(cat $HOME/.ssh/id_runpod.pub)
PAYLOAD=$(mktemp)
python3 - "$PUB" > "$PAYLOAD" <<'EOF'
import json,sys
pub=sys.argv[1].strip()
p={"name":"flashnext-ple-cap",
   "imageName":"runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
   "cloudType":"SECURE","computeType":"GPU","gpuCount":1,
   "gpuTypeIds":["NVIDIA RTX 6000 Ada Generation","NVIDIA L40","NVIDIA L40S"],
   "gpuTypePriority":"custom",
   "dataCenterIds":["US-KS-2","US-TX-4","US-GA-2"],
   "dataCenterPriority":"custom",
   "minRAMPerGPU":180,"minVCPUPerGPU":16,
   "containerDiskInGb":100,"volumeInGb":100,"volumeMountPath":"/workspace",
   "ports":["22/tcp"],"interruptible":False,
   "env":{"SSH_PUBLIC_KEY":pub,"PUBLIC_KEY":pub}}
print(json.dumps(p))
EOF
RESP=$(curl -s --max-time 90 -X POST https://rest.runpod.io/v1/pods -H "Authorization: Bearer $K" -H 'Content-Type: application/json' -d @"$PAYLOAD")
rm -f "$PAYLOAD"
PID=$(echo "$RESP" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("id",""))
except: print("")')
if [ -z "$PID" ]; then echo "$(date -u +%H:%M:%SZ) no pod id: $(echo "$RESP" | head -c 250)"; exit 2; fi
echo "$(date -u +%H:%M:%SZ) booked id=$PID"
for i in $(seq 1 75); do
  R=$(curl -s --max-time 20 -X POST https://api.runpod.io/graphql -H "Content-Type: application/json" -H "Authorization: Bearer $K" -d "{\"query\":\"query { pod(input: {podId: \\\"$PID\\\"}) { id runtime { ports { ip publicPort privatePort type } } } }\"}")
  # NOTE: plain indexing here; escaped-quote f-strings silently fail under `except: pass`
  EP=$(echo "$R" | python3 -c '
import json,sys
try:
    rt=json.load(sys.stdin)["data"]["pod"].get("runtime") or {}
    ssh=[q for q in (rt.get("ports") or []) if q.get("privatePort")==22]
    if ssh: print(ssh[0]["ip"], ssh[0]["publicPort"])
except Exception: pass' 2>/dev/null)
  if [ -n "$EP" ]; then
    echo "$PID" > /tmp/opencode/pod-id
    echo "$EP" > /tmp/opencode/pod-endpoint
    echo "$(date -u +%H:%M:%SZ) ENDPOINT $EP after ~$((i*20))s"
    ( sleep 12600; curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$PID -H "Authorization: Bearer $K" >/dev/null 2>&1; echo "dead-man $PID fired" ) &
    bash "$(dirname "$0")/pod-cap-run2.sh"
    exit 0
  fi
  sleep 20
done
echo "$(date -u +%H:%M:%SZ) $PID stuck after 25 min; deleting"
curl -s --max-time 60 -X DELETE https://rest.runpod.io/v1/pods/$PID -H "Authorization: Bearer $K" >/dev/null
exit 2
