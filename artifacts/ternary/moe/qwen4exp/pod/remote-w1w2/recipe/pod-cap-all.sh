#!/bin/bash
# pod-cap-all.sh - self-contained: book -> endpoint -> build -> upload -> cap_eval -> fetch -> delete.
set -u
K=$(cat $HOME/.runpod-api-key)
PUB=$(cat $HOME/.ssh/id_runpod.pub)
KEY=$HOME/.ssh/id_runpod
Q=<work>/hivebench/artifacts/ternary/moe/qwen4exp
SRC=/tmp/opencode/llama-qwen4exp-src.tar.gz
GGUF=$Q/qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf
EXPECT_SHA=2f4bd083861b027ede8b81eacf5f8b808f04d3c72e612c42d54758a473aa22ba

PAYLOAD=$(mktemp)
python3 - "$PUB" > "$PAYLOAD" <<'EOF'
import json,sys
pub=sys.argv[1].strip()
p={"name":"flashnext-ple-cap5",
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
if [ -z "$PID" ]; then echo "$(date -u +%H:%M:%SZ) BOOK FAILED: $(echo "$RESP" | head -c 200)"; exit 2; fi
echo "$(date -u +%H:%M:%SZ) booked id=$PID $(echo "$RESP" | python3 -c 'import json,sys
try: d=json.load(sys.stdin); print("mem=%s dc=%s"%(d.get("memoryInGb"), (d.get("machine") or {}).get("dataCenterId")))
except: pass')"
IP=""; PORT=""
for i in $(seq 1 75); do
  R=$(curl -s --max-time 20 -X POST https://api.runpod.io/graphql -H "Content-Type: application/json" -H "Authorization: Bearer $K" -d "{\"query\":\"query { pod(input: {podId: \\\"$PID\\\"}) { id runtime { ports { ip publicPort privatePort type } } } }\"}")
  read -r IP PORT <<<"$(echo "$R" | python3 -c '
import json,sys
try:
    rt=json.load(sys.stdin)["data"]["pod"].get("runtime") or {}
    ssh=[q for q in (rt.get("ports") or []) if q.get("privatePort")==22]
    if ssh: print(ssh[0]["ip"], ssh[0]["publicPort"])
except Exception: pass' 2>/dev/null)"
  if [ -n "${IP:-}" ] && [ -n "${PORT:-}" ]; then echo "$(date -u +%H:%M:%SZ) endpoint $IP:$PORT after ~$((i*20))s"; break; fi
  sleep 20
done
if [ -z "${IP:-}" ]; then
  echo "$(date -u +%H:%M:%SZ) $PID stuck 25 min; deleting"
  curl -s --max-time 60 -X DELETE https://rest.runpod.io/v1/pods/$PID -H "Authorization: Bearer $K" >/dev/null
  exit 2
fi
( sleep 12600; curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$PID -H "Authorization: Bearer $K" >/dev/null 2>&1; echo "dead-man fired for $PID at $(date -u +%H:%M:%SZ)" ) &
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -p $PORT root@$IP"
SCP="scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -P $PORT"
die_keep()   { echo "=== FAILURE(build): $1 $(date -u +%H:%M:%SZ)"; $SSH 'tail -60 /root/build.log 2>/dev/null'; echo "pod $PID kept for debug; dead-man will delete"; exit 1; }
die_delete() { echo "=== FAILURE(eval): $1 $(date -u +%H:%M:%SZ)"; $SSH 'tail -40 /root/build.log; tail -40 /root/cap-eval-48l-ple-server.log 2>/dev/null'; echo "deleting pod $PID"; curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$PID -H "Authorization: Bearer $K"; echo; exit 1; }

for t in 1 2 3 4 5 6; do $SSH 'echo SSH_OK' 2>/dev/null | grep -q SSH_OK && { echo "ssh ok (try $t)"; break; }; sleep 10; done
$SSH 'nproc; free -g | head -2; df -h / /workspace 2>/dev/null | tail -2'
$SSH 'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq rsync ninja-build cmake build-essential >/dev/null 2>&1; which rsync ninja cmake g++'
FREE_WS=$($SSH "df -Pk /workspace 2>/dev/null | awk 'NR==2{print \$4}'" || echo 0)
if [ "${FREE_WS:-0}" -gt 70000000 ]; then MODELDIR=/workspace/models; else MODELDIR=/root/models; fi
echo "MODELDIR=$MODELDIR"
$SSH "mkdir -p $MODELDIR"
$SCP "$SRC" root@$IP:/root/ || die_keep "source upload"
$SCP <work>/scion/moe/cap_eval.py <work>/scion/moe/cap_tasks.json root@$IP:/root/ || die_keep "cap_eval upload"
$SSH 'cd /root && tar -xzf llama-qwen4exp-src.tar.gz && ls llama-qwen4exp/common/build-info.cpp.in llama-qwen4exp/cmake/build-info.cmake llama-qwen4exp/src/models/qwen4exp.cpp' || die_keep "extract"
$SSH 'rm -rf /root/llama-qwen4exp/build'
echo "=== building (CPU-only) $(date -u +%H:%M:%SZ)"
$SSH 'cd /root/llama-qwen4exp && (nohup bash -c "cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && cmake --build build -j\$(nproc) --target llama-server" > /root/build.log 2>&1 &); echo build-started'
for i in $(seq 1 120); do
  if $SSH 'test -x /root/llama-qwen4exp/build/bin/llama-server'; then echo "build OK after ~$((i*20))s"; break; fi
  if [ "$i" -ge 3 ] && ! $SSH 'pgrep -f "cmake --build" >/dev/null || pgrep -x ninja >/dev/null || pgrep -x cc1plus >/dev/null || pgrep -x g++ >/dev/null || pgrep -f "cmake -B" >/dev/null'; then
    die_keep "build process ended without a binary"
  fi
  sleep 20
done
$SSH 'test -x /root/llama-qwen4exp/build/bin/llama-server' || die_keep "no llama-server after wait"
$SSH '/root/llama-qwen4exp/build/bin/llama-server --version | head -1'
echo "=== uploading model (~66 GB) $(date -u +%H:%M:%SZ)"
rsync -a --partial --info=progress2 -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -p $PORT" "$GGUF" root@$IP:$MODELDIR/ || die_delete "rsync"
RSHA=$($SSH "sha256sum $MODELDIR/qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf" | awk '{print $1}')
echo "=== remote sha: $RSHA"
[ "$RSHA" = "$EXPECT_SHA" ] || die_delete "sha mismatch"
echo "=== cap_eval (CPU, -t 16) $(date -u +%H:%M:%SZ)"
timeout 7200 $SSH "cd /root && python3 cap_eval.py --model $MODELDIR/qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf --backend server --server-bin /root/llama-qwen4exp/build/bin/llama-server --server-args \"-c 512 -t 16\" --server-log /root/cap-eval-48l-ple-server.log --out /root/cap-eval-48l-ple.json --timeout 1800 2>&1 | tail -30"
$SCP root@$IP:/root/cap-eval-48l-ple.json $Q/cap-eval-48l-ple.json && echo "fetched JSON"
$SCP root@$IP:/root/cap-eval-48l-ple-server.log $Q/cap-eval-48l-ple-server.log && echo "fetched server log"
python3 - "$Q/cap-eval-48l-ple.json" <<'EOF' || echo "JSON missing/invalid"
import json,sys
d=json.load(open(sys.argv[1]))
print("RESULT: ok %d / %d = %.3f" % (d["ok"], d["n"], d["accuracy"]))
print("harness errors:", sum(1 for r in d["results"] if r.get("error")))
for c,v in d["by_category"].items(): print(" ", c, v["ok"], "/", v["n"])
EOF
echo "=== terminating pod $PID"
curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$PID -H "Authorization: Bearer $K"; echo
echo "=== pod-cap-all end $(date -u +%H:%M:%SZ)"
