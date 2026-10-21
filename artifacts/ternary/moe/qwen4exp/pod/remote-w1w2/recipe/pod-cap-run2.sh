#!/bin/bash
# pod-cap-run2.sh - CPU-only variant (no nvcc/CUDA dependency).
# Build FIRST and verify the binary, THEN upload the model, then cap_eval.
set -u
IP=$(awk '{print $1}' /tmp/opencode/pod-endpoint)
PORT=$(awk '{print $2}' /tmp/opencode/pod-endpoint)
KEY=$HOME/.ssh/id_runpod
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -p $PORT root@$IP"
SCP="scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -P $PORT"
Q=<work>/hivebench/artifacts/ternary/moe/qwen4exp
SRC=/tmp/opencode/llama-qwen4exp-src.tar.gz
GGUF=$Q/qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf
POD=$(cat /tmp/opencode/pod-id)
die() { echo "=== FAILURE: $1 $(date -u +%H:%M:%SZ)"; $SSH 'tail -50 /root/build.log 2>/dev/null; tail -30 /root/cap-eval-48l-ple-server.log 2>/dev/null'; echo "=== deleting pod $POD"; curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$POD -H "Authorization: Bearer $(cat $HOME/.runpod-api-key)"; echo; exit 1; }
echo "=== pod-cap-run2 start $(date -u +%H:%M:%SZ) ip=$IP port=$PORT pod=$POD"
$SSH 'nproc; free -g | head -2; df -h / /workspace 2>/dev/null | tail -2'
$SSH 'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq rsync ninja-build cmake build-essential >/dev/null 2>&1; which rsync ninja cmake g++'
FREE_WS=$($SSH "df -Pk /workspace 2>/dev/null | awk 'NR==2{print \$4}'" || echo 0)
if [ "${FREE_WS:-0}" -gt 70000000 ]; then MODELDIR=/workspace/models; else MODELDIR=/root/models; fi
echo "MODELDIR=$MODELDIR (workspace free ${FREE_WS}kB)"
$SSH "mkdir -p $MODELDIR"
$SCP "$SRC" root@$IP:/root/
$SCP <work>/scion/moe/cap_eval.py <work>/scion/moe/cap_tasks.json root@$IP:/root/
$SSH 'cd /root && tar -xzf llama-qwen4exp-src.tar.gz && ls llama-qwen4exp/src/models/qwen4exp.cpp llama-qwen4exp/common/build-info.cpp.in'
$SSH 'rm -rf /root/llama-qwen4exp/build'
echo "=== building (CPU-only) $(date -u +%H:%M:%SZ)"
$SSH 'cd /root/llama-qwen4exp && (nohup bash -c "cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && cmake --build build -j\$(nproc) --target llama-server" > /root/build.log 2>&1 &); echo build-started'
for i in $(seq 1 90); do
  if $SSH 'test -x /root/llama-qwen4exp/build/bin/llama-server'; then echo "build OK after ~$((i*20))s"; break; fi
  if [ "$i" -ge 3 ] && ! $SSH 'pgrep -f "cmake --build" >/dev/null || pgrep -x ninja >/dev/null || pgrep -x cc1plus >/dev/null || pgrep -x g++ >/dev/null || pgrep -f "cmake -B" >/dev/null'; then
    die "build process ended without a binary"
  fi
  sleep 20
done
$SSH 'test -x /root/llama-qwen4exp/build/bin/llama-server' || die "no llama-server after wait"
$SSH 'ls -la /root/llama-qwen4exp/build/bin/llama-server; /root/llama-qwen4exp/build/bin/llama-server --version | head -1'
echo "=== uploading model (~66 GB) $(date -u +%H:%M:%SZ)"
rsync -a --partial --info=progress2 -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -p $PORT" "$GGUF" root@$IP:$MODELDIR/
echo "=== upload done $(date -u +%H:%M:%SZ); remote sha:"
$SSH "sha256sum $MODELDIR/qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf"
echo "=== cap_eval (CPU, -t 16) $(date -u +%H:%M:%SZ)"
timeout 7200 $SSH "cd /root && python3 cap_eval.py --model $MODELDIR/qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf --backend server --server-bin /root/llama-qwen4exp/build/bin/llama-server --server-args \"-c 512 -t 16\" --server-log /root/cap-eval-48l-ple-server.log --out /root/cap-eval-48l-ple.json --timeout 1800 2>&1 | tail -30"
$SSH 'echo "--- server log tail:"; tail -20 /root/cap-eval-48l-ple-server.log 2>/dev/null'
$SCP root@$IP:/root/cap-eval-48l-ple.json $Q/cap-eval-48l-ple.json && echo "fetched JSON"
$SCP root@$IP:/root/cap-eval-48l-ple-server.log $Q/cap-eval-48l-ple-server.log && echo "fetched server log"
echo "=== local report head:"; python3 -c 'import json; d=json.load(open("'$Q'/cap-eval-48l-ple.json")); print("ok", d["ok"], "/", d["n"], "acc", d["accuracy"]); print("harness errors:", sum(1 for r in d["results"] if r.get("error")))' 2>/dev/null || echo "JSON missing/invalid"
echo "=== terminating pod $POD"
curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$POD -H "Authorization: Bearer $(cat $HOME/.runpod-api-key)"; echo
echo "=== pod-cap-run2 end $(date -u +%H:%M:%SZ)"
