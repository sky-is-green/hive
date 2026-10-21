#!/bin/bash
# pod-parallel-finish.sh - finish the upload with parallel chunked streams,
# then sha-verify, cap_eval, fetch, delete. Replaces the single-stream rsync.
set -u
IP=195.26.232.178; PORT=27379
KEY=$HOME/.ssh/id_runpod
Q=<work>/hivebench/artifacts/ternary/moe/qwen4exp
GGUF=$Q/qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf
EXPECT=2f4bd083861b027ede8b81eacf5f8b808f04d3c72e612c42d54758a473aa22ba
POD=ycsups8e119kej
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -p $PORT root@$IP"
# fresh dead-man (3.5 h)
( sleep 12600; curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$POD -H "Authorization: Bearer $(cat $HOME/.runpod-api-key)" >/dev/null 2>&1; echo "dead-man fired $POD" ) &

# 1. stop the old driver and its rsync (careful: driver must not see rsync fail)
DRV=$(pgrep -f "bash /tmp/opencode/pod-cap-all.sh" | head -1 || true)
[ -n "${DRV:-}" ] && kill "$DRV" 2>/dev/null && echo "killed driver $DRV"
sleep 1
RS=$(pgrep -f "rsync -a --partial" | head -1 || true)
[ -n "${RS:-}" ] && kill -9 "$RS" 2>/dev/null && echo "killed rsync $RS"
sleep 2
pgrep -f "rsync -a --partial" >/dev/null && echo "WARN: rsync still running"

# 2. partial size on the pod
PART=$($SSH "ls /workspace/models/.qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf.* 2>/dev/null | head -1")
[ -n "$PART" ] || { echo "no partial file found"; exit 1; }
P=$($SSH "stat -c %s '$PART'")
echo "partial $PART size=$P ($(python3 -c "print(f'{$P/1e9:.2f} GB')"))"

# 3. local remainder + split into 5 GB chunks
tail -c +$((P+1)) "$GGUF" > /tmp/opencode/rem.bin || { echo "tail failed"; exit 1; }
REMSZ=$(stat -c %s /tmp/opencode/rem.bin)
echo "remainder $(python3 -c "print(f'{$REMSZ/1e9:.2f} GB')")"
split -b 5G -d --suffix-length=1 /tmp/opencode/rem.bin /tmp/opencode/rem.part
ls /tmp/opencode/rem.part* | wc -l | xargs echo "chunks:"

# 4. parallel upload (all chunks at once)
T0=$(date +%s)
for f in /tmp/opencode/rem.part*; do
  scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o Compression=no -i "$KEY" -P "$PORT" "$f" root@$IP:/workspace/models/ &
done
wait
T1=$(date +%s)
python3 -c "dt=$T1-$T0; print(f'parallel upload: {$REMSZ/1e6/dt:.1f} MB/s ({$REMSZ/1e6/dt*8:.0f} Mbps) in {dt}s')"
rm -f /tmp/opencode/rem.bin /tmp/opencode/rem.part*

# 5. concat + sha verify
$SSH "cd /workspace/models && cat '$PART' \$(ls rem.part* | sort) > qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf && rm -f '$PART' rem.part* && ls -la qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf && sha256sum qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf"
RSHA=$($SSH "sha256sum /workspace/models/qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf" | awk '{print $1}')
echo "remote sha: $RSHA"
if [ "$RSHA" != "$EXPECT" ]; then
  echo "SHA MISMATCH - deleting pod $POD"
  curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$POD -H "Authorization: Bearer $(cat $HOME/.runpod-api-key)"; echo; exit 1
fi
echo "SHA OK"

# 6. cap_eval
echo "=== cap_eval $(date -u +%H:%M:%SZ)"
timeout 7200 $SSH "cd /root && python3 cap_eval.py --model /workspace/models/qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf --backend server --server-bin /root/llama-qwen4exp/build/bin/llama-server --server-args \"-c 512 -t 16\" --server-log /root/cap-eval-48l-ple-server.log --out /root/cap-eval-48l-ple.json --timeout 1800 2>&1 | tail -30"
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i "$KEY" -P "$PORT" root@$IP:/root/cap-eval-48l-ple.json $Q/cap-eval-48l-ple.json && echo "fetched JSON"
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i "$KEY" -P "$PORT" root@$IP:/root/cap-eval-48l-ple-server.log $Q/cap-eval-48l-ple-server.log && echo "fetched server log"
python3 - "$Q/cap-eval-48l-ple.json" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
print("RESULT: ok %d / %d = %.3f" % (d["ok"], d["n"], d["accuracy"]))
print("harness errors:", sum(1 for r in d["results"] if r.get("error")))
for c,v in d["by_category"].items(): print("  %-12s %d/%d" % (c, v["ok"], v["n"]))
EOF
echo "=== deleting pod $POD"
curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$POD -H "Authorization: Bearer $(cat $HOME/.runpod-api-key)"; echo
echo "=== pod-parallel-finish end $(date -u +%H:%M:%SZ)"
