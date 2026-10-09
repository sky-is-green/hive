#!/bin/bash
# pod-remainder-upload.sh - append-only finish from the surviving rsync prefix.
# Local chunk staging on DISK (root btrfs), NOT /tmp (16 GB tmpfs).
set -u
IP=195.26.232.178; PORT=27379
KEY=$HOME/.ssh/id_runpod
Q=/home/penis/Desktop/work/hivebench/artifacts/ternary/moe/qwen4exp
GGUF=$Q/qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf
EXPECT=2f4bd083861b027ede8b81eacf5f8b808f04d3c72e612c42d54758a473aa22ba
POD=ycsups8e119kej
WD=/home/penis/Desktop/work/tmp-export
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -p $PORT root@$IP"
SCPOPT="-q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o Compression=no -i $KEY -P $PORT"
( sleep 12600; curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$POD -H "Authorization: Bearer $(cat $HOME/.runpod-api-key)" >/dev/null 2>&1; echo "dead-man fired $POD" ) &

$SSH 'cd /workspace/models && rm -f chunk0 chunk1 chunk2 && ls -la'
P=$($SSH "stat -c %s /workspace/models/qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf")
LOCAL=$(stat -c %s "$GGUF")
REMSZ=$((LOCAL - P))
echo "prefix=$P local=$LOCAL remainder=$REMSZ"
if [ "$REMSZ" -le 0 ]; then echo "bad sizes"; exit 1; fi
mkdir -p $WD
tail -c +$((P+1)) "$GGUF" > $WD/rem.bin
R2=$(stat -c %s $WD/rem.bin)
[ "$R2" = "$REMSZ" ] || { echo "tail size mismatch $R2 != $REMSZ"; exit 1; }
rm -f $WD/rem.part*
split -b 5G -d --suffix-length=1 $WD/rem.bin $WD/rem.part
echo "chunks:"; ls -la --block-size=M $WD/rem.part* | awk '{print $5, $9}'

T0=$(date +%s)
for f in $WD/rem.part*; do
  scp $SCPOPT "$f" root@$IP:/workspace/models/ &
done
wait
T1=$(date +%s)
python3 -c "dt=$T1-$T0; print(f'uploaded $REMSZ bytes in {dt}s = {$REMSZ/1e6/dt:.1f} MB/s')"
rm -f $WD/rem.bin $WD/rem.part*

$SSH 'cd /workspace/models && for f in $(ls rem.part* | sort); do cat "$f" >> qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf && rm -f "$f"; done; ls -la qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf; sha256sum qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf'
RSHA=$($SSH "sha256sum /workspace/models/qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf" | awk '{print $1}')
echo "remote sha: $RSHA"
if [ "$RSHA" != "$EXPECT" ]; then
  echo "SHA MISMATCH - deleting pod $POD"
  curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$POD -H "Authorization: Bearer $(cat $HOME/.runpod-api-key)"; echo; exit 1
fi
echo "SHA OK $(date -u +%H:%M:%SZ)"

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
echo "=== pod-remainder-upload end $(date -u +%H:%M:%SZ)"
