#!/bin/bash
# community-proof.sh - run the WHOLE protocol (gen -> hard -> embed -> eval) on a bounded task prefix so the
# end-to-end chain is demonstrated quickly; the full run resumes from the same sample files afterwards.
set -uo pipefail
B=/workspace/bench
R=$B/community
COM=/root/community
LIM=${LIM:-40}
CONC=${CONC:-4}
exec >> "$B/community-proof.log" 2>&1
echo "=== community-proof start $(date -u) limit=$LIM conc=$CONC"
PY=python3
run() {  # run <phase> <logfile> [--server]
    local phase="$1" log="$2" server="$3"
    echo "== $phase $(date -u)"
    $PY "$COM/community_eval.py" --prefix scion-flashnext --out-dir "$R" --limit "$LIM" \
        --tasks "$COM/tasks-community.json" --phase "$phase" $server \
        > "$R/$log" 2>&1 || echo "FAILED $phase"
}
run gen   gen.log   "--server http://127.0.0.1:8897 --concurrency $CONC --model-label SkyIsNotGreen/Scion-FlashNext-176B-A6B --model-name scion-flashnext --runtime moe-corr"
run hard  hard.log  "--server http://127.0.0.1:8897 --concurrency $CONC"
run embed embed.log "--server http://127.0.0.1:8897 --concurrency $CONC"
run eval  eval.log  ""
cp -f "$R/scion-flashnext-eval.json" "$R/scion-flashnext-eval-proof$LIM.json" 2>/dev/null || true
echo "COMMUNITY_PROOF_DONE $(date -u)"
