#!/bin/bash
# community-full.sh - the complete protocol on all 664 tasks, resuming from whatever the proof pass wrote.
# Launch only after the proof has shown the chain works end to end.
set -uo pipefail
# checkers.py runs candidate code from tempfile.NamedTemporaryFile; an inherited TMPDIR from the launching
# shell pointed at a path that does not exist on the pod, and the first code task died on it.
export TMPDIR=/tmp
B=/workspace/bench
R=$B/community
COM=/root/community
CONC=${CONC:-4}
exec >> "$B/community-full.log" 2>&1
echo "=== community-full start $(date -u) conc=$CONC"
PY=python3
run() {  # run <phase> <logfile> [server args]
    local phase="$1" log="$2" server="$3"
    echo "== $phase $(date -u)"
    $PY "$COM/community_eval.py" --prefix scion-flashnext --out-dir "$R" \
        --tasks "$COM/tasks-community.json" --phase "$phase" $server \
        > "$R/$log" 2>&1 || echo "FAILED $phase"
}
run gen   gen.log   "--server http://127.0.0.1:8897 --concurrency $CONC --model-label SkyIsNotGreen/Scion-FlashNext-176B-A6B --model-name scion-flashnext --runtime moe-corr"
run hard  hard.log  "--server http://127.0.0.1:8897 --concurrency $CONC"
run embed embed.log "--server http://127.0.0.1:8897 --concurrency $CONC"
run eval  eval.log  ""
tar czf "$B/community-artifacts.tgz" -C "$R" . 2>/dev/null || true
echo "COMMUNITY_FULL_DONE $(date -u)"
