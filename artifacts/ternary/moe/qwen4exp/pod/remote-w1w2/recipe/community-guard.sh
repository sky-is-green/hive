#!/bin/bash
# community-guard.sh <pod-id> <balance-floor> <cap-minutes>
# Minimal spend guard: delete the pod when the balance drops below the floor
# or the elapsed window cap is hit.  Logs one line every 2 minutes.
set -u
POD="${1:?pod id}"; FLOOR="${2:-5.0}"; CAP_MIN="${3:-240}"
K=$(cat "$HOME/.runpod-api-key")
start=$(date +%s)
del() { curl -s --max-time 60 -X DELETE https://rest.runpod.io/v1/pods/$POD \
        -H "Authorization: Bearer $K" >/dev/null 2>&1; }
while :; do
    bal=$(curl -s --max-time 30 "https://api.runpod.io/graphql?api_key=$K" \
          -H 'Content-Type: application/json' \
          -d '{"query":"query { myself { clientBalance } }"}' | python3 -c '
import json, sys
try: print(json.load(sys.stdin)["data"]["myself"]["clientBalance"])
except Exception: print("")')
    elapsed=$(( ( $(date +%s) - start) / 60 ))
    echo "$(date -u +%FT%TZ) guard: balance=${bal:-?} elapsed=${elapsed}m"
    if [ -n "$bal" ] && python3 -c "import sys; sys.exit(0 if float('${bal}') < float('${FLOOR}') else 1)" 2>/dev/null; then
        echo "$(date -u +%FT%TZ) balance below floor ${FLOOR}; deleting pod"
        del; exit 0
    fi
    if [ "$elapsed" -ge "$CAP_MIN" ]; then
        echo "$(date -u +%FT%TZ) window cap ${CAP_MIN}m hit; deleting pod"
        del; exit 0
    fi
    sleep 120
done
