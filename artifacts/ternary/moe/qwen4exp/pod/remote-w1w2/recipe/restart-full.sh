#!/bin/bash
# restart-full.sh - stop the current harness (if any) and relaunch the full run on the loaded server.
pkill -f "community_[e]val"
pkill -f "community-[f]ull.sh"
sleep 2
setsid nohup bash /root/community-full.sh > /root/community-full.out 2>&1 < /dev/null &
echo RESTARTED
