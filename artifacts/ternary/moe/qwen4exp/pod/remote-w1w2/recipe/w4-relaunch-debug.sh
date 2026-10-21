#!/bin/bash
# w4-relaunch-debug.sh — if train-resume step 1 never reports, kill the stage,
# add phase prints to the pod's stage_train, and relaunch once.
# Run locally (not as a unit) after stopping w4-driver.
set -u
cd <work>
Q=hivebench/artifacts/ternary/moe/qwen4exp
W4=$Q/pod/remote-w4
IP=$(awk '{print $1}' "$W4/pod-endpoint")
PORT=$(awk '{print $2}' "$W4/pod-endpoint")
KEY=$HOME/.ssh/id_runpod
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -i $KEY -p $PORT"
log() { echo "$(date -u +%FT%TZ) $*"; }

log "stopping driver unit"
systemctl --user stop w4-driver w4-watch-train >/dev/null 2>&1 || true

log "killing the stuck stage group"
PID=$($SSH 'cat /workspace/logs/train-resume.pid 2>/dev/null' | tr -cd '0-9')
[ -n "$PID" ] && $SSH "kill -TERM -$PID; kill -TERM $PID" >/dev/null 2>&1
sleep 15
$SSH 'pgrep -f "qwen4exp_proxy.py train" | head -3' || true
$SSH 'rm -rf /workspace/artifacts/.stage-lock' || true

log "patching stage_train with phase prints"
$SSH 'python3 - <<"PYEOF"
import re
p = "/workspace/scion/moe/qwen4exp_proxy.py"
s = open(p).read()
assert "DBGPHASE" not in s, "already patched"
# 1) before the forward
s = s.replace(
    "            with off_ctx:\n                logits = model_logits(model, ids)\n",
    "            print(f\"DBGPHASE step {step}: forward start\", flush=True)\n"
    "            with off_ctx:\n                logits = model_logits(model, ids)\n"
    "            print(f\"DBGPHASE step {step}: forward done\", flush=True)\n", 1)
assert "DBGPHASE" in s
# 2) after backward
s = s.replace(
    "            with off_ctx:\n                loss.backward()\n",
    "            with off_ctx:\n                loss.backward()\n"
    "            print(f\"DBGPHASE step {step}: backward done\", flush=True)\n", 1)
# 3) after optimizer step
s = s.replace(
    "            opt.step()\n            opt.zero_grad(set_to_none=True)\n",
    "            opt.step()\n            opt.zero_grad(set_to_none=True)\n"
    "            print(f\"DBGPHASE step {step}: opt done\", flush=True)\n", 1)
open(p, "w").write(s)
print("patched", s.count("DBGPHASE"), "markers")
PYEOF'

log "relaunching train-resume"
$SSH 'cd /workspace/scion && bash moe/pod_watch/stage-exec.sh train-resume bash moe/box-run-flashnext.sh train-resume'

log "restarting train watchdog"
systemctl --user reset-failed w4-watch-train >/dev/null 2>&1 || true
systemd-run --user --unit=w4-watch-train --collect \
    --property=StandardOutput=append:"$W4/w4-watch-train.log" \
    --property=StandardError=append:"$W4/w4-watch-train.log" \
    --setenv=PW_POD="$(cat $W4/pod-id)" --setenv=PW_SSH_HOST="root@$IP" \
    --setenv=PW_SSH_PORT="$PORT" --setenv=PW_DST="$W4" \
    --setenv=PW_STAGE=train-resume --setenv=PW_LOG_IDLE_S=3600 \
    --setenv=PW_STEP_IDLE_S=3600 \
    --setenv=PW_KILL_PATTERN="qwen4exp_proxy.py train" \
    bash <work>/scion/moe/pod_watch/stage_watchdog.sh

log "relaunch issued; monitor with: ssh tail /workspace/logs/train-resume.log"
