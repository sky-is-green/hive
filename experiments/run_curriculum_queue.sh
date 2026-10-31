#!/usr/bin/env bash
# Gated queue: curriculum recovery -> critic-only separate adapter.
#
# Waits for the placement-planner recovery training, evaluates controller,
# planner, placement and placement-planning, runs the curriculum gate, and only
# then trains critic-only. Stops at the first failed gate. Restores Scion at the
# end either way.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=/home/penis/.unsloth/studio/unsloth_studio/bin/python
RUN=../train-runs
CAND=$RUN/multirole-cpp-plan-v1
BEFORE=$RUN/multirole-cpp-v1
LOG=/tmp/opencode/curriculum_queue.log
LLAMA_SRC=/tmp/opencode/llama-b11960-src
SERVER=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/llama-b11960-vulkan/llama-b11960/llama-server
QUANTIZE=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/llama-b11960-vulkan/llama-b11960/llama-quantize
TOKENIZERS_PARALLELISM=false
export TOKENIZERS_PARALLELISM

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

restore_scion() {
  if curl -s -m 2 -o /dev/null http://127.0.0.1:8082/health; then
    log "scion already healthy"
    return
  fi
  log "restoring scion"
  (cd ../strata && setsid nohup env STRATA_Q8_PACKED=0 ROCM_PATH=/opt/rocm \
    .venv/bin/python serve/server.py --engine strata --config strata-35b-a3b-hive.json \
    --port 8082 >> /tmp/opencode/scion-35b-restart.log 2>&1 < /dev/null &)
  for _ in $(seq 1 30); do
    curl -s -m 2 -o /dev/null http://127.0.0.1:8082/health && { log "scion ready"; return; }
    sleep 5
  done
  log "scion did not become healthy"
}

run_eval() {
  local tasks="$1" out="$2"
  log "eval $(basename "$tasks")"
  HIP_VISIBLE_DEVICES=1 "$PY" experiments/eval_role_sft.py \
    --model-dir "$CAND/merged" --tasks "experiments/roles/$tasks" \
    --out "$CAND/$out" >> "$LOG" 2>&1
}

# 1. Wait for the recovery training to finish.
log "waiting for recovery training"
for _ in $(seq 1 180); do
  if [ -f "$CAND/merged/model.safetensors" ] && \
     ! pgrep -f "lfm3_sft.py.*multirole-cpp-plan-v1" >/dev/null; then
    break
  fi
  sleep 10
done
if [ ! -f "$CAND/merged/model.safetensors" ]; then
  log "recovery training did not produce merged/model.safetensors"
  restore_scion
  exit 2
fi

# 2. Before/after evaluation.
run_eval tasks-controller.json eval-controller.json || { log "controller eval failed"; restore_scion; exit 3; }
run_eval tasks-planner.json eval-planner.json || { log "planner eval failed"; restore_scion; exit 3; }
run_eval tasks-placement.json eval-placement.json || { log "placement eval failed"; restore_scion; exit 3; }
run_eval tasks-placement-plan.json eval-placement-plan.json || { log "placement-plan eval failed"; restore_scion; exit 3; }

# 3. Curriculum gate.
log "curriculum gate"
if ! "$PY" experiments/curriculum_gate.py \
    --candidate-dir "$CAND" --before-dir "$BEFORE" \
    --out "$CAND/curriculum-gate.json" >> "$LOG" 2>&1; then
  log "curriculum gate FAILED; stopping queue"
  restore_scion
  exit 4
fi
log "curriculum gate PASSED"

# 4. Critic-only separate adapter.
log "training critic-only"
HIP_VISIBLE_DEVICES=1 "$PY" experiments/lfm3_sft.py \
  --model-dir ../models-hf/LFM3-1.7B-Base \
  --data experiments/data/critic-sft-v1/train.jsonl \
  --out "$RUN/critic-sft-v1" \
  --epochs 3 --batch 4 --grad-accum 4 --lr 2e-4 --max-length 512 \
  --save-steps 150 --log-steps 20 --merge >> "$LOG" 2>&1 || {
    log "critic training failed"; restore_scion; exit 5;
  }

log "eval critic HF"
HIP_VISIBLE_DEVICES=1 "$PY" experiments/eval_role_sft.py \
  --model-dir "$RUN/critic-sft-v1/merged" \
  --tasks experiments/roles/tasks-critic.json \
  --out "$RUN/critic-sft-v1/eval-critic.json" >> "$LOG" 2>&1 || {
    log "critic HF eval failed"; restore_scion; exit 6;
  }

log "convert critic GGUF"
(
  cd "$LLAMA_SRC" || exit 1
  PYTHONPATH="$LLAMA_SRC/gguf-py" "$PY" convert_hf_to_gguf.py \
    "$(cd "$RUN/critic-sft-v1/merged" && pwd)" \
    --outfile "$(cd "$RUN/critic-sft-v1" && pwd)/critic-sft-v1-bf16.gguf" \
    --outtype bf16
) >> "$LOG" 2>&1 || { log "critic conversion failed"; restore_scion; exit 7; }
"$QUANTIZE" "$RUN/critic-sft-v1/critic-sft-v1-bf16.gguf" \
  "$RUN/critic-sft-v1/critic-sft-v1-Q4_K_M.gguf" Q4_K_M >> "$LOG" 2>&1 || {
    log "critic quantize failed"; restore_scion; exit 8;
  }

log "start critic server :8159"
GGML_VK_VISIBLE_DEVICES=0 setsid nohup "$SERVER" \
  -m "$RUN/critic-sft-v1/critic-sft-v1-Q4_K_M.gguf" \
  --alias critic-1.7b-sft --host 127.0.0.1 --port 8159 \
  -ngl 999 -c 8192 -np 2 --jinja --reasoning off --no-webui \
  >> /tmp/opencode/critic-server.log 2>&1 < /dev/null &
for _ in $(seq 1 20); do
  curl -s -m 2 -o /dev/null http://127.0.0.1:8159/health && break
  sleep 2
done
if ! curl -s -m 2 -o /dev/null http://127.0.0.1:8159/health; then
  log "critic server did not start"; restore_scion; exit 9
fi

log "eval critic GGUF"
"$PY" experiments/eval_role_endpoint.py \
  --base http://127.0.0.1:8159 --model critic-1.7b-sft \
  --tasks experiments/roles/tasks-critic.json --max-tokens 512 \
  --no-response-format --out "$RUN/critic-sft-v1/eval-critic-gguf.json" \
  >> "$LOG" 2>&1 || { log "critic GGUF eval failed"; restore_scion; exit 10; }

pid=$(ps -eo pid,args | grep "llama-server.*port 8159" | grep -v grep | awk '{print $1}')
[ -n "$pid" ] && kill "$pid"

log "critic-only gate"
if ! "$PY" experiments/role_gate.py --batch critic-only \
    --candidate-dir "$RUN/critic-sft-v1" \
    --out "$RUN/critic-sft-v1/gate-report.json" >> "$LOG" 2>&1; then
  log "critic-only gate FAILED"
  restore_scion
  exit 11
fi

log "critic-only gate PASSED; queue complete"
restore_scion
exit 0
