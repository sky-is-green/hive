#!/usr/bin/env bash
# splinter-session: run the splinter sidecar with exactly Studio's lifetime.
#
# Launched inside a transient systemd scope (see ~/Desktop/"Unsloth Splinter").
# Every process here lives in the scope's cgroup; when this script dies by any
# means (exit, crash, or a signal), the whole cgroup is reaped - orphans are
# impossible. No polling: event-driven via wait -n.
set -u
cd "$(dirname "$0")/.." || exit 1        # -> splinter-memory root
export OMP_NUM_THREADS=1            # encoder is 12M params; 1 thread ~5ms
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

STUDIO_BIN="${STUDIO_BIN:-$HOME/.local/bin/unsloth-web}"
PORT="${SPLINTER_PORT:-8765}"

# Standalone (2026-09-15): splinter ships its own server (splinter/server.py),
# extracted from the hivebench sidecar - no sibling checkout needed. The venv
# + conversation store live here (splinter-memory).
SPLINTER_HOME="$(pwd -P)"
export SPLINTER_HOME
PY="$SPLINTER_HOME/venv/bin/python"
STATE_DIR="${SPLINTER_STATE_DIR:-$SPLINTER_HOME/harness_state}"

port_in_use() { timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/$PORT" 2>/dev/null; }

# Retire the legacy login-scope sidecar unit, if present (superseded by this
# scope design). Idempotent, quiet.
if [ -f "$HOME/.config/systemd/user/splinter-sidecar.service" ]; then
  systemctl --user stop splinter-sidecar 2>/dev/null || true
  systemctl --user disable splinter-sidecar 2>/dev/null || true
fi

# If another splinter session already owns :8765, just open Studio (it will
# point the browser at the running instance) and skip our sidecar.
if port_in_use; then
  echo "splinter sidecar already running on :$PORT - attaching Studio only."
  exec "$STUDIO_BIN" "$@"
fi

start_sidecar() {
  ( exec "$PY" -m splinter.server --port "$PORT" --state-dir "$STATE_DIR" ) &
  SIDECAR=$!
}

"$STUDIO_BIN" "$@" &
STUDIO_PID=$!
start_sidecar
SIDE_START=$(date +%s)
FAILS=0

while :; do
  wait -n "$STUDIO_PID" "$SIDECAR" 2>/dev/null || true
  if ! kill -0 "$STUDIO_PID" 2>/dev/null; then break; fi   # Studio gone -> end
  # Sidecar died while Studio is still open: restart, unless it crash-loops.
  NOW=$(date +%s)
  [ $((NOW - SIDE_START)) -ge 5 ] && FAILS=0
  FAILS=$((FAILS + 1))
  if [ "$FAILS" -ge 5 ]; then
    echo "splinter sidecar failed to stay up (port $PORT busy or crash-loop) - Studio keeps running without splinter." >&2
    wait "$STUDIO_PID" 2>/dev/null   # keep session alive until Studio exits
    break
  fi
  sleep 3
  start_sidecar
  SIDE_START=$(date +%s)
done

kill "$SIDECAR" 2>/dev/null
wait "$SIDECAR" 2>/dev/null
exit 0
