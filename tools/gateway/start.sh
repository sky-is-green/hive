#!/usr/bin/env bash
# Start the local LiteLLM gateway (tools/gateway/config.yaml).
#
#   ./start.sh                 # foreground on :4000
#   GATEWAY_PORT=4001 ./start.sh
#
# The API key is read from the OpenCode auth store into OPENCODE_GO_KEY for
# the proxy process only; it is never written to the config or the repo.
set -euo pipefail
cd "$(dirname "$0")"

# Local-only placeholders. The gateway binds 127.0.0.1; these are not cloud
# credentials. Override them from the environment for a different setup.
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-hive-local}"
export SCION_LOCAL_KEY="${SCION_LOCAL_KEY:-sk-local}"

export OPENCODE_GO_KEY="$(
  python3 - <<'PY'
import json, os
print(json.load(open(os.path.expanduser("~/.local/share/opencode/auth.json")))["opencode-go"]["key"])
PY
)"

# Userspace PostgreSQL (pgserver): ensure it is running and hand LiteLLM the
# socket URI.  The cluster lives under ~/.local/share/hivebench-litellm.
export DATABASE_URL="$(.venv/bin/python pg.py --uri-db litellm)"

exec .venv/bin/litellm --config config.yaml \
  --host 127.0.0.1 --port "${GATEWAY_PORT:-4000}"
