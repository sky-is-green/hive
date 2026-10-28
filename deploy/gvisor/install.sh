#!/usr/bin/env bash
# Install gVisor (runsc) for Hive's rootless Docker sandbox and switch the
# sandbox policy to it.  The script verifies the release checksum, backs up the
# rootless daemon config, restarts the user Docker service, runs a real
# workspace smoke test, and reverts the policy to the daemon default (runc) if
# the smoke test fails.
#
# Run as root: sudo bash deploy/gvisor/install.sh
set -euo pipefail

HIVE_HOME="${HIVE_HOME:-/srv/hive-ai}"
AI_USER="${AI_USER:-hive-ai}"
AI_HOME="${HIVE_HOME}/home"
AI_STATE="${HIVE_HOME}/state"
AI_RUNTIME="/run/user/949"
DOCKER_HOST_URI="unix:///run/user/949/docker.sock"
RUNSC="/usr/local/bin/runsc"
DOCKER_JSON="${AI_HOME}/.config/docker/daemon.json"
BACKUP="${DOCKER_JSON}.pre-gvisor"
WORKSPACES="${HIVE_HOME}/workspaces"
HIVE_BIN="${HIVE_HOME}/venv/bin/hive"
ARCH="$(uname -m)"
RUNSC_URL="${RUNSC_URL:-https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}/runsc}"
RUNSC_SHA_URL="${RUNSC_SHA_URL:-${RUNSC_URL}.sha512}"

if [[ ${EUID} -ne 0 ]]; then
  echo "run as root: sudo bash $0" >&2
  exit 1
fi
[[ "$ARCH" == "x86_64" ]] || {
  echo "this installer is written for x86_64 (found $ARCH)" >&2
  exit 1
}
[[ -x "$HIVE_BIN" ]] || {
  echo "missing Hive CLI: $HIVE_BIN" >&2
  exit 1
}
command -v curl >/dev/null || {
  echo "curl is required" >&2
  exit 1
}

tmp="$(mktemp -d /tmp/hive-runsc.XXXXXX)"
trap 'rm -rf "$tmp"' EXIT
echo "downloading $RUNSC_URL"
curl -fsSL "$RUNSC_URL" -o "$tmp/runsc"
curl -fsSL "$RUNSC_SHA_URL" -o "$tmp/runsc.sha512"
(
  cd "$tmp"
  sha512sum -c runsc.sha512
)
install -m 0755 -o root -g root "$tmp/runsc" "$RUNSC"
"$RUNSC" --version
if "$RUNSC" --help 2>&1 | grep -q -- '--ignore-cgroups'; then
  RUNSC_ARGS='["--rootless","--ignore-cgroups"]'
else
  RUNSC_ARGS='["--rootless"]'
fi

install -d -m 0755 -o "$AI_USER" -g "$AI_USER" "$(dirname "$DOCKER_JSON")"
if [[ -f "$DOCKER_JSON" ]]; then
  [[ -e "$BACKUP" ]] || cp -a "$DOCKER_JSON" "$BACKUP"
fi
/usr/bin/python3 - "$DOCKER_JSON" "$RUNSC_ARGS" <<'PY'
import json
import os
import sys

path, args = sys.argv[1], json.loads(sys.argv[2])
data = {}
if os.path.exists(path):
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - refuse a malformed config
        raise SystemExit(f"cannot parse {path}: {exc}")
if not isinstance(data, dict):
    raise SystemExit(f"{path} is not a JSON object")
runtimes = data.setdefault("runtimes", {})
if not isinstance(runtimes, dict):
    raise SystemExit(f"{path}: runtimes is not an object")
runtimes["runsc"] = {"path": "/usr/local/bin/runsc", "runtimeArgs": args}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, path)
PY
chown "$AI_USER:$AI_USER" "$DOCKER_JSON"
chmod 0644 "$DOCKER_JSON"
echo "rootless Docker config: $DOCKER_JSON (backup: $BACKUP)"

as_ai() {
  sudo -u "$AI_USER" env \
    HOME="$AI_HOME" \
    XDG_RUNTIME_DIR="$AI_RUNTIME" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=${AI_RUNTIME}/bus" \
    DOCKER_HOST="$DOCKER_HOST_URI" \
    "$@"
}

if ! as_ai systemctl --user cat docker.service >/dev/null 2>&1; then
  echo "the hive-ai user docker.service was not found" >&2
  echo "the runsc binary and config are installed; restart the rootless daemon manually" >&2
  exit 1
fi
as_ai systemctl --user restart docker.service
for _ in $(seq 1 30); do
  if as_ai docker info >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! as_ai docker info >/dev/null 2>&1; then
  echo "the rootless Docker daemon did not come back" >&2
  exit 1
fi
runtimes="$(as_ai docker info --format '{{json .Runtimes}}' 2>/dev/null || true)"
echo "docker runtimes: ${runtimes}"
if [[ "$runtimes" != *'"runsc"'* ]]; then
  echo "runsc is not registered with the rootless daemon" >&2
  exit 1
fi

echo "switching the Hive sandbox policy to runsc"
as_ai env HIVE_HOME="$HIVE_HOME" HIVE_STATE_DIR="$AI_STATE" \
  "$HIVE_BIN" sandbox policy --runtime runsc
as_ai env HIVE_HOME="$HIVE_HOME" HIVE_STATE_DIR="$AI_STATE" \
  "$HIVE_BIN" sandbox status

smoke="${WORKSPACES}/.gvisor-smoke-$$"
install -d -m 0700 -o "$AI_USER" -g "$AI_USER" "$smoke"
cleanup() { rm -rf "$smoke"; }
trap 'cleanup; rm -rf "$tmp"' EXIT
payload="$(/usr/bin/python3 - "$smoke" <<'PY'
import json
import sys
print(json.dumps({
    "command": "pwd; id; printf 'RUNSC_OK\\n'",
    "timeout": 30,
    "use_docker": True,
    "session_id": "gvisor-smoke",
    "workspace": sys.argv[1],
    "allow_local_fallback": False,
}))
PY
)"
response="$(curl -sS -m 120 -w '\n%{http_code}' \
  -X POST http://127.0.0.1:8765/v1/sandbox/exec \
  -H 'content-type: application/json' -d "$payload")"
http_status="${response##*$'\n'}"
response="${response%$'\n'*}"
if [[ "$http_status" != "200" ]]; then
  echo "gVisor smoke HTTP $http_status: $response" >&2
  as_ai env HIVE_HOME="$HIVE_HOME" HIVE_STATE_DIR="$AI_STATE" \
    "$HIVE_BIN" sandbox policy --runtime "" || true
  echo "reverted the sandbox policy to the daemon default" >&2
  exit 1
fi
if ! /usr/bin/python3 -c '
import json, sys
body = json.load(sys.stdin)
assert body.get("backend") == "docker", body
assert body.get("exit_code") == 0, body
assert "RUNSC_OK" in (body.get("stdout") or ""), body
print("gVisor smoke: OK", body.get("container", ""))
' <<<"$response"; then
  echo "gVisor smoke did not return RUNSC_OK; response: $response" >&2
  as_ai env HIVE_HOME="$HIVE_HOME" HIVE_STATE_DIR="$AI_STATE" \
    "$HIVE_BIN" sandbox policy --runtime "" || true
  echo "reverted the sandbox policy to the daemon default" >&2
  exit 1
fi

echo "gVisor is active. Re-run the delegate proof to exercise the child through runsc."
