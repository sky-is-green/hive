#!/usr/bin/env bash
# Install the Hive sandboxed-subagent bundle.
#
# Child bash runs in Hive's rootless Docker sandbox: /v1/sandbox/exec validates
# a workspace under /srv/hive-ai/workspaces, mounts it at /workspace, and runs
# the pinned image with no network, no host home, no Hive state, and
# hive.session/hive.parent labels.  The parent keeps the base DSH sandbox.
#
# Run as root from anywhere: sudo bash .../install.sh
# The script is idempotent: it refreshes the app files, hook, plugin, profile
# block, and workspace root.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
HIVE_HOME="${HIVE_HOME:-/srv/hive-ai}"
PROFILE="${HIVE_HOME}/dsh/cordis.patch.yml"
BACKUP="${PROFILE}.pre-docker-sandbox"
WORKSPACES="${HIVE_HOME}/workspaces"
AI_USER="${AI_USER:-hive-ai}"
PLUGIN_SRC="${HERE}/plugin"
PLUGIN_DST="${HIVE_HOME}/tools/dsh/node_modules/@deepseek-ai/hive-docker-bash"
MARKER='# Hive sandboxed subagents (managed by deploy/hive-sandbox/install.sh).'

# The versions currently installed by the Docker conversion.  The next run
# accepts these (or the new source already) and refuses any other live file.
APP_PY_EXPECTED="d2ffd072422454d42f27d445d2f44d020f1f72cbb3086bec1ef06de17cd11aff"
SANDBOX_PY_EXPECTED="d66109edd8c4a6b0d716f6f31570048f3935b23bd269d2594f572565727dcf3a"
SECURITY_PY_EXPECTED="2945003717db7daff203060d53192e16a2e000e0f90297a0a5336f5210283482"
HOOK_PY_EXPECTED="62f78ff5b813c187c26efa27923683ea9943fbe8595479834174350dfe511826"

if [[ ${EUID} -ne 0 ]]; then
  echo "run as root: sudo bash $0" >&2
  exit 1
fi
[[ -x /usr/bin/python3 ]] || {
  echo "/usr/bin/python3 is required" >&2
  exit 1
}
[[ -f "$PROFILE" ]] || {
  echo "missing DSH profile: $PROFILE" >&2
  exit 1
}
[[ -f "$PLUGIN_SRC/index.js" ]] || {
  echo "missing plugin source: $PLUGIN_SRC/index.js" >&2
  exit 1
}

# The bwrap experiment is retired; remove its installed runner if present.
rm -f "${HIVE_HOME}/bin/hive-bwrap.py"

if [[ ! -d "$WORKSPACES" ]]; then
  install -d -m 0750 -o "$AI_USER" -g "$AI_USER" "$WORKSPACES"
else
  chown "$AI_USER:$AI_USER" "$WORKSPACES"
  chmod 0750 "$WORKSPACES"
fi

validate_live() {
  local src="$1" dst="$2" expected="$3" live new
  if [[ ! -f "$src" ]]; then
    echo "missing source: $src" >&2
    return 1
  fi
  if [[ ! -f "$dst" ]]; then
    echo "missing live file: $dst" >&2
    return 1
  fi
  live="$(sha256sum "$dst" | awk '{print $1}')"
  new="$(sha256sum "$src" | awk '{print $1}')"
  if [[ "$live" == "$new" || "$live" == "$expected" ]]; then
    return 0
  fi
  echo "refusing to replace $dst" >&2
  echo "  live:     $live" >&2
  echo "  expected: $expected" >&2
  echo "  new:      $new" >&2
  echo "merge the file by hand, then re-run" >&2
  return 1
}

install_live() {
  local src="$1" dst="$2" expected="$3" live new
  if [[ ! -f "$src" ]]; then
    echo "missing source: $src" >&2
    return 1
  fi
  if [[ ! -f "$dst" ]]; then
    echo "missing live file: $dst" >&2
    return 1
  fi
  live="$(sha256sum "$dst" | awk '{print $1}')"
  new="$(sha256sum "$src" | awk '{print $1}')"
  if [[ "$live" == "$new" ]]; then
    echo "current: $dst"
    return 0
  fi
  if [[ "$live" != "$expected" ]]; then
    echo "refusing to replace $dst" >&2
    echo "  live:     $live" >&2
    echo "  expected: $expected" >&2
    echo "  new:      $new" >&2
    echo "merge the file by hand, then re-run" >&2
    return 1
  fi
  [[ -e "$dst.pre-docker-sandbox" ]] || cp -a "$dst" "$dst.pre-docker-sandbox"
  install -m 0644 -o root -g root "$src" "$dst"
  echo "installed $dst (backup $dst.pre-docker-sandbox)"
}

validate_live "${REPO_ROOT}/harness/app.py" \
  "${HIVE_HOME}/app/harness/app.py" "$APP_PY_EXPECTED"
validate_live "${REPO_ROOT}/harness/sandbox.py" \
  "${HIVE_HOME}/app/harness/sandbox.py" "$SANDBOX_PY_EXPECTED"
validate_live "${REPO_ROOT}/harness/security.py" \
  "${HIVE_HOME}/app/harness/security.py" "$SECURITY_PY_EXPECTED"
validate_live "${REPO_ROOT}/harness/tool_hook.py" \
  "${HIVE_HOME}/app/harness/tool_hook.py" "$HOOK_PY_EXPECTED"
install_live "${REPO_ROOT}/harness/app.py" \
  "${HIVE_HOME}/app/harness/app.py" "$APP_PY_EXPECTED"
install_live "${REPO_ROOT}/harness/sandbox.py" \
  "${HIVE_HOME}/app/harness/sandbox.py" "$SANDBOX_PY_EXPECTED"
install_live "${REPO_ROOT}/harness/security.py" \
  "${HIVE_HOME}/app/harness/security.py" "$SECURITY_PY_EXPECTED"
install_live "${REPO_ROOT}/harness/tool_hook.py" \
  "${HIVE_HOME}/app/harness/tool_hook.py" "$HOOK_PY_EXPECTED"

install -d -m 0755 "$PLUGIN_DST"
install -m 0644 -o root -g root "$PLUGIN_SRC/package.json" \
  "$PLUGIN_DST/package.json"
install -m 0644 -o root -g root "$PLUGIN_SRC/index.js" \
  "$PLUGIN_DST/index.js"
chown -R root:root "$PLUGIN_DST"
chmod 0755 "$PLUGIN_DST"
if command -v node >/dev/null 2>&1; then
  node --check "$PLUGIN_DST/index.js"
fi

if grep -qF "$MARKER" "$PROFILE"; then
  [[ -e "$BACKUP" ]] || cp -a "$PROFILE" "$BACKUP"
  awk -v marker="$MARKER" 'index($0, marker) {exit} {print}' \
    "$PROFILE" > "${PROFILE}.new"
  mv "${PROFILE}.new" "$PROFILE"
  chown root:"$AI_USER" "$PROFILE"
  chmod 0640 "$PROFILE"
fi
{
  printf '\n'
  cat "${HERE}/cordis.patch.yml"
} >> "$PROFILE"

systemctl restart hive.service
healthy=0
for _ in $(seq 1 60); do
  if curl -fsS -m 2 http://127.0.0.1:8765/health >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 1
done
if [[ "$healthy" -ne 1 ]]; then
  echo "hive.service did not become healthy; inspect the journal" >&2
  echo "rollback: sudo cp -a $BACKUP $PROFILE && sudo systemctl restart hive.service" >&2
  exit 1
fi
echo "hive.service: active"

SMOKE="${WORKSPACES}/.docker-smoke-$$"
install -d -m 0700 -o "$AI_USER" -g "$AI_USER" "$SMOKE"
cleanup() { rm -rf "$SMOKE"; }
trap cleanup EXIT
payload="$(/usr/bin/python3 - "$SMOKE" <<'PY'
import json, sys
print(json.dumps({
    "command": "pwd; id; printf 'DOCKER_SANDBOX_OK\\n'",
    "timeout": 30,
    "use_docker": True,
    "session_id": "install-smoke",
    "workspace": sys.argv[1],
    "allow_local_fallback": False,
}))
PY
)"
response="$(curl -sS -m 90 -w '\n%{http_code}' \
  -X POST http://127.0.0.1:8765/v1/sandbox/exec \
  -H 'content-type: application/json' -d "$payload")"
http_status="${response##*$'\n'}"
response="${response%$'\n'*}"
if [[ "$http_status" != "200" ]]; then
  echo "docker sandbox HTTP $http_status: $response" >&2
  echo "inspect: journalctl -u hive.service -n 80 --no-pager" >&2
  exit 1
fi
/usr/bin/python3 -c '
import json, sys
body = json.load(sys.stdin)
assert body.get("backend") == "docker", body
assert body.get("exit_code") == 0, body
assert "DOCKER_SANDBOX_OK" in (body.get("stdout") or ""), body
assert body.get("workspace"), body
print("docker sandbox: OK", body.get("container", ""))
' <<<"$response"
