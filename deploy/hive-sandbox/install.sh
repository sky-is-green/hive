#!/usr/bin/env bash
# Install the Hive subagent sandbox runner and profile rows.
#
# Run as root from this directory: sudo ./install.sh
# The script is idempotent: it always refreshes the root-owned wrapper and the
# workspace root, and it appends the profile block only once.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
HIVE_HOME="${HIVE_HOME:-/srv/hive-ai}"
PROFILE="${HIVE_HOME}/dsh/cordis.patch.yml"
BACKUP="${PROFILE}.pre-subagent-sandbox"
WRAPPER="${HIVE_HOME}/bin/hive-bwrap.py"
WORKSPACES="${HIVE_HOME}/workspaces"
AI_USER="${AI_USER:-hive-ai}"
HOOK_SRC="${REPO_ROOT}/harness/tool_hook.py"
HOOK_DST="${HIVE_HOME}/app/harness/tool_hook.py"
HOOK_BACKUP="${HOOK_DST}.pre-subagent-sandbox"
# The live app is a copy of the repo.  Only replace its hook when the file is
# exactly the pre-change version; otherwise refuse and ask for a manual merge.
HOOK_EXPECTED="1b0396ce905c28609f51c9066e459f70c7a4e88bf66cf7905bc1fab3281d0bed"

if [[ ${EUID} -ne 0 ]]; then
  echo "run as root: sudo $0" >&2
  exit 1
fi
command -v bwrap >/dev/null || {
  echo "bubblewrap is required (install bwrap first)" >&2
  exit 1
}
[[ -x /usr/bin/python3 ]] || {
  echo "/usr/bin/python3 is required" >&2
  exit 1
}
[[ -f "$PROFILE" ]] || {
  echo "missing DSH profile: $PROFILE" >&2
  exit 1
}

install -d -m 0755 "${HIVE_HOME}/bin"
chown root:root "${HIVE_HOME}/bin"
chmod 0755 "${HIVE_HOME}/bin"
install -m 0755 -o root -g root "${HERE}/hive_bwrap.py" "$WRAPPER"

if [[ ! -d "$WORKSPACES" ]]; then
  install -d -m 0750 -o "$AI_USER" -g "$AI_USER" "$WORKSPACES"
else
  chown "$AI_USER:$AI_USER" "$WORKSPACES"
  chmod 0750 "$WORKSPACES"
fi

# Prove the exact runner works as the AI user before changing the profile.
SMOKE="${WORKSPACES}/.smoke-$$"
install -d -m 0700 -o "$AI_USER" -g "$AI_USER" "$SMOKE"
printf 'seed\n' > "$SMOKE/seed.txt"
chown "$AI_USER:$AI_USER" "$SMOKE/seed.txt"
cleanup() { rm -rf "$SMOKE"; }
trap cleanup EXIT

(
  cd "$SMOKE"
  sudo -u "$AI_USER" /usr/bin/python3 "$WRAPPER" \
    --ro-bind / / --dev /dev --unshare-pid --proc /proc --die-with-parent \
    --tmpfs /tmp --bind "$SMOKE" "$SMOKE" -- \
    bash -c '
      set -eu
      test -w .
      test -f seed.txt
      printf "ok\n" > written.txt
      test -z "$(ls -A /home 2>/dev/null)"
      test -z "$(ls -A /run 2>/dev/null)"
      test -z "$(ls -A /srv/hive-ai/state 2>/dev/null)"
      test -z "$(ls -A /srv/hive-ai/dsh 2>/dev/null)"
      if curl -sS -m 1 http://127.0.0.1:8150/health >/dev/null 2>&1; then
        printf "network is visible\n" >&2
        exit 9
      fi
      printf "HIVE_SANDBOX_OK\n"
    '
)
test -f "$SMOKE/written.txt" || {
  echo "smoke test did not write to the workspace" >&2
  exit 1
}
echo "runner smoke test: OK"

if [[ ! -f "$HOOK_SRC" ]]; then
  echo "missing hook source: $HOOK_SRC" >&2
  exit 1
fi
if [[ ! -f "$HOOK_DST" ]]; then
  echo "missing live hook: $HOOK_DST" >&2
  exit 1
fi
live_hook_sha="$(sha256sum "$HOOK_DST" | awk '{print $1}')"
new_hook_sha="$(sha256sum "$HOOK_SRC" | awk '{print $1}')"
if [[ "$live_hook_sha" == "$new_hook_sha" ]]; then
  echo "live hook already has the subagent workspace rule"
elif [[ "$live_hook_sha" == "$HOOK_EXPECTED" ]]; then
  [[ -e "$HOOK_BACKUP" ]] || cp -a "$HOOK_DST" "$HOOK_BACKUP"
  install -m 0644 -o root -g root "$HOOK_SRC" "$HOOK_DST"
  echo "installed the subagent workspace hook"
  echo "hook backup: $HOOK_BACKUP"
else
  echo "refusing to replace $HOOK_DST" >&2
  echo "live checksum: $live_hook_sha" >&2
  echo "expected:      $HOOK_EXPECTED" >&2
  echo "new:           $new_hook_sha" >&2
  echo "merge harness/tool_hook.py by hand, then re-run" >&2
  exit 1
fi

if grep -q 'id: hive-subagent-tool' "$PROFILE"; then
  echo "profile already has the sandboxed subagent rows"
else
  [[ -e "$BACKUP" ]] || cp -a "$PROFILE" "$BACKUP"
  {
    printf '\n'
    cat "${HERE}/cordis.patch.yml"
  } >> "$PROFILE"
  echo "appended the sandboxed subagent rows"
  echo "profile backup: $BACKUP"
fi

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
curl -fsS -m 3 http://127.0.0.1:8765/health
printf '\n'
systemctl is-active hive.service
