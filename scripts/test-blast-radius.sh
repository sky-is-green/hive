#!/usr/bin/env bash
# Blast-radius check for the dedicated Hive AI user.
#
# Run as root AFTER scripts/setup-hive-account.sh and the rootless daemon setup.
# It creates disposable sentinels, runs a destructive shell inside the same
# systemd hardening as hive.service as the hive-ai user, and verifies that only
# the AI writable tree was damaged.
#
# It never deletes real user files. The only AI file it removes is
# /srv/hive-ai/state/HIVE-BLAST-AI-<pid>, created by this script.
set -euo pipefail

HIVE_USER="${HIVE_USER:-hive-ai}"
HIVE_BASE="${HIVE_BASE:-/srv/hive-ai}"
HUMAN_USER="${HUMAN_USER:-${SUDO_USER:-penis}}"
HUMAN_HOME="${HUMAN_HOME:-/home/$HUMAN_USER}"
YES=0
if [[ "${1:-}" == "--yes" ]]; then YES=1; fi
if [[ "$EUID" -ne 0 ]]; then
  echo "run this with sudo: sudo $0 --yes" >&2
  exit 1
fi
if [[ "$HIVE_USER" == "hive" ]]; then
  echo "refusing to test 'hive': that account belongs to GUARD, not Hive." >&2
  echo "Use the default hive-ai user." >&2
  exit 1
fi
if [[ "$YES" -ne 1 ]]; then
  cat <<EOF
This will create and delete only disposable sentinel files:
  $HUMAN_HOME/HIVE-BLAST-HUMAN-$$
  $HIVE_BASE/app/HIVE-BLAST-APP-$$
  $HIVE_BASE/state/HIVE-BLAST-AI-$$
  /tmp/HIVE-BLAST-HUMAN-$$
It will then run a destructive shell as '$HIVE_USER' with the same systemd
hardening as hive.service. Re-run with --yes to continue.
EOF
  exit 2
fi
if ! id "$HIVE_USER" >/dev/null 2>&1; then
  echo "AI user '$HIVE_USER' does not exist; run setup first" >&2
  exit 1
fi
if [[ ! -d "$HIVE_BASE" ]]; then
  echo "$HIVE_BASE does not exist; run setup first" >&2
  exit 1
fi
HIVE_UID="$(id -u "$HIVE_USER")"
HIVE_GROUP="$(id -gn "$HIVE_USER")"
if id -nG "$HIVE_USER" | tr ' ' '\n' | grep -qx docker; then
  echo "FAIL: '$HIVE_USER' is in the human docker group" >&2
  exit 1
fi
if id hive >/dev/null 2>&1 && [[ "$(id -u hive)" == "$HIVE_UID" ]]; then
  echo "FAIL: '$HIVE_USER' shares GUARD's uid; they must be separate accounts" >&2
  exit 1
fi
HUMAN="$HUMAN_HOME/HIVE-BLAST-HUMAN-$$"
APP="$HIVE_BASE/app/HIVE-BLAST-APP-$$"
AI="$HIVE_BASE/state/HIVE-BLAST-AI-$$"
TMPH="/tmp/HIVE-BLAST-HUMAN-$$"
GUARD="${GUARD_DIR:-$HUMAN_HOME/.guard}"
cleanup() {
  rm -f "$HUMAN" "$APP" "$AI" "$TMPH" \
    "${GUARD:-}/HIVE-BLAST-GUARD-$$" 2>/dev/null || true
}
trap cleanup EXIT

for path in "$HUMAN" "$APP" "$AI" "$TMPH"; do
  printf 'sentinel %s\n' "$$" > "$path"
  chmod 600 "$path"
done
# Make the human sentinel readable in principle so the test also proves the
# AI user cannot traverse the human home.  The path remains inaccessible to
# hive-ai; the file being readable to its owner is harmless.
chmod 644 "$HUMAN"
chown "$HIVE_USER:$HIVE_GROUP" "$AI"

out="$(mktemp)"
trap 'rm -f "$out"; cleanup' EXIT
systemd-run --wait --pipe --collect \
  --unit="hive-blast-$$" \
  --uid="$HIVE_USER" --gid="$HIVE_GROUP" \
  --setenv=HUMAN="$HUMAN" \
  --setenv=HUMAN_HOME="$HUMAN_HOME" \
  --setenv=APP="$APP" \
  --setenv=AI="$AI" \
  --setenv=TMPH="$TMPH" \
  --setenv=GUARD="$GUARD" \
  --property=ProtectSystem=strict \
  --property=ProtectHome=no \
  --property=InaccessiblePaths="/home -/root" \
  --property=PrivateTmp=yes \
  --property=NoNewPrivileges=yes \
  --property=ProtectKernelTunables=yes \
  --property=ProtectKernelModules=yes \
  --property=ProtectControlGroups=yes \
  --property=RestrictRealtime=yes \
  --property=RestrictSUIDSGID=yes \
  --property=LockPersonality=yes \
  --property=CapabilityBoundingSet= \
  --property=AmbientCapabilities= \
  --property=ReadWritePaths="$HIVE_BASE" \
  --property=UMask=0077 \
  /bin/bash -c '
    set +e
    # InaccessiblePaths=/home leaves an empty /home mountpoint visible; the
    # actual human home must be gone, not the parent mountpoint.
    test -e "$HUMAN_HOME"; echo "home_visible_rc=$?"
    rm -f "$HUMAN"; echo "human_rc=$?"
    cat "$HUMAN" >/dev/null 2>&1; echo "human_read_rc=$?"
    # -f prevents rm from waiting at an interactive "write-protected file?"
    # prompt.  The timeout is a belt-and-braces bound on a slow filesystem.
    timeout 5 rm -f "$APP"; echo "app_rc=$?"
    # PrivateTmp makes the host sentinel absent; -f would hide that with rc 0.
    if [ -e "$TMPH" ]; then rm -f "$TMPH"; echo "tmp_rc=$?"; else echo "tmp_rc=1"; fi
    rm -f "$AI"; echo "ai_rc=$?"
    touch "/run/user/$(id -u)/HIVE-BLAST-RUNTIME-$$"
    echo "runtime_write_rc=$?"
    rm -f "/run/user/$(id -u)/HIVE-BLAST-RUNTIME-$$"
    if [ -d "$GUARD" ]; then
      ls "$GUARD" >/dev/null 2>&1; echo "guard_list_rc=$?"
      touch "$GUARD/HIVE-BLAST-GUARD-$$" 2>/dev/null
      echo "guard_touch_rc=$?"
      rm -f "$GUARD/HIVE-BLAST-GUARD-$$" 2>/dev/null
    fi
    docker --host unix:///var/run/docker.sock info >/dev/null 2>&1
    echo "human_docker_rc=$?"
    docker --host "unix:///run/user/$(id -u)/docker.sock" info >/dev/null 2>&1
    echo "rootless_docker_rc=$?"
  ' >"$out" 2>&1 || true

cat "$out"
get() { sed -n "s/^$1=//p" "$out" | tail -1; }
fail=0
for name in human_rc app_rc tmp_rc; do
  if [[ "$(get "$name")" == "0" ]]; then
    echo "FAIL: $name was removable (host boundary failed)" >&2
    fail=1
  fi
done
if [[ "$(get home_visible_rc)" == "0" ]]; then
  echo "FAIL: /home is visible inside the AI unit" >&2
  fail=1
fi
if [[ "$(get human_read_rc)" == "0" ]]; then
  echo "FAIL: hive-ai could read the human-home sentinel" >&2
  fail=1
fi
if [[ "$(get ai_rc)" != "0" ]]; then
  echo "FAIL: AI sentinel was not removable (AI environment is not writable)" >&2
  fail=1
fi
if [[ "$(get runtime_write_rc)" != "0" ]]; then
  echo "FAIL: AI could not write its own /run/user socket directory" >&2
  fail=1
fi
if [[ -n "$(get guard_list_rc)" && "$(get guard_list_rc)" == "0" ]]; then
  echo "FAIL: hive-ai could list GUARD's directory" >&2
  fail=1
fi
if [[ -n "$(get guard_touch_rc)" && "$(get guard_touch_rc)" == "0" ]]; then
  echo "FAIL: hive-ai could write in GUARD's directory" >&2
  fail=1
fi
if [[ "$(get human_docker_rc)" == "0" ]]; then
  echo "FAIL: hive-ai can reach the human system Docker socket" >&2
  fail=1
fi
for path in "$HUMAN" "$APP" "$TMPH"; do
  if [[ ! -f "$path" ]]; then
    echo "FAIL: host sentinel disappeared: $path" >&2
    fail=1
  fi
done
if [[ -e "$AI" ]]; then
  echo "FAIL: AI sentinel survived (the AI tree was not writable)" >&2
  fail=1
fi
if [[ "$(get rootless_docker_rc)" != "0" ]]; then
  echo "WARN: rootless Docker is not reachable yet; run the rootless setup." >&2
fi
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
echo "PASS: destructive shell could only remove the AI sentinel."
