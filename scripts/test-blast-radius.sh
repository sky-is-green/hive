#!/usr/bin/env bash
# Blast-radius check for the dedicated Hive AI user.
#
# Run as root AFTER scripts/setup-hive-account.sh and the rootless daemon setup.
# It creates disposable sentinels, runs a destructive shell inside the same
# systemd hardening as hive.service as the hive user, and verifies that only
# the AI writable tree was damaged.
#
# It never deletes real user files. The only AI file it removes is
# /srv/hive/state/HIVE-BLAST-AI-<pid>, created by this script.
set -euo pipefail

HIVE_USER="${HIVE_USER:-hive}"
HIVE_BASE="${HIVE_BASE:-/srv/hive}"
HUMAN_USER="${HUMAN_USER:-penis}"
HUMAN_HOME="${HUMAN_HOME:-/home/$HUMAN_USER}"
YES=0
if [[ "${1:-}" == "--yes" ]]; then YES=1; fi
if [[ "$EUID" -ne 0 ]]; then
  echo "run this with sudo: sudo $0 --yes" >&2
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
HUMAN="$HUMAN_HOME/HIVE-BLAST-HUMAN-$$"
APP="$HIVE_BASE/app/HIVE-BLAST-APP-$$"
AI="$HIVE_BASE/state/HIVE-BLAST-AI-$$"
TMPH="/tmp/HIVE-BLAST-HUMAN-$$"
cleanup() {
  rm -f "$HUMAN" "$APP" "$AI" "$TMPH" 2>/dev/null || true
}
trap cleanup EXIT

for path in "$HUMAN" "$APP" "$AI" "$TMPH"; do
  printf 'sentinel %s\n' "$$" > "$path"
  chmod 600 "$path"
done
chown "$HIVE_USER:$HIVE_USER" "$AI"

out="$(mktemp)"
trap 'rm -f "$out"; cleanup' EXIT
systemd-run --wait --pipe --collect \
  --unit="hive-blast-$$" \
  --uid="$HIVE_USER" --gid="$HIVE_USER" \
  --setenv=HUMAN="$HUMAN" \
  --setenv=APP="$APP" \
  --setenv=AI="$AI" \
  --setenv=TMPH="$TMPH" \
  --property=ProtectSystem=strict \
  --property=ProtectHome=read-only \
  --property=PrivateTmp=yes \
  --property=NoNewPrivileges=yes \
  --property=CapabilityBoundingSet= \
  --property=AmbientCapabilities= \
  --property=ReadWritePaths="$HIVE_BASE /run/user/$HIVE_UID" \
  --property=UMask=0077 \
  /bin/bash -c '
    set +e
    rm -f "$HUMAN"; echo "human_rc=$?"
    rm -f "$APP"; echo "app_rc=$?"
    rm -f "$TMPH"; echo "tmp_rc=$?"
    rm -f "$AI"; echo "ai_rc=$?"
    docker --host unix:///var/run/docker.sock info >/dev/null 2>&1
    echo "human_docker_rc=$?"
    docker --host "unix:///run/user/$(id -u)/docker.sock" info >/dev/null 2>&1
    echo "rootless_docker_rc=$?"
  ' >"$out" 2>&1 || true

cat "$out"
get() { sed -n "s/^$1=//p" "$out" | tail -1; }
fail=0
for name in human_rc app_rc; do
  if [[ "$(get "$name")" == "0" ]]; then
    echo "FAIL: $name was writable" >&2
    fail=1
  fi
done
if [[ "$(get ai_rc)" != "0" ]]; then
  echo "FAIL: AI sentinel was not removable (AI environment is not writable)" >&2
  fail=1
fi
if [[ "$(get human_docker_rc)" == "0" ]]; then
  echo "FAIL: hive can reach the human system Docker socket" >&2
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
