#!/usr/bin/env bash
# Prepare the dedicated Hive AI account and /srv/hive-ai layout.
#
# This script is intentionally root-only and deliberately does NOT:
#   - reuse GUARD's 'hive' account or touch /home/penis/.guard;
#   - add the AI user to the docker group;
#   - start or modify the human system Docker daemon;
#   - install or start a rootless Docker daemon automatically.
#
# Run it with sudo, review the printed rootless-Docker step, then start Hive.
set -euo pipefail

HIVE_USER="${HIVE_USER:-hive-ai}"
HIVE_BASE="${HIVE_BASE:-/srv/hive-ai}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DRY_RUN=0
WITH_VENV=1

usage() {
  cat <<'EOF'
Usage: sudo ./scripts/setup-hive-account.sh [--dry-run] [--no-venv] [REPO]

Creates/updates the dedicated Hive AI user and /srv/hive-ai layout, copies the
tracked application tree, and installs the optional Studio environment.
It never adds the user to the docker group and never starts system Docker.

The default account is hive-ai.  GUARD already owns the 'hive' account, so
using User=hive for Hive would share GUARD's queue, approvals, and allowlist.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --no-venv) WITH_VENV=0 ;;
    -h|--help) usage; exit 0 ;;
    *) REPO="$1" ;;
  esac
  shift
done

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

run_shell() {
  printf '+ %s\n' "$*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    bash -c "$*"
  fi
}

if [[ "$DRY_RUN" -eq 0 && "$EUID" -ne 0 ]]; then
  echo "run this with sudo: sudo $0" >&2
  exit 1
fi
if [[ "$HIVE_USER" == "hive" ]]; then
  cat >&2 <<'EOF'
refusing to use the 'hive' account: it is owned by GUARD.

GUARD runs as hive with HOME=/home/penis and owns /home/penis/.guard.  Giving
Hive the same UID would let it read GUARD's queue, approvals, and allowlist.
Use the default hive-ai account, or set HIVE_USER to a different unused name.
EOF
  exit 1
fi
if [[ ! -f "$REPO/pyproject.toml" ]]; then
  echo "repository not found: $REPO" >&2
  exit 1
fi
if [[ "$HIVE_BASE" != "/srv/hive" && -e /srv/hive ]]; then
  echo "note: leaving the existing /srv/hive tree alone; this deployment uses $HIVE_BASE" >&2
fi
if ! id "$HIVE_USER" >/dev/null 2>&1; then
  if ! command -v useradd >/dev/null 2>&1; then
    echo "user '$HIVE_USER' does not exist and useradd is unavailable" >&2
    exit 1
  fi
  run useradd --system --user-group --home-dir "$HIVE_BASE/home" \
    --shell /usr/sbin/nologin "$HIVE_USER"
fi
if id "$HIVE_USER" >/dev/null 2>&1; then
  HIVE_GROUP="$(id -gn "$HIVE_USER")"
  HIVE_UID="$(id -u "$HIVE_USER")"
  CURRENT_HOME="$(getent passwd "$HIVE_USER" | cut -d: -f6 || true)"
  if [[ "$CURRENT_HOME" != "$HIVE_BASE/home" ]]; then
    echo "note: $HIVE_USER passwd home is $CURRENT_HOME; the unit uses $HIVE_BASE/home via HOME=" >&2
  fi
else
  # --dry-run did not execute useradd; keep printing a plausible plan.
  HIVE_GROUP="$HIVE_USER"
  HIVE_UID="<new>"
fi
if [[ "$HIVE_UID" =~ ^[0-9]+$ ]] && [[ "$HIVE_UID" -eq 0 ]]; then
  echo "AI user '$HIVE_USER' resolved to uid 0; refusing" >&2
  exit 1
fi
if id hive >/dev/null 2>&1 && [[ "$(id -u hive)" == "$HIVE_UID" ]]; then
  echo "AI user '$HIVE_USER' shares GUARD's uid; refusing" >&2
  exit 1
fi
if id -nG "$HIVE_USER" 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
  echo "AI user '$HIVE_USER' is in the human docker group; refusing" >&2
  exit 1
fi

# Do not call usermod here.  GUARD's process already runs as 'hive', and
# usermod refuses to change a home while the account is in use.  The unit
# sets HOME explicitly, so the passwd home is informational only.

# The base and the code/venv are root-owned, so the AI cannot rename them
# even though it can write the state directories below.  Only the named
# per-user directories belong to the AI.
run install -d -o root -g "$HIVE_GROUP" -m 0750 "$HIVE_BASE"
for directory in home state dsh models stacks logs tools; do
  run install -d -o "$HIVE_USER" -g "$HIVE_GROUP" -m 0700 "$HIVE_BASE/$directory"
done

# Rootless container prerequisites.  The AI user must never join the system
# docker group; its own daemon gets its own socket and data root.
if ! grep -q "^${HIVE_USER}:" /etc/subuid 2>/dev/null; then
  run usermod --add-subuids 100000-165535 "$HIVE_USER"
fi
if ! grep -q "^${HIVE_USER}:" /etc/subgid 2>/dev/null; then
  run usermod --add-subgids 100000-165535 "$HIVE_USER"
fi
run loginctl enable-linger "$HIVE_USER"
if command -v pacman >/dev/null 2>&1; then
  run pacman -S --needed --noconfirm rootlesskit slirp4netns fuse-overlayfs
else
  echo "Install rootlesskit, slirp4netns and fuse-overlayfs with your package manager." >&2
fi

# Copy the committed application tree.  The deployment is read-only for the
# AI user after the optional install; mutable state lives in $HIVE_BASE.
run install -d -o root -g "$HIVE_GROUP" -m 0750 "$HIVE_BASE/app"
run install -d -o root -g "$HIVE_GROUP" -m 0750 "$HIVE_BASE/venv"
if [[ "$DRY_RUN" -eq 0 ]]; then
  find "$HIVE_BASE/app" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi
run_shell "git -C '$REPO' archive --format=tar HEAD | tar -x -C '$HIVE_BASE/app'"

if [[ "$WITH_VENV" -eq 1 ]]; then
  if [[ ! -x "$HIVE_BASE/venv/bin/python" ]]; then
    # pip needs a writable window inside the venv; lock it again below.
    if [[ "$DRY_RUN" -eq 0 ]]; then
      chown "$HIVE_USER:$HIVE_GROUP" "$HIVE_BASE/venv"
      chmod 0700 "$HIVE_BASE/venv"
    fi
    run runuser -u "$HIVE_USER" -- env HOME="$HIVE_BASE/home" \
      python -m venv "$HIVE_BASE/venv"
    run runuser -u "$HIVE_USER" -- env HOME="$HIVE_BASE/home" \
      "$HIVE_BASE/venv/bin/python" -m pip install --upgrade pip
    # The editable install needs a short writable window; lock the tree again
    # immediately after it so the agent cannot rewrite its own application.
    if [[ "$DRY_RUN" -eq 0 ]]; then
      chown -R "$HIVE_USER:$HIVE_GROUP" "$HIVE_BASE/app"
    fi
    run runuser -u "$HIVE_USER" -- env HOME="$HIVE_BASE/home" \
      "$HIVE_BASE/venv/bin/pip" install "$HIVE_BASE/app[studio]"
  fi
  # The Studio imports the absorbed splinter project (cortex/sieve/backend).
  # Install it into the root-owned venv as root so a pre-existing venv is
  # upgraded too; the app tree is read-only by this point.
  run "$HIVE_BASE/venv/bin/pip" install "$HIVE_BASE/app/splinter"
fi
if [[ "$DRY_RUN" -eq 0 ]]; then
  chown -R root:"$HIVE_GROUP" "$HIVE_BASE/app" "$HIVE_BASE/venv"
  chmod -R g+rX,o-rwx "$HIVE_BASE/app" "$HIVE_BASE/venv"
fi

# Keep the service environment outside the AI-writable tree.  A root-owned
# file inside $HIVE_BASE could still be unlinked and replaced, because the
# directory belongs to the AI user.
run install -d -o root -g root -m 0755 /etc/hive
# Preserve operator-added DSH/model settings; only DOCKER_HOST is owned here.
ENV_TMP="/tmp/hive.env.$$"
if [[ -f /etc/hive/hive.env ]]; then
  grep -v '^DOCKER_HOST=' /etc/hive/hive.env > "$ENV_TMP" || true
else
  : > "$ENV_TMP"
fi
printf 'DOCKER_HOST=unix:///run/user/%s/docker.sock\n' "$HIVE_UID" >> "$ENV_TMP"
run install -o root -g root -m 0600 "$ENV_TMP" /etc/hive/hive.env
rm -f "$ENV_TMP"

UNIT_INSTALLED=0
if [[ "$HIVE_USER" == "hive-ai" && "$HIVE_BASE" == "/srv/hive-ai" ]]; then
  run install -o root -g root -m 0644 "$REPO/deploy/hive.service" \
    /etc/systemd/system/hive.service
  run systemctl daemon-reload
  UNIT_INSTALLED=1
else
  echo "Custom user/base: copy deploy/hive.service and adjust User/WorkingDirectory." >&2
fi

ROOTLESS_SETUP="$(command -v dockerd-rootless-setuptool.sh 2>/dev/null || true)"
cat <<EOF

Prepared:
  user        $HIVE_USER (uid $HIVE_UID)
  app         $HIVE_BASE/app
  state       $HIVE_BASE/state
  DSH home    $HIVE_BASE/dsh
  env         /etc/hive/hive.env
  unit        $([[ "$UNIT_INSTALLED" -eq 1 ]] && echo '/etc/systemd/system/hive.service' || echo 'not installed (custom user/base)')

Next, set up the AI user's OWN rootless Docker daemon. Do not add $HIVE_USER
to the docker group and do not point DOCKER_HOST at /var/run/docker.sock.

EOF
if [[ -n "$ROOTLESS_SETUP" ]]; then
  cat <<EOF
  sudo -u $HIVE_USER -H env \\
    HOME=$HIVE_BASE/home \\
    XDG_RUNTIME_DIR=/run/user/$HIVE_UID \\
    DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$HIVE_UID/bus \\
    $ROOTLESS_SETUP install

EOF
else
  cat <<EOF
  sudo -u $HIVE_USER -H env \\
    HOME=$HIVE_BASE/home \\
    XDG_RUNTIME_DIR=/run/user/$HIVE_UID \\
    DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$HIVE_UID/bus \\
    bash -c 'curl -fsSL https://get.docker.com/rootless | sh'

EOF
fi
cat <<EOF
Verify from the same environment:

  sudo -u $HIVE_USER -H env \\
    HOME=$HIVE_BASE/home \\
    XDG_RUNTIME_DIR=/run/user/$HIVE_UID \\
    DOCKER_HOST=unix:///run/user/$HIVE_UID/docker.sock \\
    docker info

If the installer did not start the user service, start it as the AI user:

  sudo -u $HIVE_USER -H env \\
    HOME=$HIVE_BASE/home \\
    XDG_RUNTIME_DIR=/run/user/$HIVE_UID \\
    DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$HIVE_UID/bus \\
    systemctl --user enable --now docker

Then run the destructive blast-radius test:

  sudo ./scripts/test-blast-radius.sh --yes

Then start Hive:

  sudo systemctl enable --now hive.service
  curl http://127.0.0.1:8765/health
  curl -s http://127.0.0.1:8765/v1/sandbox/status | python3 -m json.tool

The human system Docker daemon, if wanted, is separate:
  sudo systemctl enable --now docker
  # Do not add $HIVE_USER to the docker group.
EOF
