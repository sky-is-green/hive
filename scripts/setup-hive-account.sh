#!/usr/bin/env bash
# Prepare the dedicated Hive AI account and /srv/hive layout.
#
# This script is intentionally root-only and deliberately does NOT:
#   - add the AI user to the docker group;
#   - start or modify the human system Docker daemon;
#   - install or start a rootless Docker daemon automatically.
#
# Run it with sudo, review the printed rootless-Docker step, then start Hive.
set -euo pipefail

HIVE_USER="${HIVE_USER:-hive}"
HIVE_BASE="${HIVE_BASE:-/srv/hive}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DRY_RUN=0
WITH_VENV=1

usage() {
  cat <<'EOF'
Usage: sudo ./scripts/setup-hive-account.sh [--dry-run] [--no-venv] [REPO]

Creates/updates the dedicated Hive user and /srv/hive layout, copies the
tracked application tree, and installs the optional Studio environment.
It never adds the user to the docker group and never starts system Docker.
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
if [[ ! -f "$REPO/pyproject.toml" ]]; then
  echo "repository not found: $REPO" >&2
  exit 1
fi
if ! id "$HIVE_USER" >/dev/null 2>&1; then
  if ! command -v useradd >/dev/null 2>&1; then
    echo "user '$HIVE_USER' does not exist and useradd is unavailable" >&2
    exit 1
  fi
  run useradd --system --user-group --home-dir "$HIVE_BASE/home" \
    --shell /usr/sbin/nologin "$HIVE_USER"
fi
HIVE_GROUP="$(id -gn "$HIVE_USER")"
HIVE_UID="$(id -u "$HIVE_USER")"

# Account home and the per-user writable tree.
run install -d -o "$HIVE_USER" -g "$HIVE_GROUP" -m 0700 "$HIVE_BASE"
for directory in home state dsh models stacks logs tools; do
  run install -d -o "$HIVE_USER" -g "$HIVE_GROUP" -m 0700 "$HIVE_BASE/$directory"
done
run usermod -d "$HIVE_BASE/home" "$HIVE_USER"

# Rootless container prerequisites.  The AI user must never join the system
# docker group; its own daemon gets its own socket and data root.
if ! grep -q "^${HIVE_USER}:" /etc/subuid 2>/dev/null; then
  run usermod --add-subuids 100000-165535 --add-subgids 100000-165535 "$HIVE_USER"
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
if [[ "$DRY_RUN" -eq 0 ]]; then
  find "$HIVE_BASE/app" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi
run_shell "git -C '$REPO' archive --format=tar HEAD | tar -x -C '$HIVE_BASE/app'"

if [[ "$WITH_VENV" -eq 1 ]]; then
  if [[ ! -x "$HIVE_BASE/venv/bin/python" ]]; then
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
fi
if [[ "$DRY_RUN" -eq 0 ]]; then
  chown -R root:"$HIVE_GROUP" "$HIVE_BASE/app"
  chmod -R g+rX,o-rwx "$HIVE_BASE/app"
fi

cat > "/tmp/hive.env.$$" <<EOF
DOCKER_HOST=unix:///run/user/${HIVE_UID}/docker.sock
EOF
run install -o root -g "$HIVE_GROUP" -m 0640 "/tmp/hive.env.$$" "$HIVE_BASE/hive.env"
rm -f "/tmp/hive.env.$$"

if [[ "$HIVE_USER" == "hive" && "$HIVE_BASE" == "/srv/hive" ]]; then
  run install -o root -g root -m 0644 "$REPO/deploy/hive.service" \
    /etc/systemd/system/hive.service
  run systemctl daemon-reload
else
  echo "Custom user/base: copy deploy/hive.service and adjust User/WorkingDirectory." >&2
fi

cat <<EOF

Prepared:
  user        $HIVE_USER (uid $HIVE_UID)
  app         $HIVE_BASE/app
  state       $HIVE_BASE/state
  DSH home    $HIVE_BASE/dsh
  env         $HIVE_BASE/hive.env
  unit        /etc/systemd/system/hive.service (defaults only)

Next, set up the AI user's OWN rootless Docker daemon. Do not add $HIVE_USER
to the docker group and do not point DOCKER_HOST at /var/run/docker.sock.

  sudo -u $HIVE_USER -H env \\
    HOME=$HIVE_BASE/home \\
    XDG_RUNTIME_DIR=/run/user/$HIVE_UID \\
    DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$HIVE_UID/bus \\
    bash -c 'curl -fsSL https://get.docker.com/rootless | sh'

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

Then start Hive:

  sudo systemctl enable --now hive.service
  curl http://127.0.0.1:8765/health
  curl -s http://127.0.0.1:8765/v1/sandbox/status | python3 -m json.tool

The human system Docker daemon, if wanted, is separate:
  sudo systemctl enable --now docker
  # Do not add $HIVE_USER to the docker group.
EOF
