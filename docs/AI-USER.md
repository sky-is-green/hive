# Dedicated AI user and rootless Docker

**Status:** deployment pattern, 2026-10-26 · `deploy/hive.service` ·
`scripts/setup-hive-account.sh`

Hive can run as a dedicated Unix user (`hive`) instead of the human account.
That keeps the agent’s state, credentials, and container runtime away from the
human’s files, shell, and Docker daemon.

The important rule: **do not add `hive` to the `docker` group and do not point
it at `/var/run/docker.sock`.** That would give the AI control over every human
container. The AI user gets its own **rootless Docker daemon**, socket, data
root, and image store.

## Layout

```text
/srv/hive/
  app/          read-only application copy (root:hive, 0750)
  venv/         Python environment owned by hive
  state/        HIVE_STATE_DIR, settings, audit, grants, controller traces
  dsh/          DSH_HOME, profiles, sessions, hook configuration
  home/         account HOME
  models/       HIVE_HOME/models -> model library
  stacks/       HIVE_HOME/stacks
  logs/
  tools/        backends and mutable tool data
  hive.env      DOCKER_HOST and other environment
```

`HIVE_HOME=/srv/hive`, `HIVE_STATE_DIR=/srv/hive/state`, and
`DSH_HOME=/srv/hive/dsh`. The app copy is root-owned and read-only to the AI
user after installation, so the agent cannot rewrite its own harness code.

## Prepare the account

Run the setup script with sudo:

```bash
sudo scripts/setup-hive-account.sh --dry-run   # inspect the plan
sudo scripts/setup-hive-account.sh
```

It creates/updates the existing `hive` account, `/srv/hive`, subuid/subgid
entries, linger, and the read-only application copy. It does **not** start
Docker or add `hive` to the `docker` group.

The script also prints the rootless Docker step. Run it as the AI user, not as
root:

```bash
HIVE_UID=952   # id -u hive
sudo -u hive -H env \
  HOME=/srv/hive/home \
  XDG_RUNTIME_DIR=/run/user/$HIVE_UID \
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$HIVE_UID/bus \
  bash -c 'curl -fsSL https://get.docker.com/rootless | sh'
```

Then verify the AI socket:

```bash
sudo -u hive -H env \
  HOME=/srv/hive/home \
  XDG_RUNTIME_DIR=/run/user/$HIVE_UID \
  DOCKER_HOST=unix:///run/user/$HIVE_UID/docker.sock \
  docker info
```

If the installer did not start the user service:

```bash
sudo -u hive -H env \
  HOME=/srv/hive/home \
  XDG_RUNTIME_DIR=/run/user/$HIVE_UID \
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$HIVE_UID/bus \
  systemctl --user enable --now docker
```

`/srv/hive/hive.env` should contain:

```bash
DOCKER_HOST=unix:///run/user/952/docker.sock
```

## Start Hive

```bash
sudo systemctl enable --now hive.service
curl http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/v1/sandbox/status | python3 -m json.tool
```

The unit runs as `hive`, binds only `127.0.0.1`, hides `/home`, and makes only
`/srv/hive` writable. It uses the user’s rootless Docker socket, not the system
socket.

`hive sandbox status` now reports the daemon host and whether it is a rootless
per-user socket. A human running the command against their own state will not
see the AI user’s state; use the web Studio, or run the CLI in the AI
environment.

## Human Docker remains separate

If you want the human Docker daemon too:

```bash
sudo systemctl enable --now docker
```

Do **not** add `hive` to the `docker` group. The AI cannot see or control the
human daemon, containers, or images through its rootless socket.

## Sharing models

Put models where `hive` can read them, preferably through the AI’s own model
library:

```bash
sudo -u hive -H env HOME=/srv/hive/home \
  /srv/hive/venv/bin/hive models download <repo>
```

If a human downloads models, copy them into `/srv/hive/models/gguf` with
explicit ownership. Do not make `/srv/hive/state` or `/srv/hive/dsh` human
group-writable; those contain grants, audit, and session state.

## Security notes

- Rootless Docker uses user namespaces and shares the host kernel. It is a
  strong separation from the human Docker daemon, but it is not a VM. For
  hostile code, use a dedicated VM/microVM or separate machine.
- The AI user’s home and state are not a secret store. Do not put human SSH
  keys, browser profiles, or cloud credentials there.
- The Studio binds localhost. If you expose it beyond localhost, set
  `HARNESS_TOKEN` and use a reverse proxy with TLS.
- Keep the AI app copy read-only; the agent should not be able to rewrite the
  harness that is enforcing its policy.
