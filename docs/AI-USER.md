# Dedicated AI user and rootless Docker

**Status:** deployment pattern, 2026-10-26 · `deploy/hive.service` ·
`scripts/setup-hive-account.sh`

Hive can run as a dedicated Unix user (`hive`) instead of the human account.
That keeps the agent’s state, credentials, and container runtime away from the
human’s files, shell, and Docker daemon. The systemd unit runs the **entire
Hive application** as `hive`: Studio, DSH runtime, agent tools, the optional
browser subprocess, the sandbox client, and the controller. The only separate
process is the AI user’s own rootless Docker daemon.

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

Run the setup script from the repository root with sudo:

```bash
cd /path/to/hivebench
sudo ./scripts/setup-hive-account.sh --dry-run   # inspect the plan
sudo ./scripts/setup-hive-account.sh
```

If `sudo` says `command not found`, you are not in the repository root or
you are on a branch without the script. Use the absolute path instead:

```bash
sudo /absolute/path/hivebench/scripts/setup-hive-account.sh --dry-run
# or, if the checkout is mounted noexec:
sudo bash /absolute/path/hivebench/scripts/setup-hive-account.sh --dry-run
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

The unit runs as `hive`, binds only `127.0.0.1`, makes `/home` read-only, and
makes only `/srv/hive` writable. `ProtectHome=read-only` is deliberate: `yes`
would also hide `/run/user`, which is where the rootless Docker socket lives.
The human home remains unreadable because it is `0700`/`0750` and `hive` is not
in its group.

`hive sandbox status` reports the daemon host and whether it is a rootless
per-user socket. A human running the command against their own state will not
see the AI user’s state; use the web Studio, or run the CLI in the AI
environment.

## Blast-radius test

After the account and rootless daemon are up, run the destructive sentinel test
as root:

```bash
sudo ./scripts/test-blast-radius.sh --yes
```

It creates disposable files in the human home, the read-only app tree, the AI
state tree, and host `/tmp`, then runs a destructive shell as `hive` inside a
transient systemd unit with the same `ProtectSystem`, `ProtectHome`,
`PrivateTmp`, capability, and writable-path settings as `hive.service`. This
is the approved-worst-case test: it assumes the agent was allowed to run the
command, so the remaining boundary is the OS user and unit hardening.

Expected result:

- the human, app, and host `/tmp` sentinels survive;
- the AI state sentinel is removed;
- `docker --host unix:///var/run/docker.sock` is refused;
- the rootless Docker socket is reachable once rootless Docker is running.

The test never deletes a real file; the only AI file it removes is its own
sentinel.

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

## Kernel boundary

Rootless Docker and the `hive` user bound **application** damage: a destructive
command can only remove files the AI user can write, and cannot touch the
human home, the read-only app tree, the host `/tmp`, or the human Docker
socket. They do **not** protect against a host-kernel exploit: the AI’s
containers and the Hive process still share the host kernel.

Without a dedicated VM, the realistic options are:

1. **Separate physical host.** Strongest hard boundary without virtualization;
   the AI simply has no path to the human machine.
2. **gVisor (`runsc`).** A user-space application kernel intercepts container
   syscalls and reduces host-kernel attack surface. It is not a formal
   guarantee and rootless/GPU compatibility must be tested, but it is much
   stronger than plain `runc`.
3. **Kata Containers / Firecracker.** Per-workload microVMs. They are
   virtualization, but not a dedicated full VM host; they give a separate
   kernel per workload.
4. **Host hardening without a VM.** Seccomp, LSM/AppArmor, patched/minimal
   kernel, no unneeded modules, and the systemd restrictions above reduce the
   attack surface. They do not eliminate a kernel bug.

Hive can select a Docker runtime without code changes:

```bash
hive sandbox policy --runtime runsc        # gVisor
hive sandbox policy --runtime kata-runtime # Kata
hive sandbox status                        # shows the selected runtime
```

The runtime must be installed and registered with the AI user’s Docker daemon;
`runsc` and Kata are not installed by this repository. The default empty value
uses the daemon’s default (`runc`).

For a truly adversarial AI, treat the Hive host as expendable and put it on a
separate physical machine or a microVM. The app/user split is a large
improvement, not a proof against kernel exploits.

## Security notes

- The AI user’s home and state are not a secret store. Do not put human SSH
  keys, browser profiles, or cloud credentials there.
- The Studio binds localhost. If you expose it beyond localhost, set
  `HARNESS_TOKEN` and use a reverse proxy with TLS.
- Keep the AI app copy read-only; the agent should not be able to rewrite the
  harness that is enforcing its policy.
