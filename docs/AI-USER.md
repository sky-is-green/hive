# Dedicated AI user and rootless Docker

**Status:** deployment pattern, 2026-10-26 · `deploy/hive.service` ·
`scripts/setup-hive-account.sh`

Hive runs as a dedicated, disposable Unix user (`hive-ai`) instead of the human
account. That keeps the agent’s state, credentials, and container runtime away
from the human’s files, shell, and Docker daemon. The systemd unit runs the
**entire Hive application** as `hive-ai`: Studio, DSH runtime, agent tools, the
optional browser subprocess, the sandbox client, and the controller. The only
separate process is the AI user’s own rootless Docker daemon.

The important rule: **do not add `hive-ai` to the `docker` group and do not
point it at `/var/run/docker.sock`.** That would give the AI control over every
human container. The AI user gets its own **rootless Docker daemon**, socket,
data root, and image store.

## Why not the existing `hive` account?

`hive` already belongs to GUARD:

```text
guard.service -> /usr/bin/python3 /home/penis/.guard/guard-daemon.py
User=hive   Group=hiveq   HOME=/home/penis
```

`/home/penis` grants `user:hive:r-x`, and `/home/penis/.guard` is owned
`hive:hiveq`. Running Hive as `hive` would give the agent the same UID as
GUARD, including its queue, approvals, and allowlist. The deployment therefore
uses a separate `hive-ai` account and leaves GUARD, its `hive` account, and
`/home/penis/.guard` untouched.

## Layout

```text
/srv/hive-ai/            root:hive-ai, 0750
  app/          read-only application copy (root:hive-ai, 0750)
  venv/         read-only Python environment (root:hive-ai, 0750)
  state/        HIVE_STATE_DIR, settings, audit, grants, controller traces
  dsh/          DSH_HOME, profiles, sessions, hook configuration
  home/         account HOME
  models/       HIVE_HOME/models -> model library
  stacks/       HIVE_HOME/stacks
  logs/
  tools/        backends and mutable tool data
```

The service environment lives outside the deployment tree at
`/etc/hive/hive.env` (root-owned, `0600`), so a future permission mistake in
`/srv/hive-ai` cannot let the AI replace the service configuration. The venv
contains both `hivebench[studio]` and the absorbed `splinter-memory` project,
because the Studio imports `cortex`, `sieve`, and `backend` from it.

`HIVE_HOME=/srv/hive-ai`, `HIVE_STATE_DIR=/srv/hive-ai/state`,
`DSH_HOME=/srv/hive-ai/dsh`, and
`HIVE_DSH_SESSION_ROOT=/srv/hive-ai/dsh/sessions`. The unit also passes
absolute `--state-dir`, `--log-dir`, and `--providers-file` paths: the app copy
is read-only, so the CLI's relative defaults would try to write inside it. The
base directory, app copy, and venv are root-owned and read-only to the AI user
after installation, so the agent cannot rename the app or rewrite its own
harness code. The writable state directories (`home`, `state`, `dsh`, `models`,
`stacks`, `logs`, `tools`) are owned by `hive-ai`.

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

It creates the `hive-ai` account if needed, `/srv/hive-ai`, subuid/subgid
entries, linger, the read-only application copy, and the systemd unit. It does
**not** start Docker, add `hive-ai` to the `docker` group, or modify GUARD.

The script deliberately does not call `usermod -d`. GUARD’s process already
runs as `hive`, and changing a home while the account is in use fails. The unit
sets `HOME=/srv/hive-ai/home` explicitly, so the passwd home is informational.

The script also prints the rootless Docker step. Run it as the AI user, not as
root:

```bash
HIVE_UID="$(id -u hive-ai)"
sudo -u hive-ai -H env \
  HOME=/srv/hive-ai/home \
  XDG_RUNTIME_DIR=/run/user/$HIVE_UID \
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$HIVE_UID/bus \
  bash -c 'curl -fsSL https://get.docker.com/rootless | sh'
```

If `dockerd-rootless-setuptool.sh` is installed, the setup script prints that
command instead.

Then verify the AI socket:

```bash
HIVE_UID="$(id -u hive-ai)"
sudo -u hive-ai -H env \
  HOME=/srv/hive-ai/home \
  XDG_RUNTIME_DIR=/run/user/$HIVE_UID \
  DOCKER_HOST=unix:///run/user/$HIVE_UID/docker.sock \
  docker info
```

If the installer did not start the user service:

```bash
sudo -u hive-ai -H env \
  HOME=/srv/hive-ai/home \
  XDG_RUNTIME_DIR=/run/user/$HIVE_UID \
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$HIVE_UID/bus \
  systemctl --user enable --now docker
```

`/etc/hive/hive.env` should contain the AI UID, not a shared socket:

```bash
DOCKER_HOST=unix:///run/user/<hive-ai-uid>/docker.sock
```

## Start Hive

```bash
sudo systemctl enable --now hive.service
curl http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/v1/sandbox/status | python3 -m json.tool
```

The unit runs as `hive-ai`, binds only `127.0.0.1`, and makes only
`/srv/hive-ai` writable at the mount level. The base, `app/`, and `venv/` are
root-owned, so `hive-ai` can write only its state subdirectories.
`InaccessiblePaths=/home -/root` hides the human and
root homes, and `ProtectHome=no` is deliberate: `yes` or `read-only` would also
hide or freeze `/run/user`, where the rootless Docker socket lives. `%U` is not
expanded inside `ReadWritePaths`, so the unit cannot name
`/run/user/<uid>` directly. The per-user runtime directory is a tmpfs that
stays writable under `ProtectSystem=strict`; the blast-radius test checks that
`hive-ai` can still write its own socket directory.

`hive sandbox status` reports the daemon host and whether it is a rootless
per-user socket. A human running the command against their own state will not
see the AI user’s state; use the web Studio, or run the CLI in the AI
environment.

## Local DSH runtime and OpenAI-compatible route

The unit runs the Studio, but the DSH agent runtime and its approval-hook
plugin are separate Node installs under `/srv/hive-ai/tools/dsh`. Install
them as the AI user, not as root:

```bash
HIVE_UID="$(id -u hive-ai)"
sudo -u hive-ai -H env HOME=/srv/hive-ai/home \
  npm install --prefix /srv/hive-ai/tools/dsh \
  @deepseek-ai/dsh@0.2.1-rc.2 \
  @deepseek-ai/dsh-hooks-claude-code@0.2.1-rc.2
```

If npm 12 reports blocked install scripts, approve the five DSH helpers
(`@deepseek-ai/dsh-subprocess-local`, `koffi`, `node-pty`, `@google/genai`,
`protobufjs`), run `npm rebuild`, then restore `root:hive-ai` ownership and
`g+rX,o-rwx` modes.

The profile must be able to resolve the plugin, and a local llama.cpp server
must use DSH’s OpenAI-compatible `pi-ai` route rather than the DeepSeek
Messages adapter:

```bash
sudo install -d -o hive-ai -g hive-ai -m 0755 \
  /srv/hive-ai/dsh/profiles/sdk/node_modules
sudo ln -sfn /srv/hive-ai/tools/dsh/node_modules/@deepseek-ai \
  /srv/hive-ai/dsh/profiles/sdk/node_modules/@deepseek-ai
```

Append this route to `/srv/hive-ai/dsh/cordis.patch.yml` (the hook installer
adds its own `hooks-claude-code` row separately):

```yaml
- insert:
    - id: llm-pi-ai
      name: '@deepseek-ai/dsh-llm-pi-ai'
      config:
        providers:
          hive-local:
            apiKeyEnv: HIVE_LOCAL_API_KEY
            api: openai-completions
            baseURL: http://127.0.0.1:8150/v1
            compat:
              thinkingFormat: deepseek
            models:
              - id: deepseek-v4.1-flash-gsq
                name: DeepSeek-V4.1-Flash-GSQ
                contextWindow: 32768
                maxTokens: 8192
```

`HIVE_LOCAL_API_KEY=lm-studio` is a local, non-secret placeholder; do not use
a human or cloud key here. The unit sets `HIVE_DSH_PROVIDER=hive-local` and
`HIVE_DSH_MODEL=deepseek-v4.1-flash-gsq`; the same values are pinned in
`/etc/systemd/system/hive.service.d/ds41.conf` so a setup rerun cannot revert
them. The provider/engine files stay under `/srv/hive-ai/state` so a
read-only app tree never blocks persistence. Point `HARNESS_LLAMA_SERVER` at
the AI-readable llama.cpp binary and put the GGUF under
`/srv/hive-ai/models/gguf` for the Qwen route; the DeepSeek-V4.1 route uses the
Strata engine service below instead.

### DS4.1 engine service (single spare GPU)

The local heavy tier is a durable system service, not a transient user
process:

- Unit: `/etc/systemd/system/ds41-engine.service`, enabled at boot, runs as
  the dedicated `hive-engine` account, `MemoryMax=12G`, `MemoryHigh=9G`,
  `ProtectSystem=strict`, `ProtectHome=yes`, `InaccessiblePaths=/home`.
- Config: `/etc/hive-engine/ds41.json`, mode 0640 root:hive-engine.  It binds
  the 152 GB V41-GSQ model and the two Engram layers read-only into
  `/srv/hive-engine/model` and `/srv/hive-engine/engram`.  The service does
  **not** see `/run/media/penis` or `/home`; only those three mounts.
- GPU: `HIP_VISIBLE_DEVICES=1` (the spare card, PCI 0000:07:00.0); the display
  card stays with the desktop.  `hive-engine` is **not** in the docker group
  and has no docker socket.
- API: `http://127.0.0.1:8150/v1`, OpenAI-compatible, model id
  `deepseek-v4.1-flash-gsq`, context 32768.  `hive.service` has
  `Wants=`/`After=ds41-engine.service` via
  `/etc/systemd/system/hive.service.d/ds41.conf`.
- Tuned 2026-10-27 for this 20 GB / 30 GiB box: `--prefill-chunk 1024`,
  `STRATA_SERVE_STEP_MAX=0`, `--prompt-batch 8192`, `--arena-gib 4`,
  `--pf-b 0`, `STRATA_PRESTAGE_GIB=1.5`, `STRATA_WAVE_THREADS=16`,
  `STRATA_WAVE_PARTS=16`, `STRATA_D3_PF=1` (the neutral default), and
  `STRATA_SERVE_DSPARK` unset.  The exact **pass** path is only ~2.3 tok/s
  with a 4 GiB arena, so every prompt uses the layer-major batch: a 341-token
  turn is 21-24 s (was 128 s), a 648-token turn 23 s (was 255 s), and a
  73-token turn 18 s (was 35 s).  The 6.1k prompt is 51.7-55 s
  (121-127 tok/s); decode at 6.1k is 16 tok/s, and a 16-token follow-up is
  7.6 s with the 6,115-token prefix reused.
- `--prefill-chunk 1024` is the maximum that handles the 6.1k prompt on this
  card: 2048 returns 503 and 4096 fails a 17.5 GiB dense-graph allocation.
  The wave reader (`STRATA_WAVE_THREADS=16`, `STRATA_WAVE_PARTS=16`) gave
  120.8 / 121.3 / 121.2 tok/s for 16/8, 16/16 and 32/16 over a 117.8
  baseline; the final confirmation was 126.5 tok/s.  `--pf-b 1.43` regressed
  to 99.3 tok/s prefill / 11.2 decode; `STEP_MAX=20` helped the 16-token
  follow-up but dropped the 128-token decode to 12.4 tok/s; D3 off was
  neutral.  Those were not applied.
- Verified: a 6,110-token direct prefill at 126.5 tok/s, and a live Hive turn
  (`DS41_HIVE_OK`) on the trimmed profile: 2,682 prompt tokens in 59 s,
  129 thinking tokens at 20.9 tok/s, a tool call/result, and a follow-up turn
  that reused the parked prefix in 5 s and answered 92 tokens at 17.8 tok/s.
- Prompt trim (2026-10-27): the Hive profile disables `tool-working-directory`,
  `tool-goal`, `tool-subagent-control`, `tool-subagent-list-agents`,
  `tool-subagent`, `tool-subagent-fork`, `tool-workflow`, `tool-todo`,
  `plan-mode`, and `tool-web`.  It keeps `bash`, `edit`, `glob`, `grep`,
  `job_kill`, `job_list`, `job_output`, `read`, `read_image`, `skill`, and
  `write`: 25 tools -> 11, system prompt 2,571 -> 1,204 chars, Hive prompt
  5,734 -> 2,682 tokens.  The first turn is 77 -> 68 s (59 s prefill); the
  follow-up prompt is 2,828 tokens.  Backup:
  `/srv/hive-ai/dsh/cordis.patch.yml.pre-trim`.
- Previous config backups: `/etc/hive-engine/ds41.json.pre-tune2`,
  `/etc/hive-engine/ds41.json.pre-chunk1024`, and
  `/etc/hive-engine/ds41.json.pre-tune`.  The service log is in the journal;
  the engine's own log is `/srv/hive-engine/logs/engine.log`.

## Blast-radius test

After the account and rootless daemon are up, run the destructive sentinel test
as root:

```bash
sudo ./scripts/test-blast-radius.sh --yes
```

It creates disposable files in the human home, the read-only app tree, the AI
state tree, and host `/tmp`, then runs a destructive shell as `hive-ai` inside a
transient systemd unit with the same `ProtectSystem`, `ProtectHome`,
`InaccessiblePaths`, `PrivateTmp`, capability, and writable-path settings as
`hive.service`. This is the approved-worst-case test: it assumes the agent was
allowed to run the command, so the remaining boundary is the OS user and unit
hardening.

Expected result:

- the human, app, and host `/tmp` sentinels survive and the human one cannot be
  read;
- the AI state sentinel is removed;
- `docker --host unix:///var/run/docker.sock` is refused;
- the rootless Docker socket is reachable once rootless Docker is running;
- if `/home/penis/.guard` exists, `hive-ai` cannot list or write it.

The test never deletes a real file; the only AI file it removes is its own
sentinel.

## Human Docker remains separate

If you want the human Docker daemon too:

```bash
sudo systemctl enable --now docker
```

Do **not** add `hive-ai` to the `docker` group. The AI cannot see or control the
human daemon, containers, or images through its rootless socket.

## Sharing models

Put models where `hive-ai` can read them, preferably through the AI’s own model
library:

```bash
sudo -u hive-ai -H env HOME=/srv/hive-ai/home \
  /srv/hive-ai/venv/bin/hive models download <repo>
```

If a human downloads models, copy them into `/srv/hive-ai/models/gguf` with
explicit ownership. Do not make `/srv/hive-ai/state` or `/srv/hive-ai/dsh`
human group-writable; those contain grants, audit, and session state.

## Kernel boundary

Rootless Docker and the `hive-ai` user bound **application** damage: a
destructive command can only remove files the AI user can write, and cannot
touch the human home, the read-only app tree, the host `/tmp`, or the human
Docker socket. They do **not** protect against a host-kernel exploit: the AI’s
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
- Keep the AI app copy and venv root-owned and read-only; the parent directory
  must also be root-owned, or the agent could rename the app and replace it.
- Do not reuse GUARD’s `hive` account for Hive. The setup script refuses
  `HIVE_USER=hive` for exactly that reason.
- The old partial `/srv/hive` tree from the failed setup is left untouched.
  Remove it only after confirming nothing uses it.
