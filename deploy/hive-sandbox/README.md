# Sandboxed subagents (Docker)

This bundle gives Hive one `subagent` tool whose children run in a separate DSH
agent preset with only a Docker-backed `bash` tool.  The parent keeps its base
sandbox and full tool set; the child gets its own workspace mounted at
`/workspace` in the pinned `hive-sandbox:1` image.

## Files

- `plugin/` - `@deepseek-ai/hive-docker-bash`, a child-only DSH tool that sends
  each command to `/v1/sandbox/exec` with the child's workspace and session id.
- `cordis.patch.yml` - the profile block: the preset registry, the
  `hive-sandbox` preset, the subagent service/provider, and a new
  `hive-subagent-tool` row with `maxDepth: 1`.
- `install.sh` - root installer.  It checksum-guards and updates the live
  `harness/app.py`, `harness/sandbox.py`, and `harness/tool_hook.py`, installs
  the plugin package, replaces the profile block, restarts `hive.service`, and
  runs a Docker workspace smoke test.
- `hive_bwrap.py` - the earlier bwrap runner, retained as a fallback reference.
  The child does not use it: `hive.service`'s systemd sandbox blocks bwrap's
  mount path with `Function not implemented`, while the rootless Docker daemon
  runs outside that sandbox.

## Install

From the hivebench checkout:

```bash
sudo bash deploy/hive-sandbox/install.sh
```

Backups:

- `/srv/hive-ai/dsh/cordis.patch.yml.pre-docker-sandbox`
- `/srv/hive-ai/app/harness/app.py.pre-docker-sandbox`
- `/srv/hive-ai/app/harness/sandbox.py.pre-docker-sandbox`
- `/srv/hive-ai/app/harness/tool_hook.py.pre-docker-sandbox`

## What the child can see

The endpoint mounts only the validated child workspace at `/workspace` and runs
the pinned image with:

- `--network none`
- `--cap-drop ALL` and `--security-opt no-new-privileges`
- `--memory 512m`, `--cpus 1`, `--pids-limit 128`
- `hive.session=<child>` and `hive.parent=<parent>` labels
- no host root, `/home`, `/srv/hive-ai/state`, `/srv/hive-ai/dsh`, DSH runtime,
  GUARD, engine, or Docker socket mounts.

The child has `bash` only.  The preset omits `read`, `write`, `edit`, `glob`,
`grep`, web, subagent, goal, todo, and plan tools.  `maxDepth: 1` lets the parent
create one child; the child has no `subagent` tool, so it cannot recurse.

## Workspace rule

The parent creates `/srv/hive-ai/workspaces/<name>` with its own bash, then
calls:

```text
subagent(prompt="...", cwd="/srv/hive-ai/workspaces/<name>")
```

`harness/tool_hook.py` refuses a missing or outside cwd before the approval
gateway.  The child plugin uses the child's working directory, and
`sandbox.workspace_path()` refuses the root, a parent, a missing directory, a
symlink escape, and a path Docker would parse as a volume option.

## Cease and cleanup

`kill_session_containers` stops containers labelled `hive.session=<id>` or
`hive.parent=<id>`; the Hive cease path passes the parent session.  The child's
in-process agent dies with the DSH runtime, and its container is stopped by the
label.  Workspaces persist for audit; remove
`/srv/hive-ai/workspaces/<name>` when done.

## Verification

Unit tests:

```bash
.venv/bin/python -m pytest tests/unit/test_sandbox.py \
  tests/unit/test_hive_sandbox.py tests/unit/test_tool_hook.py -q
```

The installer's Docker smoke test mounts a real workspace and checks the
container's `/workspace` output.  A live proof should show the child returning
`DOCKER_SANDBOX_OK`-style markers and no host paths.

## Boundary

This is rootless Docker, not a dedicated VM.  It removes the host filesystem
and network paths above, but the container still shares the host kernel.  For
untrusted code, use gVisor or Kata (see `docs/AI-USER.md`), or a dedicated host.
