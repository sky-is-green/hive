# Sandboxed delegation (Docker)

This bundle gives Hive one synchronous `delegate` tool whose children run in a
separate DSH agent preset with only a Docker-backed `bash` tool.  `delegate`
starts the child, waits for its final answer, and returns that answer as the
tool result; there is no asynchronous child id to poll.  The parent keeps its
base sandbox and full tool set; the child gets its own workspace mounted at
`/workspace` in the pinned `hive-sandbox:1` image.

## Files

- `delegate/` - `@deepseek-ai/hive-delegate`, the parent's waiting delegation
  tool.  It calls `ctx.subagents.startActivation` with `delivery: 'caller'`,
  awaits the result, returns the child's final text, and disposes the
  activation.
- `plugin/` - `@deepseek-ai/hive-docker-bash`, a child-only DSH tool that sends
  each command to `/v1/sandbox/exec` with the child's workspace and session id.
- `cordis.patch.yml` - the profile block: the preset registry, the
  `hive-sandbox` preset, the subagent service/provider, and a new
  `hive-delegate-tool` row with `maxDepth: 1`.
- `install.sh` - root installer.  It checksum-guards and updates the live
  `harness/app.py`, `harness/sandbox.py`, `harness/security.py`, and
  `harness/tool_hook.py`, installs both plugin packages, replaces the profile
  block, restarts `hive.service`, and runs a Docker workspace smoke test.


## Install

From the hivebench checkout:

```bash
sudo bash deploy/hive-sandbox/install.sh
```

Backups:

- `/srv/hive-ai/dsh/cordis.patch.yml.pre-docker-sandbox`
- `/srv/hive-ai/app/harness/app.py.pre-docker-sandbox`
- `/srv/hive-ai/app/harness/sandbox.py.pre-docker-sandbox`
- `/srv/hive-ai/app/harness/security.py.pre-docker-sandbox`
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
`grep`, web, subagent, delegate, goal, todo, and plan tools.  `maxDepth: 1`
lets the parent create one child; the child has no delegation tool, so it
cannot recurse.

## Workspace rule

The parent creates `/srv/hive-ai/workspaces/<name>` with its own bash, then
calls:

```text
delegate(prompt="...", cwd="/srv/hive-ai/workspaces/<name>")
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
container's `/workspace` output.  The 2026-10-28 live `delegate` proof started
child `2d1ce915-a9a8-4808-b97b-4c3f843a88f6`; the parent tool waited and
returned `/workspace`, `CHILD_OK`, `STATE_ABSENT`, `SOCKET_ABSENT`,
`HOME_ABSENT`, `SIBLING_ABSENT`, `NET_BLOCKED`, and `CHILD_DONE` with exit 0.
The parent's turn finished `completed` with no `job_output`, no duplicate
child, and no completion-notice wait.

A live cease test killed a `sleep 60` container in about three seconds: the
sandbox request returned exit 137 with `timed_out: false`, and
`actions.stop.containers.killed` named the container.

The gateway still refuses `job_output`/`job_kill` on a known child id, for a
stray call or the disabled asynchronous path; real background jobs follow the
normal approval path.

## Boundary

This is rootless Docker, not a dedicated VM.  It removes the host filesystem
and network paths above, but the container still shares the host kernel.  For
untrusted code, use gVisor or Kata (see `docs/AI-USER.md`), or a dedicated host.
