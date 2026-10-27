# Sandboxed subagents (bubblewrap)

This bundle gives Hive one `subagent` tool whose children run in a separate DSH
agent preset and a hardened bubblewrap sandbox.  It is the smallest safe
delegation path: DSH's sandbox seam is built for confined executors, while a
Docker provider would have to replace the whole `ctx.shell` and `ctx.fs`
capabilities.  The bwrap path already gives per-session policy and per-call
`cwd`; this bundle adds network, path, and workspace isolation on top.

## Files

- `hive_bwrap.py` - the child-only runner.  It receives the provider's bwrap
  profile, then adds private user/network/IPC/UTS/PID namespaces, drops
  capabilities, disables nested user namespaces, masks host paths, and rebinds
  only the child's own workspace.
- `cordis.patch.yml` - the profile block: the preset registry, the
  `hive-sandbox` preset, the subagent service/provider, and a new
  `hive-subagent-tool` row with `maxDepth: 0`.
- `install.sh` - root installer.  It is idempotent: refreshes the runner,
  creates `/srv/hive-ai/workspaces`, runs a smoke test as `hive-ai`, installs
  the workspace rule into the live hook, appends the profile block once, and
  restarts `hive.service`.  The hook replacement is checksum-guarded: the
  installer refuses if the live `/srv/hive-ai/app/harness/tool_hook.py` is not
  the expected pre-change file, and backs it up to
  `tool_hook.py.pre-subagent-sandbox` before writing.

## Install

```bash
sudo ./deploy/hive-sandbox/install.sh
```

The profile backup is `/srv/hive-ai/dsh/cordis.patch.yml.pre-subagent-sandbox`.

## What the child can see

The child gets its own scratch directory under `/srv/hive-ai/workspaces`, a
fresh `/tmp`, `/proc`, and `/dev`, and a read-only host root with these paths
replaced by empty tmpfs mounts:

`/home`, `/root`, `/run`, `/media`, `/mnt`, `/run/media`, `/var/tmp`,
`/etc/hive`, `/srv/hive`, `/srv/hive-engine`, and the Hive app, bin, DSH
profile/session, home, logs, runs, state, tools, transcripts, harness-state,
and `.dsh-home` trees.

Sibling workspaces are masked too; the child's own workspace is re-created and
bound back.  The provider's policy bind is the parent's permission root
(`/srv/hive-ai`); the runner removes that broader bind, requires the command
cwd under `/srv/hive-ai/workspaces`, and binds only that cwd.  A bash call whose
`workdir` leaves the workspaces root fails closed.  The network namespace has
no route, so loopback services (the engine, Hive, the rootless Docker daemon)
are unreachable.  `HOME` and `TMPDIR` point at the fresh `/tmp`.

The child preset mounts `bash` only.  `dsh-fs-sandbox` fences writes but not
reads, so a `read`/`write`/`edit` tool would bypass the bwrap masks; bash is
the only file surface.  The preset also omits web, subagent, goal, todo, and
plan tools.  `maxDepth: 1` permits the parent to create one child; the child
has no `subagent` tool, so it cannot recurse.

## Workspace rule

The parent creates `/srv/hive-ai/workspaces/<name>` with its own bash, then
calls:

```text
subagent(prompt="...", cwd="/srv/hive-ai/workspaces/<name>")
```

`harness/tool_hook.py` refuses a missing, relative-outside, root, or
non-existent cwd before the approval gateway.  The child's session cwd is its
`workspace-write` root, so the workspace rule and the bwrap bind agree.

## Cease and cleanup

Cease kills the `hive.service` cgroup and bwrap's `--die-with-parent` kills the
command.  This path has no Docker container and no `hive.session` label; the
rootless Docker `/v1/sandbox/exec` path is unchanged.  Workspaces persist for
audit; remove `/srv/hive-ai/workspaces/<name>` when done.

## Verification

Unit tests:

```bash
.venv/bin/python -m pytest tests/unit/test_hive_sandbox.py tests/unit/test_tool_hook.py -q
```

The installer's smoke test runs the exact runner as `hive-ai` and checks a
workspace write, a masked home, state, and runtime socket, and no loopback.
After install, a live Hive turn should show the `subagent` tool and a child
that can write only its workspace.

## Boundary

This is same-world bubblewrap, not a container.  It removes the file and
network paths above but still shares the host kernel.  For untrusted code, use
the rootless Docker path with gVisor or Kata, or a dedicated host.
