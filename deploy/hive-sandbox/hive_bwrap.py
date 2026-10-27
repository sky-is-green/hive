#!/usr/bin/python3
"""Bubblewrap runner for Hive's sandboxed subagents.

DSH's ``dsh-sandbox-local`` provider appends a bwrap-compatible profile to the
configured ``runnerCommand``.  This wrapper adds Hive's deployment policy:

* a private user, network, IPC, UTS, and PID namespace, with further user
  namespaces disabled;
* no capabilities, a fresh terminal session, and a fixed hostname;
* tmpfs masks over the host paths a subagent must not read (the human home,
  the Hive state and app trees, GUARD, the engine, the runtime socket dir);
* a private scratch workspace: sibling workspaces are masked and the child's
  own workspace is re-created and bound back.

The wrapper is installed root-owned under ``/srv/hive-ai/bin`` and is invoked
by the child preset only.  It never reads a path or executable from the
environment, so a model-controlled variable cannot change the boundary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BWRAP = "/usr/bin/bwrap"
WORKSPACE_ROOT = "/srv/hive-ai/workspaces"

#: Paths a subagent must not see.  Missing paths are skipped so the runner
#: still starts on a minimal host; the installer creates the important ones.
MASK_DIRS = (
    "/home",
    "/root",
    "/run",
    "/media",
    "/mnt",
    "/run/media",
    "/var/tmp",
    "/etc/hive",
    "/srv/hive",
    "/srv/hive-engine",
    "/srv/hive-ai/app",
    "/srv/hive-ai/bin",
    "/srv/hive-ai/dsh",
    "/srv/hive-ai/home",
    "/srv/hive-ai/logs",
    "/srv/hive-ai/runs",
    "/srv/hive-ai/state",
    "/srv/hive-ai/tools",
    "/srv/hive-ai/transcripts",
    "/srv/hive-ai/harness_state",
    "/srv/hive-ai/.dsh-home",
)

#: Namespace and process flags.  These must precede the provider profile so the
#: mounts are created inside the new namespaces.  ``--disable-userns`` requires
#: ``--unshare-user`` on this bubblewrap build.
NAMESPACE_ARGS = (
    "--unshare-user",
    "--disable-userns",
    "--unshare-net",
    "--unshare-ipc",
    "--unshare-uts",
    "--new-session",
    "--hostname",
    "hive-subagent",
    "--cap-drop",
    "ALL",
)


class ProfileError(ValueError):
    """The provider profile is not the shape this runner expects."""


def split_profile(args: list[str]) -> tuple[list[str], list[str]]:
    """Split the provider's bwrap profile from the command argv at ``--``."""
    try:
        separator = args.index("--")
    except ValueError as exc:
        raise ProfileError("missing profile separator") from exc
    if separator == 0:
        raise ProfileError("empty profile")
    return args[:separator], args[separator + 1:]


def workspace_bind(profile: list[str], workspace_root: str) -> str:
    """The workspace path from the provider's ``--bind <path> <path>`` pair."""
    root = Path(workspace_root)
    for index in range(len(profile) - 2):
        if profile[index] != "--bind":
            continue
        source, target = profile[index + 1], profile[index + 2]
        if source != target:
            continue
        if not Path(source).is_absolute():
            continue
        try:
            relative = Path(source).relative_to(root)
        except ValueError:
            continue
        if any(part in ("", ".", "..") for part in relative.parts):
            raise ProfileError("workspace path is not normalized: " + source)
        if relative.parts:
            return source
    raise ProfileError("profile has no workspace bind under " + str(root))


def mask_args(workspace_root: str,
              mask_dirs: tuple[str, ...] = MASK_DIRS) -> list[str]:
    """Flags that hide host paths and reveal only the child's workspace."""
    root = Path(workspace_root)
    args: list[str] = []
    for path in mask_dirs:
        if Path(path).is_dir():
            args += ["--tmpfs", path]
    args += ["--tmpfs", str(root)]
    return args


def build_bwrap_argv(
    profile: list[str],
    command: list[str],
    *,
    bwrap: str = BWRAP,
    workspace_root: str = WORKSPACE_ROOT,
    mask_dirs: tuple[str, ...] = MASK_DIRS,
) -> list[str]:
    """The complete bwrap argv for one confined command."""
    if not command:
        raise ProfileError("empty command")
    workspace = workspace_bind(profile, workspace_root)
    root = Path(workspace_root)
    relative = Path(workspace).relative_to(root)
    args = [bwrap, *NAMESPACE_ARGS, *profile,
            *mask_args(workspace_root, mask_dirs)]
    current = root
    for part in relative.parts:
        current /= part
        args += ["--dir", str(current)]
    args += ["--bind", workspace, workspace]
    args += ["--setenv", "HOME", "/tmp", "--setenv", "TMPDIR", "/tmp"]
    args += ["--", *command]
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        profile, command = split_profile(
            list(argv if argv is not None else sys.argv[1:]))
        target = build_bwrap_argv(profile, command)
    except ProfileError as exc:
        print(f"hive-bwrap: {exc}", file=sys.stderr)
        return 125
    try:
        os.execv(target[0], target)
    except OSError as exc:
        print(f"hive-bwrap: cannot exec {target[0]}: {exc}", file=sys.stderr)
        return 126
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
