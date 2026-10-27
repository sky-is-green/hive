#!/usr/bin/python3
"""Bubblewrap runner for Hive's sandboxed subagents.

DSH's ``dsh-sandbox-local`` provider appends a bwrap-compatible profile to the
configured ``runnerCommand``.  This wrapper adds Hive's deployment policy:

* a private user, network, IPC, UTS, and PID namespace, with further user
  namespaces disabled;
* no capabilities, a fresh terminal session, and a fixed hostname;
* tmpfs masks over the host paths a subagent must not read (the human home,
  the Hive state and app trees, GUARD, the engine, the runtime socket dir);
* a private scratch workspace: the provider's policy bind is replaced by the
  command's actual cwd, sibling workspaces are masked, and the cwd is bound
  back and made the command's starting directory.

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
    # /run covers /run/media, /run/user and the rootless Docker socket.
    "/run",
    "/media",
    "/mnt",
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


def profile_bind(profile: list[str]) -> int:
    """Index of the provider's single ``--bind <path> <path>`` workspace pair.

    DSH's sandbox policy roots a child at the parent's permission root, so the
    provider's bind can be broader than the child's scratch directory.  The
    runner replaces it with the actual command cwd.
    """
    found = [index for index in range(len(profile) - 2)
             if profile[index] == "--bind"
             and profile[index + 1] == profile[index + 2]]
    if len(found) != 1:
        raise ProfileError(f"expected one workspace bind, found {len(found)}")
    return found[0]


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
    cwd: str | None = None,
) -> list[str]:
    """The complete bwrap argv for one confined command.

    The policy bind is the provider's permission root, which the child inherits
    from the parent.  The actual command cwd is the child's scratch directory;
    the runner narrows the bind to it and refuses a cwd outside the workspace
    root.
    """
    if not command:
        raise ProfileError("empty command")
    root = Path(workspace_root).resolve()
    target = Path(cwd if cwd is not None else os.getcwd()).resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ProfileError(
            f"command cwd is outside {root}: {target}") from exc
    if not relative.parts:
        raise ProfileError(f"command cwd must be a subdirectory of {root}")
    if not target.is_dir():
        raise ProfileError(f"command cwd does not exist: {target}")
    index = profile_bind(profile)
    confined = profile[:index] + profile[index + 3:]
    args = [bwrap, *NAMESPACE_ARGS, *confined,
            *mask_args(str(root), mask_dirs)]
    current = root
    for part in relative.parts:
        current /= part
        args += ["--dir", str(current)]
    args += ["--bind", str(target), str(target), "--chdir", str(target)]
    args += ["--setenv", "HOME", "/tmp", "--setenv", "TMPDIR", "/tmp"]
    args += ["--", *command]
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        profile, command = split_profile(
            list(argv if argv is not None else sys.argv[1:]))
        target = build_bwrap_argv(profile, command, cwd=os.getcwd())
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
