"""Install the Hive PreToolUse hook into a DSH home.

The DSH runtime ships a Claude-compatible hook bridge.  Hive writes a tiny
process-level hook config and appends one row to the DSH home patch.  The hook
itself (``harness.tool_hook``) fails closed if the Studio gateway is absent.

Existing user patches are never overwritten.  If a ``hooks-claude-code`` row is
already present, Hive leaves it alone and reports ``installed=False``; the
caller can surface that instead of pretending the gateway is active.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

HOOK_ID = "hooks-claude-code"
HOOK_PLUGIN = "@deepseek-ai/dsh-hooks-claude-code"
HOOK_FILE = "hive-hooks.json"
PATCH_FILE = "cordis.patch.yml"


def hook_command(python: str | Path | None = None) -> str:
    script = Path(__file__).resolve().with_name("tool_hook.py")
    return " ".join((
        shlex.quote(str(python or sys.executable)),
        shlex.quote(str(script)),
    ))


def hook_config(python: str | Path | None = None) -> dict[str, Any]:
    return {
        "hooks": {
            "PreToolUse": [{
                "matcher": "*",
                "hooks": [{
                    "type": "command",
                    "command": hook_command(python),
                }],
            }],
        },
    }


def ensure_hook_profile(dsh_home: str | Path, cwd: str | Path, *,
                        python: str | Path | None = None) -> dict[str, Any]:
    """Write the hook file and patch row; return what was actually installed."""
    home = Path(dsh_home).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    hooks_path = home / HOOK_FILE
    config = hook_config(python)
    hooks_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    patch_path = home / PATCH_FILE
    existing = patch_path.read_text(encoding="utf-8") if patch_path.is_file() else ""
    if f"id: {HOOK_ID}" in existing:
        ours = (HOOK_PLUGIN in existing
                and str(hooks_path) in existing
                and "- insert:" in existing)
        return {
            "installed": ours,
            "reason": (f"{HOOK_ID} already installed by Hive" if ours
                       else f"{HOOK_ID} already present with a different config"),
            "hooks": str(hooks_path),
            "patch": str(patch_path),
        }
    prefix = existing
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    row = (
        "\n# Hive PreToolUse approval bridge (added by Hive).\n"
        "- insert:\n"
        "    - id: " + HOOK_ID + "\n"
        "      name: '" + HOOK_PLUGIN + "'\n"
        "      config:\n"
        "        configPath: " + json.dumps(str(hooks_path)) + "\n"
        "        projectDir: " + json.dumps(str(Path(cwd).expanduser().resolve())) + "\n"
    )
    patch_path.write_text(prefix + row, encoding="utf-8")
    return {
        "installed": True,
        "reason": "hook installed",
        "hooks": str(hooks_path),
        "patch": str(patch_path),
    }


def runtime_hook_error(stderr: str, stdout: str = "") -> str:
    """Return the DSH activation error for this hook, or an empty string."""
    for line in (stderr + "\n" + stdout).splitlines():
        if "hooks-claude-code" not in line:
            continue
        if "failed to import" in line or "did not activate" in line:
            return line.strip()
    return ""


def preflight_runtime(dsh_bin: str | Path, dsh_home: str | Path,
                      cwd: str | Path, *, profile: str = "sdk",
                      timeout_s: float = 30.0) -> None:
    """Boot an external DSH runtime once and reject an inactive hook.

    DSH logs a plugin import/activation failure to stderr and continues.  A
    hook that never mounted is not a weaker guard; it is no guard at all, so
    Hive must refuse the agent before the first turn.
    """
    env = dict(os.environ)
    env["DSH_HOME"] = str(Path(dsh_home).expanduser().resolve())
    env.setdefault("DSH_PERMISSION_MODE", "workspace-write")
    command = [str(Path(dsh_bin).expanduser().resolve()),
               "--profile", str(profile)]
    try:
        proc = subprocess.run(
            command, cwd=str(Path(cwd).expanduser().resolve()), env=env,
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=max(1.0, float(timeout_s)))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "DSH hook preflight timed out; refusing an unverified agent") from exc
    except OSError as exc:
        raise RuntimeError(f"DSH hook preflight could not start: {exc}") from exc
    error = runtime_hook_error(proc.stderr or "", proc.stdout or "")
    if error:
        raise RuntimeError(
            f"DSH hook preflight failed: {error}. The hook plugin must be "
            "available to the selected DSH profile before Hive runs tools.")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
        detail = " | ".join(tail) or "no diagnostic"
        raise RuntimeError(
            f"DSH runtime preflight exited {proc.returncode}: {detail}")
