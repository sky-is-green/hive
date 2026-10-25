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
import shlex
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
        return {
            "installed": False,
            "reason": f"{HOOK_ID} already present in {patch_path}",
            "hooks": str(hooks_path),
            "patch": str(patch_path),
        }
    prefix = existing
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    row = (
        "\n# Hive PreToolUse approval bridge (added by Hive).\n"
        "- id: " + HOOK_ID + "\n"
        "  name: '" + HOOK_PLUGIN + "'\n"
        "  config:\n"
        "    configPath: " + json.dumps(str(hooks_path)) + "\n"
        "    projectDir: " + json.dumps(str(Path(cwd).expanduser().resolve())) + "\n"
    )
    patch_path.write_text(prefix + row, encoding="utf-8")
    return {
        "installed": True,
        "reason": "hook installed",
        "hooks": str(hooks_path),
        "patch": str(patch_path),
    }
