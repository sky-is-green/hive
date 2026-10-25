"""The DSH hook installer must be idempotent and never overwrite a user patch."""

from __future__ import annotations

import json
from pathlib import Path

from harness.agent_hooks import HOOK_ID, ensure_hook_profile, hook_config


def test_hook_config_targets_pretooluse(tmp_path):
    config = hook_config("/usr/bin/python3")
    hook = config["hooks"]["PreToolUse"][0]
    assert hook["matcher"] == "*"
    assert hook["hooks"][0]["type"] == "command"
    assert "harness/tool_hook.py" in hook["hooks"][0]["command"]
    assert "/usr/bin/python3" in hook["hooks"][0]["command"]


def test_hook_profile_is_installed_and_idempotent(tmp_path):
    home = tmp_path / "dsh-home"
    first = ensure_hook_profile(home, tmp_path, python="/usr/bin/python3")
    assert first["installed"] is True
    hooks = json.loads((home / "hive-hooks.json").read_text(encoding="utf-8"))
    assert hooks["hooks"]["PreToolUse"][0]["hooks"][0]["type"] == "command"
    patch = (home / "cordis.patch.yml").read_text(encoding="utf-8")
    assert f"id: {HOOK_ID}" in patch
    assert "hooks-claude-code" in patch
    second = ensure_hook_profile(home, tmp_path, python="/usr/bin/python3")
    assert second["installed"] is False
    assert (home / "cordis.patch.yml").read_text(encoding="utf-8").count(
        "id: hooks-claude-code") == 1


def test_existing_user_patch_is_preserved(tmp_path):
    home = tmp_path / "dsh-home"
    home.mkdir()
    patch = home / "cordis.patch.yml"
    patch.write_text("- id: my-own-row\n  name: '@example/plugin'\n",
                     encoding="utf-8")
    result = ensure_hook_profile(home, tmp_path, python="/usr/bin/python3")
    assert result["installed"] is True
    text = patch.read_text(encoding="utf-8")
    assert "my-own-row" in text
    assert "id: hooks-claude-code" in text
