"""The DSH hook installer must be idempotent and never overwrite a user patch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.agent_hooks import (
    HOOK_ID,
    ensure_hook_profile,
    hook_config,
    preflight_runtime,
    runtime_hook_error,
)


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
    assert "- insert:" in patch
    assert f"id: {HOOK_ID}" in patch
    assert "hooks-claude-code" in patch
    assert "configPath:" in patch
    second = ensure_hook_profile(home, tmp_path, python="/usr/bin/python3")
    assert second["installed"] is True
    assert "already installed" in second["reason"]
    assert (home / "cordis.patch.yml").read_text(encoding="utf-8").count(
        "id: hooks-claude-code") == 1


def test_a_conflicting_user_row_is_not_claimed(tmp_path):
    home = tmp_path / "dsh-home"
    home.mkdir()
    (home / "cordis.patch.yml").write_text(
        "- id: hooks-claude-code\n  name: '@example/not-hive'\n",
        encoding="utf-8")
    result = ensure_hook_profile(home, tmp_path, python="/usr/bin/python3")
    assert result["installed"] is False
    assert "different config" in result["reason"]


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


def test_runtime_hook_error_only_matches_activation_failures():
    assert runtime_hook_error(
        "dsh: warning: hooks-claude-code: failed to import: boom") != ""
    assert runtime_hook_error("hooks-claude-code: did not activate") != ""
    assert runtime_hook_error("ordinary warning") == ""
    assert runtime_hook_error("hooks-claude-code: fine") == ""


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_preflight_refuses_an_inactive_hook(tmp_path):
    script = _script(tmp_path / "dsh", (
        "echo 'hooks-claude-code: failed to import: no protocol' >&2\n"
        "exit 0\n"))
    with pytest.raises(RuntimeError, match="failed to import"):
        preflight_runtime(script, tmp_path / "home", tmp_path)


def test_preflight_accepts_a_clean_boot(tmp_path):
    script = _script(tmp_path / "dsh", "exit 0\n")
    preflight_runtime(script, tmp_path / "home", tmp_path)


def test_preflight_refuses_a_boot_failure_even_without_hook_text(tmp_path):
    script = _script(tmp_path / "dsh", "echo 'profile failed' >&2\nexit 1\n")
    with pytest.raises(RuntimeError, match="exited 1"):
        preflight_runtime(script, tmp_path / "home", tmp_path)


def test_preflight_reports_a_missing_runtime(tmp_path):
    with pytest.raises(RuntimeError, match="could not start"):
        preflight_runtime(tmp_path / "missing", tmp_path / "home", tmp_path)
