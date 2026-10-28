"""The parent delegate waits; the child preset is Docker-only."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy" / "hive-sandbox"


def _data() -> list[dict]:
    return yaml.safe_load((DEPLOY / "cordis.patch.yml").read_text())


def _rows() -> list[dict]:
    return [entry for patch in _data() if "insert" in patch
            for entry in patch["insert"]]


def _preset() -> dict:
    return next(row for row in _rows()
                if row.get("id") == "preset-hive-sandbox")


def _delegate() -> dict:
    return next(row for row in _rows()
                if row.get("id") == "hive-delegate-tool")


def _plugin_names() -> list[str]:
    return [str(plugin.get("name") or "")
            for plugin in _preset()["config"]["plugins"]]


def test_the_parent_has_one_waiting_delegate():
    row = _delegate()
    assert row["name"] == "@deepseek-ai/hive-delegate"
    assert row["config"]["toolName"] == "delegate"
    assert row["config"]["agentPreset"] == "hive-sandbox"
    assert row["config"]["maxDepth"] == 1
    assert not any(row.get("id") == "hive-subagent-tool" for row in _rows())


def test_the_async_subagent_tool_is_disabled():
    disabled = [row.get("id") for row in _data()
                if row.get("disabled") is True]
    assert "tool-subagent" in disabled
    assert "tool-subagent-control" in disabled
    assert "tool-subagent-fork" in disabled


def test_the_child_uses_the_docker_bash_tool_only():
    names = _plugin_names()
    assert "@deepseek-ai/hive-docker-bash" in names
    assert "@deepseek-ai/dsh-persona" in names
    assert not any("dsh-bash-sandbox" in name for name in names)
    assert not any("dsh-sandbox-local" in name for name in names)
    assert not any("dsh-tool-fs" in name for name in names)
    assert not any("dsh-tool-web" in name for name in names)
    assert not any("dsh-tool-subagent" in name for name in names)
    assert not any("hive-delegate" in name for name in names)


def test_the_child_persona_is_the_whole_prompt():
    persona = next(plugin for plugin in _preset()["config"]["plugins"]
                   if plugin.get("id") == "persona")
    assert persona["config"]["complete"] is True
    assert "/workspace" in persona["config"]["prefix"]
