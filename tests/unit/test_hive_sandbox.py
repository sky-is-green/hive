"""The child preset is Docker-only and permits one delegation level."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy" / "hive-sandbox"


def _rows() -> list[dict]:
    data = yaml.safe_load((DEPLOY / "cordis.patch.yml").read_text())
    return [entry for patch in data if "insert" in patch
            for entry in patch["insert"]]


def _preset() -> dict:
    return next(row for row in _rows()
                if row.get("id") == "preset-hive-sandbox")


def _tool() -> dict:
    return next(row for row in _rows()
                if row.get("id") == "hive-subagent-tool")


def _plugin_names() -> list[str]:
    return [str(plugin.get("name") or "")
            for plugin in _preset()["config"]["plugins"]]


def test_the_parent_tool_allows_one_delegation_level():
    assert _tool()["config"]["maxDepth"] == 1


def test_the_child_uses_the_docker_bash_tool_only():
    names = _plugin_names()
    assert "@deepseek-ai/hive-docker-bash" in names
    assert "@deepseek-ai/dsh-persona" in names
    assert not any("dsh-bash-sandbox" in name for name in names)
    assert not any("dsh-sandbox-local" in name for name in names)
    assert not any("dsh-tool-fs" in name for name in names)
    assert not any("dsh-tool-web" in name for name in names)
    assert not any("dsh-tool-subagent" in name for name in names)


def test_the_child_persona_is_the_whole_prompt():
    persona = next(plugin for plugin in _preset()["config"]["plugins"]
                   if plugin.get("id") == "persona")
    assert persona["config"]["complete"] is True
    assert "/workspace" in persona["config"]["prefix"]
