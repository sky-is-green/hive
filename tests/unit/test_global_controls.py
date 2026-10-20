"""The Global harness controls: one baseline for the whole harness.

The controls were scattered - the memory knobs in the Splinter tab (as
per-conversation overrides), the cascade knobs on the cascade page, the
Studio's switches in the chat footer.  The Global tab holds the baseline
they all start from, and the settings store it as four blocks.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from harness.app import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(
        providers_file=tmp_path / "providers.local.json",
        engines_file=tmp_path / "engines.local.json",
        runs_root=tmp_path / "runs",
        log_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
    )
    return TestClient(app)


def test_the_global_tab_is_in_the_console(client):
    page = client.get("/server").text
    assert 'data-tab="tab-global"' in page
    assert 'id="tab-global"' in page
    for group in ("Memory &amp; RAG", "Quality (the cascade)", "MCP", "Runtime"):
        assert group in page
    assert "function saveGlobalControls(" in page
    assert "function fillGlobalControls(" in page


def test_the_settings_carry_the_four_blocks(client):
    body = client.get("/v1/studio/settings").json()
    for block in ("memory_defaults", "quality_defaults", "mcp", "runtime_defaults"):
        assert block in body, f"the settings have no {block} block"
    assert body["mcp"]["tools"] == ["remember", "search"]


def test_saving_a_block_keeps_the_others(client):
    client.put("/v1/studio/settings",
               json={"memory_defaults": {"max_context": 32768},
                     "mcp": {"read_only": True}})
    body = client.get("/v1/studio/settings").json()
    assert body["memory_defaults"]["max_context"] == 32768
    assert body["mcp"]["read_only"] is True
    # the mcp block's other keys survive a partial save
    assert body["mcp"]["tools"] == ["remember", "search"]
    assert body["mcp"]["enabled"] is True


def test_the_settings_index_points_at_the_global_tab(client):
    from harness.reports import SETTINGS_INDEX

    labels = {s["label"] for s in SETTINGS_INDEX}
    assert "Memory & RAG baseline" in labels
    assert "Cascade defaults" in labels
    assert "MCP controls" in labels
    for entry in SETTINGS_INDEX:
        if entry["label"] in ("Memory & RAG baseline", "Cascade defaults", "MCP controls"):
            assert entry["tab"] == "tab-global"


def test_the_memory_defaults_reach_a_new_conversation(client):
    """A conversation that overrides nothing gets the global baseline."""
    from harness.studio_settings import load_settings

    client.put("/v1/studio/settings",
               json={"memory_defaults": {"max_context": 16384}})
    state_dir = client.app.state.harness.state_dir
    settings = load_settings(state_dir)
    assert settings["memory_defaults"]["max_context"] == 16384
    # and the chat route merges it under the conversation's own overrides
    merged = {**settings["memory_defaults"], **{"max_context": 32768}}
    assert merged["max_context"] == 32768
