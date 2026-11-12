"""The Global harness controls store one baseline for the whole harness.

The visual controls moved to the new shell (Settings + Setup). These tests
cover the backend contract they share: the four settings blocks, partial
saves, and the encoder choice.
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


def test_the_encoder_choice_is_stored(client):
    """The drone the Studio builds follows the choice, not just the env."""
    client.put("/v1/studio/settings",
               json={"encoder": {"embedding_model": "BAAI/bge-small-en-v1.5"}})
    body = client.get("/v1/studio/settings").json()
    assert body["encoder"]["embedding_model"] == "BAAI/bge-small-en-v1.5"
    # the defaults of the block survive the partial save
    assert body["encoder"]["embedding_backend"] == "local"


def test_the_quality_defaults_are_stored(client):
    body = client.put("/v1/studio/settings",
                      json={"quality_defaults": {"judge": "deterministic",
                                                 "repair": "auto"}}).json()
    assert body["quality_defaults"]["judge"] == "deterministic"
    assert body["quality_defaults"]["repair"] == "auto"
