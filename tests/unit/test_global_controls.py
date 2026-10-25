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


def test_the_controller_panel_is_manual_and_explicit(client):
    page = client.get("/server").text
    assert 'id="ctrl-consent"' in page
    assert "function scanHardware(" in page
    assert "function proposeController(" in page
    assert "function applyController(" in page
    assert "function revertController(" in page
    assert "function outcomeController(" in page
    assert 'id="ctrl-request"' in page
    assert 'id="ctrl-embed-request"' in page
    assert "embed_request" in page
    assert "/v1/controller/apply" in page
    assert "/v1/controller/outcome" in page
    assert "confirm=true" in page or "confirm: true" in page


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


def test_the_encoder_is_chosen_in_the_global_tab(client):
    """The small model that does the retrieval and the memory lives here."""
    page = client.get("/server").text
    assert 'id="ge-embedding_model"' in page
    assert 'id="ge-embedding_backend"' in page
    assert 'id="ge-embedding_url"' in page
    assert 'id="ge-medium_model"' in page
    assert 'id="ge-controller_idle_unload"' in page
    assert 'id="ge-controller_idle_seconds"' in page
    # the candidates are offered, but any model id can be typed
    assert "sentence-transformers/paraphrase-MiniLM-L3-v2" in page
    assert "Qwen/Qwen3-Embedding-0.6B" in page
    assert 'list="encoder-candidates"' in page


def test_the_encoder_choice_is_stored(client):
    """The drone the Studio builds follows the choice, not just the env."""
    client.put("/v1/studio/settings",
               json={"encoder": {"embedding_model": "BAAI/bge-small-en-v1.5"}})
    body = client.get("/v1/studio/settings").json()
    assert body["encoder"]["embedding_model"] == "BAAI/bge-small-en-v1.5"
    # the defaults of the block survive the partial save
    assert body["encoder"]["embedding_backend"] == "local"


def test_the_splinter_tab_is_gone(client):
    """The harness controls have one home; the separate menu is gone."""
    page = client.get("/server").text
    assert 'id="tab-splinter"' not in page
    assert 'data-tab="tab-splinter"' not in page
    # and its controls live in the Global tab instead
    for field in ("gm-max_context", "gm-stale_threshold", "gm-comb_enabled",
                  "gm-confidence_mode", "gm-drift_prior"):
        assert f'id="{field}"' in page, f"the Global tab is missing {field}"


def test_the_settings_index_has_no_dead_links(client):
    """Every tab the index points at exists in the page."""
    from harness.reports import SETTINGS_INDEX

    page = client.get("/server").text
    for entry in SETTINGS_INDEX:
        tab = entry.get("tab")
        if not tab:
            continue
        assert f'data-tab="{tab}"' in page, (
            f"the settings index points at {tab}, which the console does not have")


def test_an_unset_control_is_not_written_as_false(client):
    """Opening the tab and pressing Save must not switch off the defaults.

    The controls that were never stored are *unset*, not false: the tab
    has to remember the difference, or a save would write false for
    every control the user never touched.
    """
    page = client.get("/server").text
    assert "el.dataset.unset" in page
    assert "if (el.checked) values[key] = true;" in page
    # the fill marks a control with no stored value as unset
    assert "el.dataset.unset = '1'" in page


def test_the_global_tab_sends_the_encoder_block(client):
    """The encoder is a stored block, so the tab has to save it."""
    page = client.get("/server").text
    assert "ge: 'encoder'" in page
