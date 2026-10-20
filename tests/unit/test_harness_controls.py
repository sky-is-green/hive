"""The Global tab's controls, wired into the paths that run.

The controls were stored but not consulted: the memory baseline reached
the chat by hand, the encoder never reached the drone, the cascade's
knobs were not the baseline of a run, the MCP block was not in force and
the runtime baseline was not the launch's default.  These tests pin the
wiring down.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from harness.app import create_app
from harness import controls


@pytest.fixture()
def app(tmp_path):
    return create_app(
        providers_file=tmp_path / "providers.local.json",
        engines_file=tmp_path / "engines.local.json",
        runs_root=tmp_path / "runs",
        log_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
    )


@pytest.fixture()
def client(app):
    return TestClient(app)


def _set(client, **blocks):
    return client.put("/v1/studio/settings", json=blocks)


# ----------------------------------------------------------------------
# the curation baseline
# ----------------------------------------------------------------------
def test_the_memory_baseline_reaches_a_conversation(client, app):
    """A conversation starts from the Global tab's curation baseline."""
    _set(client, memory_defaults={"max_context": 4096, "stale_threshold": 7})
    st = app.state.harness
    splinter = st.splinter_for("conv-baseline", None, with_backend=False)
    assert splinter.config.max_context == 4096
    assert splinter.config.stale_threshold == 7


def test_a_conversation_may_override_the_baseline(client, app):
    """The baseline is a floor, not a cage."""
    _set(client, memory_defaults={"max_context": 4096})
    st = app.state.harness
    splinter = st.splinter_for("conv-override", {"max_context": 1024},
                               with_backend=False)
    assert splinter.config.max_context == 1024


def test_the_baseline_reaches_the_mcp_conversation(client, app):
    """The MCP path builds its conversation the same way."""
    _set(client, memory_defaults={"max_context": 2048})
    st = app.state.harness
    splinter = st.splinter_for("mcp-conv", None, with_backend=False)
    assert splinter.config.max_context == 2048


# ----------------------------------------------------------------------
# the encoder
# ----------------------------------------------------------------------
def test_the_chosen_encoder_reaches_the_drone(client, app):
    """The model the user picked is the model the drone embeds with."""
    _set(client, encoder={"embedding_model": "BAAI/bge-small-en-v1.5"})
    st = app.state.harness
    choice = st.drone_choice()
    assert choice["model"] == "BAAI/bge-small-en-v1.5"
    drone = st.ultra(choice)
    assert getattr(drone, "_model_name", "") == "BAAI/bge-small-en-v1.5"


def test_a_different_encoder_gets_its_own_drone(client, app):
    """Two encoders never share a drone (the vectors would not match)."""
    st = app.state.harness
    one = st.ultra({"model": "a", "backend": "local", "url": "",
                    "medium_model": "", "vocab_boost": 0.15,
                    "confidence_mode": "off"})
    two = st.ultra({"model": "b", "backend": "local", "url": "",
                    "medium_model": "", "vocab_boost": 0.15,
                    "confidence_mode": "off"})
    assert one is not two
    # the same choice is cached
    assert st.ultra({"model": "a", "backend": "local", "url": "",
                      "medium_model": "", "vocab_boost": 0.15,
                      "confidence_mode": "off"}) is one


def test_the_encoder_lands_in_the_conversation_config(client, app):
    """The conversation records the encoder it ran with."""
    _set(client, encoder={"embedding_model": "intfloat/e5-small-v2"})
    st = app.state.harness
    splinter = st.splinter_for("conv-encoder", None, with_backend=False)
    assert splinter.config.ultra_model == "intfloat/e5-small-v2"


# ----------------------------------------------------------------------
# the runtime baseline: the sampling and the launch
# ----------------------------------------------------------------------
def test_the_sampling_baseline_reaches_the_conversation(client, app):
    """The reply is drawn with the sampling the user asked for."""
    _set(client, runtime_defaults={"temperature": 0.25, "top_p": 0.9})
    st = app.state.harness
    splinter = st.splinter_for("conv-sampling", None, with_backend=False)
    assert splinter.config.sampling["temperature"] == 0.25
    assert splinter.config.sampling["top_p"] == 0.9


def test_the_engine_profile_wins_over_the_baseline(client, app):
    """A named engine is more specific than the harness-wide baseline."""
    _set(client, runtime_defaults={"temperature": 0.25})
    st = app.state.harness
    from splinter.backend.engines import EngineProfile

    st.engines.engines = [EngineProfile(name="fast", kind="lmstudio",
                                        sampling={"temperature": 0.9})]
    st.engines.default = "fast"
    splinter = st.splinter_for("conv-profile", None, with_backend=False)
    assert splinter.config.sampling["temperature"] == 0.9


def test_the_launch_baseline_fills_the_context(client, app):
    """A launch that says nothing takes the runtime baseline's context."""
    _set(client, runtime_defaults={"ctx_size": 12345, "ngl": 33})
    st = app.state.harness
    launch = st.controls()["launch"]
    assert launch["ctx_size"] == 12345
    assert launch["ngl"] == 33


# ----------------------------------------------------------------------
# the cascade baseline
# ----------------------------------------------------------------------
def _cascade_client(tmp_path, defaults, seen):
    """A standalone app with the cascade router and a fake launcher."""
    from fastapi import FastAPI

    from harness.cascade import api as cascade_api

    def fake_spawn(cmd, cwd, log):
        seen["cmd"] = cmd
        return 4321

    app = FastAPI()
    app.include_router(cascade_api.create_router(
        runs_root=tmp_path / "runs",
        tasks_root=tmp_path,
        runner=tmp_path / "runner.py",
        spawn=fake_spawn,
        defaults=lambda: defaults,
    ))
    (tmp_path / "tasks-hard.json").write_text("[]")
    (tmp_path / "runner.py").write_text("")
    return TestClient(app)


def test_the_cascade_baseline_reaches_the_runner(tmp_path):
    """The argv a run is launched with carries the baseline."""
    seen: dict = {}
    client = _cascade_client(tmp_path, {"judge": "tiny-jev", "repair": "off",
                                        "human_gate": "off"}, seen)
    body = client.post("/v1/cascade/run", json={"tasks": "tasks-hard.json",
                                                "run_name": "baseline-run"})
    assert body.status_code == 200, body.text
    cmd = seen["cmd"]
    assert "--judge" in cmd and "tiny-jev" in cmd
    assert "--repair" in cmd and "off" in cmd
    assert "--human-gate" in cmd and "off" in cmd


def test_a_run_may_override_the_cascade_baseline(tmp_path):
    """The baseline is a floor: a run that names its own judge wins."""
    seen: dict = {}
    client = _cascade_client(tmp_path, {"judge": "tiny-jev"}, seen)
    client.post("/v1/cascade/run", json={"tasks": "tasks-hard.json",
                                          "run_name": "own-run",
                                          "judge": "deterministic"})
    cmd = seen["cmd"]
    assert cmd[cmd.index("--judge") + 1] == "deterministic"


def test_the_cascade_baseline_reaches_the_studio_router(client, app, tmp_path):
    """The Studio's own cascade router starts from the Global tab's judge."""
    _set(client, quality_defaults={"judge": "tiny-jev", "human_gate": "off"})
    assert app.state.harness.controls()["quality"]["judge"] == "tiny-jev"


# ----------------------------------------------------------------------
# the MCP block
# ----------------------------------------------------------------------
def test_the_mcp_block_can_switch_the_endpoint_off(client):
    """Off means off: the endpoint refuses."""
    _set(client, mcp={"enabled": False})
    r = client.post("/v1/mcp", json={"jsonrpc": "2.0", "id": 1,
                                      "method": "tools/list"})
    assert r.status_code == 403
    assert "MCP" in r.json()["detail"]


def test_the_mcp_read_only_withholds_the_write_tool(client):
    """Read-only: the write tool is not offered and does not work."""
    _set(client, mcp={"read_only": True})
    r = client.post("/v1/mcp", json={"jsonrpc": "2.0", "id": 1,
                                      "method": "tools/list"})
    assert r.status_code == 200
    names = [t["name"] for t in r.json()["result"]["tools"]]
    assert "splinter_remember" not in names
    assert "splinter_search" in names

    r = client.post("/v1/mcp", json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "splinter_remember",
                   "arguments": {"conversation_id": "c1", "text": "hello"}}})
    assert r.status_code == 200
    assert "error" in r.json() or r.json().get("result", {}).get("isError")


def test_the_mcp_block_can_withhold_the_search_tool(client):
    """The tool list is the block's list, not a fixed one."""
    _set(client, mcp={"tools": ["remember"]})
    r = client.post("/v1/mcp", json={"jsonrpc": "2.0", "id": 1,
                                      "method": "tools/list"})
    names = [t["name"] for t in r.json()["result"]["tools"]]
    assert names == ["splinter_remember"]


def test_the_mcp_endpoint_still_works_by_default(client):
    """Nothing configured: the memory is exposed as it always was."""
    r = client.post("/v1/mcp", json={"jsonrpc": "2.0", "id": 1,
                                      "method": "tools/list"})
    assert r.status_code == 200
    names = [t["name"] for t in r.json()["result"]["tools"]]
    assert "splinter_remember" in names and "splinter_search" in names


# ----------------------------------------------------------------------
# the chat reports what it ran with
# ----------------------------------------------------------------------
def test_the_chat_says_what_it_ran_with(client, app, monkeypatch):
    """The meta frame carries the encoder and the curation."""
    client.post("/v1/provider/config", json={
        "providers": [{"name": "fake", "base_url": "http://127.0.0.1:9/v1",
                       "api_key": "k", "model": "m"}],
        "default": "fake",
    })
    _set(client, memory_defaults={"max_context": 4096},
         encoder={"embedding_model": "BAAI/bge-small-en-v1.5"})

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'
            yield "data: [DONE]"

        def close(self):
            return None

    monkeypatch.setattr("harness.app.requests.post",
                        lambda *a, **k: FakeResponse())
    with client.stream("POST", "/v1/splinter/stream",
                       json={"query": "hello", "conversation_id": "c-chat"}) as r:
        frames = [json.loads(line[6:]) for line in r.iter_lines()
                  if line.startswith("data: ") and line[6:].strip() != "[DONE]"]
    meta = next(f for f in frames if f.get("type") == "meta")
    assert meta["harness"]["encoder"] == "BAAI/bge-small-en-v1.5"
    assert meta["harness"]["memory"]["max_context"] == 4096


# ----------------------------------------------------------------------
# the resolvers themselves
# ----------------------------------------------------------------------
def test_the_resolvers_read_the_stored_blocks():
    settings = {
        "memory_defaults": {"max_context": 4096, "nonsense": 1},
        "runtime_defaults": {"temperature": 0.3, "context": 8192,
                              "gpu_layers": 99, "top_p": 0.8},
        "encoder": {"embedding_model": "m", "embedding_backend": "served",
                    "embedding_url": "http://x", "medium_model": "mm"},
        "quality_defaults": {"judge": "tiny-jev"},
        "mcp": {"enabled": False, "read_only": True, "tools": ["search"]},
    }
    assert controls.memory_of(settings) == {"max_context": 4096}
    assert controls.sampling_of(settings) == {"temperature": 0.3, "top_p": 0.8}
    assert controls.launch_of(settings) == {"ctx_size": 8192, "ngl": 99}
    assert controls.encoder_of(settings)["model"] == "m"
    assert controls.encoder_of(settings)["backend"] == "served"
    assert controls.quality_of(settings) == {"judge": "tiny-jev"}
    assert controls.mcp_of(settings) == {"enabled": False, "read_only": True,
                                          "tools": ["search"]}


def test_an_empty_settings_block_is_not_a_baseline():
    """No settings, no baseline: the defaults stand."""
    assert controls.memory_of({}) == {}
    assert controls.sampling_of({}) == {}
    assert controls.launch_of({}) == {}
    assert controls.quality_of({}) == {}
    assert controls.mcp_of({})["enabled"] is True


# ----------------------------------------------------------------------
# the harvest: the data the encoder is trained on
# ----------------------------------------------------------------------
def _fake_provider(client):
    client.post("/v1/provider/config", json={
        "providers": [{"name": "fake", "base_url": "http://127.0.0.1:9/v1",
                       "api_key": "k", "model": "m"}],
        "default": "fake",
    })


def test_the_harvest_is_off_by_default(client):
    """The file holds the user's own words, so it is their call."""
    body = client.get("/v1/studio/settings").json()
    assert body["harvest_enabled"] is False
    assert client.get("/v1/studio/harvest").json()["enabled"] is False


def test_a_turn_is_harvested_when_asked_for(client, app, monkeypatch):
    """With the harvest on, the turn and its curation are recorded."""
    _fake_provider(client)
    _set(client, harvest_enabled=True)

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            yield 'data: {"choices":[{"delta":{"content":"the answer"}}]}'
            yield "data: [DONE]"

        def close(self):
            return None

    monkeypatch.setattr("harness.app.requests.post",
                        lambda *a, **k: FakeResponse())
    with client.stream("POST", "/v1/splinter/stream",
                       json={"query": "the question",
                             "conversation_id": "c-harvest"}) as r:
        list(r.iter_lines())
    body = client.get("/v1/studio/harvest").json()
    assert body["turns"] == 1
    record = body["records"][0]
    assert record["query"] == "the question"
    assert record["reply"] == "the answer"
    assert record["encoder"]
    assert "config" in record


def test_the_harvest_can_be_thrown_away(client, app, monkeypatch):
    """The user's data, the user's call."""
    _fake_provider(client)
    _set(client, harvest_enabled=True)

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            yield 'data: {"choices":[{"delta":{"content":"x"}}]}'
            yield "data: [DONE]"

        def close(self):
            return None

    monkeypatch.setattr("harness.app.requests.post",
                        lambda *a, **k: FakeResponse())
    with client.stream("POST", "/v1/splinter/stream",
                       json={"query": "q", "conversation_id": "c-clear"}) as r:
        list(r.iter_lines())
    assert client.get("/v1/studio/harvest").json()["turns"] == 1
    client.delete("/v1/studio/harvest")
    assert client.get("/v1/studio/harvest").json()["turns"] == 0
