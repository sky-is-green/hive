"""The Studio's chat turn, its stop, and the unload.

The Studio owns the provider connection for a chat turn, so a stop can
actually stop the engine: llama.cpp and strata both stop generating when
the client connection drops.  These tests cover the contract, not the
engine's behaviour (the engine's part is measured on the live serve).
"""

from __future__ import annotations

import json

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


def test_chat_stop_reports_nothing_to_stop(client):
    """Stopping a conversation that is not running is not an error."""
    r = client.post("/v1/chat/stop", json={"conversation_id": "idle"})
    body = r.json()
    assert r.status_code == 200
    assert body["stopped"] is False
    assert body["conversation_id"] == "idle"


def test_unload_loaded_is_safe_when_nothing_is_loaded(client):
    """Unloading with nothing loaded must answer, not raise."""
    r = client.post("/v1/server/unload-loaded")
    assert r.status_code == 200
    body = r.json()
    assert body["instances"] == []
    # the strata engine is not running in the test environment
    assert body["strata"] is not None


def test_chat_stream_answers_without_a_provider(client):
    """No provider configured: a clear error, not a hang."""
    r = client.post("/v1/splinter/stream",
                    json={"query": "hello", "conversation_id": "c"})
    assert r.status_code == 502
    assert "provider" in r.json()["detail"].lower()


def test_chat_stream_rejects_an_empty_query(client):
    r = client.post("/v1/splinter/stream",
                    json={"query": "   ", "conversation_id": "c"})
    assert r.status_code == 422


def test_the_studio_serves_its_own_chat_route(client):
    """The Studio's route shadows the mount, so the stop can work."""
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/splinter/stream" in paths
    assert "/v1/chat/stop" in paths
    assert "/v1/server/unload-loaded" in paths


def test_the_chat_route_answers_sse_events(client, monkeypatch):
    """A turn yields the meta/delta/done frames the UI expects."""
    import harness.app as app_module

    class _FakeResponse:
        """A provider response that streams two chunks."""

        def __init__(self):
            self.closed = False

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            for line in (
                'data: {"choices":[{"delta":{"content":"He"}}]}',
                'data: {"choices":[{"delta":{"content":"llo"}}]}',
                "data: [DONE]",
            ):
                yield line

        def close(self):
            self.closed = True

    fake = _FakeResponse()

    class _FakeProvider:
        base_url = "http://127.0.0.1:9"
        api_key = ""
        model = "fake"
        extra_headers: dict = {}

    monkeypatch.setattr(
        app_module.requests, "post",
        lambda *a, **k: fake,
    )
    app = client.app
    # a provider that resolves, and a splinter that curates nothing
    from harness.studio_settings import save_settings
    save_settings({"strata_base": "http://127.0.0.1:8081/v1"}, app.state.harness.state_dir)
    state = app.state.harness
    monkeypatch.setattr(
        type(state.registry), "resolve",
        lambda self, name=None: _FakeProvider(),
    )

    with client.stream("POST", "/v1/splinter/stream",
                       json={"query": "hi", "conversation_id": "c"}) as response:
        assert response.status_code == 200
        frames = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    kinds = [f.get("type") for f in frames]
    assert "delta" in kinds, kinds
    text = "".join(f.get("text", "") for f in frames if f.get("type") == "delta")
    assert text == "Hello"
    assert fake.closed, "the provider response must be closed when the turn ends"
