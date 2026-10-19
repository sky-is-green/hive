"""The load and unload path: what the Studio reports, and what it leaves behind.

Two things went wrong before:

  * the Studio claimed a model was loaded when the engine was up but its
    model was not in VRAM (the header said "Loaded: ..." after an unload);
  * unloading retired the server but left its provider and engine profile
    behind - an orphan pointing at a dead port.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from backend.engines import EngineProfile
from backend.providers import Provider
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


class _FakeEngine(BaseHTTPRequestHandler):
    """A stand-in for the Strata serve: /v1/status and /v1/models."""

    loaded = False

    def do_GET(self):  # noqa: N802 - http.server's naming
        body = json.dumps({
            "service": "strata", "model": "fake-176b", "loaded": self.loaded,
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the test output quiet
        pass


@pytest.fixture()
def fake_engine():
    server = HTTPServer(("127.0.0.1", 0), _FakeEngine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


def _point_at(app, server):
    base = f"http://127.0.0.1:{server.server_port}"
    from harness.studio_settings import save_settings

    save_settings({"strata_base": base + "/v1"},
                  app.state.harness.state_dir)


def test_an_engine_that_is_up_but_empty_is_not_a_loaded_model(client, fake_engine):
    """alive is not loaded: the model list must not claim a model."""
    app = client.app
    _point_at(app, fake_engine)
    _FakeEngine.loaded = False
    body = client.get("/v1/server/status").json()
    assert body["strata"]["alive"] is True
    assert body["strata"]["loaded"] is False
    assert body["loaded"] == [], "an engine without its model is not a loaded model"
    assert body["loaded_model"] is None


def test_a_loaded_engine_is_reported(client, fake_engine):
    app = client.app
    _point_at(app, fake_engine)
    _FakeEngine.loaded = True
    body = client.get("/v1/server/status").json()
    assert body["strata"]["loaded"] is True
    assert [m.get("model") for m in body["loaded"]] == ["fake-176b"]
    assert body["loaded_model"] == "fake-176b"


def test_unloading_retires_the_provider_and_profile_it_orphans(client):
    """A dead server must not leave a provider pointing at its port."""
    app = client.app
    st = app.state.harness
    st.registry.providers.append(
        Provider(name="local-x", base_url="http://127.0.0.1:59999", model="m"))
    st.registry.default = "local-x"
    st.engines.engines.append(EngineProfile(name="local-x", kind="llamacpp"))
    st.engines.default = "local-x"

    r = client.post("/v1/server/unload-loaded")
    assert r.status_code == 200
    body = r.json()
    assert "instances" in body and "strata" in body
    assert "local-x" not in [p.name for p in st.registry.providers]
    assert "local-x" not in [e.name for e in st.engines.engines]
    # the default must not keep pointing at the dead one either
    assert (st.registry.default or "").lower() != "local-x"
    assert (st.engines.default or "").lower() != "local-x"


def test_the_page_carries_the_load_bar(client):
    """The bar is in the page, with the bits the script drives."""
    page = client.get("/server").text
    assert 'id="loadbar"' in page
    assert 'id="loadbar-title"' in page
    assert 'id="loadbar-detail"' in page
    assert "function showLoadBar(" in page
    assert "function hideLoadBar(" in page
    # a successful unload says so, as a toast
    assert "showToast('Unload Successful')" in page
