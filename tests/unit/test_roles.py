"""The live role file and the role API."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from backend.openai_compat import OpenAICompatBackend
from cortex.e2e import FakeUltraSmall, MockTransport
from harness.app import ServerStartRequest, create_app
from harness.roles import (
    DEFAULT_PLACEMENTS,
    DEFAULT_ROLES,
    load_roles,
    provider_name,
    resolve_provider,
    save_roles,
)


def test_roles_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "roles.json"
    monkeypatch.setenv("HIVE_ROLES_FILE", str(path))
    assert load_roles()["roles"] == DEFAULT_ROLES
    save_roles({**DEFAULT_ROLES, "controller": "scion35"},
               {**DEFAULT_PLACEMENTS, "controller": "non_display"})
    loaded = load_roles()
    assert loaded["roles"]["controller"] == "scion35"
    assert loaded["placements"]["controller"] == "non_display"
    assert json.loads(path.read_text())["roles"]["controller"] == "scion35"


def test_lfm3_launches_with_reasoning_off():
    req = ServerStartRequest(backend="lfm3", alias="conductor-1.7b")
    args = req.extra_args()
    assert args[args.index("--reasoning") + 1] == "off"
    explicit = ServerStartRequest(backend="lfm3", reasoning="on")
    assert explicit.extra_args()[explicit.extra_args().index("--reasoning") + 1] == "on"


def test_legacy_clerk_key_is_migrated(tmp_path, monkeypatch):
    path = tmp_path / "roles.json"
    monkeypatch.setenv("HIVE_ROLES_FILE", str(path))
    path.write_text(json.dumps({"roles": {"agency": "clerk17"}}))
    assert load_roles()["roles"]["agency"] == "toolmanager17"


def test_provider_aliases():
    assert provider_name("lfm3") == "local-LFM3-1.7B-Q4_K_M"
    assert provider_name("lfm3vl") == "local-LFM3-VL-2B-Q8_0"
    assert provider_name("d1-2b") == "local-d1-2B-Q8_0"
    assert provider_name("controller17") == "controller17"
    assert provider_name("toolmanager17") == "toolmanager17"


def test_provider_resolution_follows_a_library_launch():
    from backend.providers import Provider, ProviderRegistry

    registry = ProviderRegistry(providers=[
        Provider(name="local-hive-conductor-1_7b-v1-Q4_K_M",
                 base_url="http://127.0.0.1:9001",
                 model="conductor-1.7b"),
    ])
    found = resolve_provider(registry, "controller17")
    assert found is not None
    assert found.base_url.endswith(":9001")


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_ROLES_FILE", str(tmp_path / "roles.json"))
    providers = tmp_path / "providers.local.json"
    providers.write_text(json.dumps({
        "providers": [
            {"name": "scion35", "base_url": "http://127.0.0.1:8082",
             "api_key": "", "model": "scion-35b-a3b-pq2_0-corr"},
            {"name": "controller17", "base_url": "http://127.0.0.1:8156",
             "api_key": "", "model": "conductor-1.7b"},
        ],
        "default": "scion35",
    }))
    return create_app(
        ultra_factory=FakeUltraSmall,
        backend_factory=lambda model=None, provider=None: OpenAICompatBackend(
            base_url="http://mock", model=model or "mock-model",
            transport=MockTransport(latency_ms=0)),
        runs_root=tmp_path / "runs",
        providers_file=providers,
        log_dir=str(tmp_path / "logs"),
        state_dir=tmp_path / "harness_state",
    )


def test_role_chat_overrides_the_request_model(tmp_path, monkeypatch):
    import requests

    captured = {}

    class FakeResponse:
        status_code = 200
        content = b'{"ok": true}'
        headers = {"content-type": "application/json"}

        def close(self):
            pass

    def fake_post(self, url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        captured["headers"] = kwargs.get("headers")
        return FakeResponse()

    monkeypatch.setattr(requests.Session, "post", fake_post)
    app = _app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.post("/v1/role/chat/completions", json={
            "role": "controller",
            "model": "not-the-role-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert response.status_code == 200
        assert captured["json"]["model"] == "conductor-1.7b"
        assert captured["headers"]["x-hive-role"] == "controller"
        assert captured["url"].endswith("/v1/chat/completions")


def test_roles_api_resolves_and_saves(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        got = client.get("/v1/roles").json()
        assert got["roles"]["controller"] == "controller17"
        assert got["resolved"]["controller"]["ready"] is True
        assert got["resolved"]["mechanics"]["ready"] is False
        assert got["resolved"]["risk"]["chat"] is False

        saved = client.post("/v1/roles", json={
            "roles": {"controller": "scion35"},
            "placements": {"controller": "non_display"},
        })
        assert saved.status_code == 200
        assert saved.json()["roles"]["controller"] == "scion35"
        assert load_roles()["placements"]["controller"] == "non_display"

        refused = client.post("/v1/role/chat/completions", json={
            "role": "risk", "messages": [{"role": "user", "content": "x"}],
        })
        assert refused.status_code == 409
