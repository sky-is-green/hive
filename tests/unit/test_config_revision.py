"""The provider/engine config endpoints implement the optimistic-concurrency
guard the Studio console claims: a stale ``expectedRevision`` is a 409, and a
matching one applies the write and returns the new revision.
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


def test_provider_config_returns_a_revision(client):
    body = client.get("/v1/provider/config").json()
    assert body["revision"], "the GET must expose the revision token"


def test_provider_config_rejects_a_stale_revision(client):
    stale = client.get("/v1/provider/config").json()["revision"]
    # someone else changes the config in between
    client.post("/v1/provider/config", json={
        "providers": [{"name": "other", "base_url": "http://127.0.0.1:9999/v1"}],
        "default": "other",
    })
    r = client.post("/v1/provider/config", json={
        "providers": [{"name": "mine", "base_url": "http://127.0.0.1:1234/v1"}],
        "default": "mine",
        "expectedRevision": stale,
    })
    assert r.status_code == 409, r.text
    # the rejected write must not have landed
    names = {p["name"] for p in client.get("/v1/provider/config").json()["providers"]}
    assert names == {"other"}


def test_provider_config_accepts_a_current_revision(client):
    rev = client.get("/v1/provider/config").json()["revision"]
    r = client.post("/v1/provider/config", json={
        "providers": [{"name": "mine", "base_url": "http://127.0.0.1:1234/v1"}],
        "default": "mine",
        "expectedRevision": rev,
    })
    assert r.status_code == 200, r.text
    assert r.json()["revision"] != rev, "a write must produce a new revision"


def test_engine_config_rejects_a_stale_revision(client):
    stale = client.get("/v1/engines").json()["revision"]
    client.post("/v1/engines", json={
        "engines": [{"name": "other", "kind": "llama_cpp"}],
        "default": "other",
    })
    r = client.post("/v1/engines", json={
        "engines": [{"name": "mine", "kind": "llama_cpp"}],
        "default": "mine",
        "expectedRevision": stale,
    })
    assert r.status_code == 409, r.text
    names = {e["name"] for e in client.get("/v1/engines").json()["engines"]}
    assert names == {"other"}
