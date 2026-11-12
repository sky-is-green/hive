"""The Strata launch backend contract.

The launch UI moved to the new shell (Engines + Lifecycle); these tests cover
the backend routes it calls and the settings they read.
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
    test_client = TestClient(app)
    # The Strata engine is a separate serve on a fixed port, so a test that
    # launches one would otherwise adopt (or collide with) whatever happens to
    # be running on this machine. Point the Studio at a dead port.
    test_client.put("/v1/studio/settings",
                    json={"strata_base": "http://127.0.0.1:1/v1"})
    return test_client


def test_strata_configs_answers(client):
    body = client.get("/v1/strata/configs").json()
    assert "repo" in body
    assert isinstance(body["configs"], list)
    for entry in body["configs"]:
        assert entry["file"].endswith(".json")


def test_strata_launch_needs_a_config(client):
    r = client.post("/v1/strata/launch", json={})
    assert r.status_code == 400
    assert "config" in r.json()["detail"].lower()


def test_strata_launch_needs_a_checkout(client):
    r = client.post("/v1/strata/launch",
                    json={"config": "strata-x.json", "repo": "/nope/strata"})
    assert r.status_code == 400
    assert "repo" in r.json()["detail"].lower()


def test_the_studio_settings_know_the_strata_checkout(client):
    body = client.get("/v1/studio/settings").json()
    assert "strata_repo" in body
    updated = client.put("/v1/studio/settings",
                         json={"strata_repo": "/opt/strata"}).json()
    assert updated["strata_repo"] == "/opt/strata"
