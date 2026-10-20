"""The launch console: one tab per engine, and a Strata that can be started.

The Launch panel used to be a handful of fields that could not launch
anything a downloaded model needed.  It now has the shared dials (the
context window, the port) and one tab per engine for what differs:
llama.cpp's own flags on one side, the Strata config and engine flags on
the other.
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
    # The Strata engine is a separate serve on a fixed port, so a test
    # that launches one would otherwise adopt (or collide with) whatever
    # happens to be running on this machine.  Point the Studio at a dead
    # port: the tests then exercise the Studio, not the box.
    test_client.put("/v1/studio/settings",
                    json={"strata_base": "http://127.0.0.1:1/v1"})
    return test_client


def test_the_launch_panel_has_an_engine_tab(client):
    page = client.get("/server").text
    assert 'id="launch-engine-tabs"' in page
    assert 'data-engine="llamacpp"' in page
    assert 'data-engine="strata"' in page
    assert 'id="launch-pane-llamacpp"' in page
    assert 'id="launch-pane-strata"' in page
    assert "function selectLaunchEngine(" in page


def test_the_llamacpp_pane_carries_the_launch_dials(client):
    page = client.get("/server").text
    for field in ("launch-model-select", "ngl", "l-autofit", "l-devices",
                  "hfrepo", "hffile", "l-threads", "l-fa", "l-parallel",
                  "l-mlock", "l-nommap", "l-ctk", "l-ctv", "l-batch",
                  "l-ubatch", "l-alias", "l-apikey"):
        assert f'id="{field}"' in page, f"the llama.cpp pane is missing {field}"


def test_the_strata_pane_carries_its_own_dials(client):
    page = client.get("/server").text
    for field in ("l-strata-config", "l-strata-repo", "l-strata-vram",
                  "l-strata-prefill", "l-strata-spec", "l-strata-args",
                  "l-strata-timeout"):
        assert f'id="{field}"' in page, f"the strata pane is missing {field}"
    # the shared dials sit above the tabs
    assert 'id="ctx"' in page
    assert 'id="l-port"' in page


def test_the_start_routes_to_the_engine_of_the_tab(client):
    page = client.get("/server").text
    assert "/v1/strata/launch" in page
    assert "function startStrataEngine(" in page
    assert "function strataLaunchBody(" in page


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
    # and it can be set
    updated = client.put("/v1/studio/settings",
                         json={"strata_repo": "/opt/strata"}).json()
    assert updated["strata_repo"] == "/opt/strata"
