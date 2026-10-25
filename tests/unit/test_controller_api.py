"""The Studio API exposes the profile/controller lane without auto-applying."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.app import create_app
from harness.controller.profile import HardwareProfiler
from harness.controller.service import ControllerService


def _summary():
    return {
        "cpu_count": 16,
        "total_ram_gb": 30.9,
        "available_ram_gb": 18.2,
        "vram_source": "sysfs",
        "devices": [
            {"index": 0, "name": "AMD GPU 0", "backend": "rocm",
             "memory_gb": 21.46, "used_gb": 19.32, "free_gb": 2.14,
             "display": True, "visible": True},
            {"index": 1, "name": "AMD GPU 1", "backend": "rocm",
             "memory_gb": 21.46, "used_gb": 0.55, "free_gb": 20.91,
             "display": False, "visible": True},
        ],
        "disk": {"models": {"free_gb": 23.7, "total_gb": 100.0}},
    }


@pytest.fixture()
def client(tmp_path, monkeypatch):
    class FakeProfiler(HardwareProfiler):
        def __init__(self, state):
            super().__init__(state, probe=_summary)

    monkeypatch.setattr("harness.controller.profile.HardwareProfiler", FakeProfiler)
    app = create_app(
        providers_file=tmp_path / "providers.local.json",
        engines_file=tmp_path / "engines.local.json",
        runs_root=tmp_path / "runs",
        log_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
    )
    return TestClient(app)


def test_scan_requires_consent(client):
    r = client.post("/v1/profiles/scan", json={})
    assert r.status_code == 403
    r = client.post("/v1/profiles/scan", json={"consent": "hardware"})
    assert r.status_code == 200
    assert r.json()["profile"]["gpus"][1]["display"] is False


def test_profile_is_readable_after_scan(client):
    client.post("/v1/profiles/scan",
                json={"consent": "hardware", "profile_id": "local-test"})
    r = client.get("/v1/profiles/hardware")
    assert r.status_code == 200
    assert r.json()["profile"]["profile_id"] == "local-test"


def test_propose_apply_requires_confirm_and_rolls_back(client):
    client.post("/v1/profiles/scan",
                json={"consent": "hardware", "profile_id": "local-test"})
    r = client.post("/v1/controller/propose", json={
        "model_gib": 12, "native_ctx": 32768, "desired_ctx": 16384,
        "request": "How do I test this?"})
    assert r.status_code == 200
    proposal = r.json()["proposal"]
    assert proposal["request_features"]["intent"] == "qa"
    proposal_id = proposal["id"]

    labelled = client.post("/v1/controller/outcome", json={
        "proposal_id": proposal_id, "outcome": "accepted",
        "metrics": {"latency_ms": 12}})
    assert labelled.status_code == 200
    assert labelled.json()["outcome"] == "accepted"

    denied = client.post("/v1/controller/apply", json={"proposal_id": proposal_id})
    assert denied.status_code == 403
    applied = client.post("/v1/controller/apply", json={
        "proposal_id": proposal_id, "confirm": True})
    assert applied.status_code == 200
    assert applied.json()["settings"]["runtime_defaults"]["ngl"] == 999

    status = client.get("/v1/controller/status").json()
    assert status["latest_proposal"]["id"] == proposal_id
    assert status["controller"]["last_apply"]["proposal_id"] == proposal_id

    reverted = client.post("/v1/controller/revert", json={"confirm": True})
    assert reverted.status_code == 200
    assert "ngl" not in reverted.json()["settings"]["runtime_defaults"]


def test_encoder_status_does_not_load(client):
    r = client.get("/v1/encoder/status")
    assert r.status_code == 200
    body = r.json()
    assert body["loaded"] is False
    assert body["model"]


def test_embed_flag_traces_a_vector_without_raw_text(client, monkeypatch):
    monkeypatch.setattr(
        ControllerService, "_request_embedding",
        lambda self, request: {"model": "fake", "dimensions": 2,
                               "vector": [0.25, -0.5]})
    client.post("/v1/profiles/scan",
                json={"consent": "hardware", "profile_id": "local-test"})
    r = client.post("/v1/controller/propose", json={
        "model_gib": 12, "request": "private request text",
        "embed_request": True})
    assert r.status_code == 200
    state = Path(client.app.state.harness.state_dir)
    trace = next((state / "profiles").glob("controller-traces-*.jsonl")).read_text(
        encoding="utf-8")
    row = json.loads(trace.splitlines()[0])
    assert row["request_embedding"]["vector"] == [0.25, -0.5]
    assert "private request text" not in trace
