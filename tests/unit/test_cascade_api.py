"""Unit tests for the ``/v1/cascade`` router (Track C console contract)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from harness.cascade import api as cascade_api
from harness.cascade.registry import default_registry
from harness.cascade.senses import MockSensoryEncoder, SensesService

REPO_ROOT = Path(__file__).resolve().parents[2]
CASCADE = REPO_ROOT / "experiments" / "cascade"


@pytest.fixture()
def client(tmp_path):
    """Router-only app over a tmp runs root, with a recording fake launcher."""
    tasks_dir = tmp_path / "experiments"
    tasks_dir.mkdir()
    (tasks_dir / "tasks-hard.json").write_text(
        json.dumps({"name": "t", "tasks": [{"id": "x", "bucket": "qa", "prompt": "?",
                                            "checker": {"type": "contains", "expect": ["a"]}}]}),
        encoding="utf-8",
    )
    runner = tasks_dir / "run_cascade.py"
    runner.write_text("# fake runner\n", encoding="utf-8")
    spawned: list[list[str]] = []

    def fake_spawn(cmd, cwd, log_path):
        spawned.append(list(cmd))
        Path(log_path).write_text("", encoding="utf-8")
        return 4242

    router = cascade_api.create_router(
        runs_root=tmp_path / "runs",
        tasks_root=tasks_dir,
        runner=runner,
        spawn=fake_spawn,
        registry=default_registry(),
        senses=SensesService(
            MockSensoryEncoder(),
            store_dir=tmp_path / "senses",
            max_bytes=64,
        ),
    )
    app = FastAPI()
    app.include_router(router)
    test_client = TestClient(app)
    test_client.spawned = spawned  # type: ignore[attr-defined]
    test_client.runs_root = tmp_path / "runs"  # type: ignore[attr-defined]
    return test_client


def test_roles_lists_the_taxonomy(client):
    data = client.get("/v1/cascade/roles").json()
    assert data["count"] == 22
    ids = {role["id"] for role in data["roles"]}
    assert {"A1", "C1", "D3", "E4", "H1"} <= ids
    c1 = next(role for role in data["roles"] if role["id"] == "C1")
    assert c1["kind"] == "route"
    assert "igpu" in c1["devices"]


def test_paths_and_route_table(client):
    data = client.get("/v1/cascade/paths").json()
    assert [p["id"] for p in data["paths"]] == ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]
    assert data["route_table"]["qa"] == "P2"
    assert data["route_table"]["reasoning"] == "P5"


def test_registry_covers_every_role(client):
    data = client.get("/v1/cascade/registry").json()
    assert data["count"] == 26
    # G1 is the deterministic harness policy (no model candidate by design).
    assert data["missing_roles"] == ["G1"]
    ids = {c["id"] for c in data["candidates"]}
    assert {"qwen3.5-2b", "ornith-1.5-9b", "scion-35b-a3b",
            "flash-next-strata", "clef-flash-9b", "clef-27b"} <= ids


def test_plan_prices_the_route(client):
    data = client.get("/v1/cascade/plan", params={"bucket": "qa", "confidence": 0.9, "out_len": 10}).json()
    plan = data["plan"]
    assert plan["path"] == "P2"
    assert plan["blocking_ms"] == pytest.approx(260.0)
    assert plan["escalation_path"] == "P3"
    assert plan["total_expected_ms"] == pytest.approx(301.0)
    assert data["latency_model"]["costs"] == []


def test_plan_rejects_out_of_range_confidence(client):
    assert client.get("/v1/cascade/plan", params={"confidence": 1.5}).status_code == 422
    assert client.get("/v1/cascade/plan", params={"out_len": -1}).status_code == 422


def test_plan_unknown_bucket_falls_back_to_standard_qa(client):
    plan = client.get("/v1/cascade/plan", params={"bucket": "nonsense"}).json()["plan"]
    assert plan["path"] == "P2"


def test_senses_status_and_image_upload_route_to_p6(client):
    status = client.get("/v1/cascade/senses/status").json()
    assert status["configured"] is True
    assert status["model"] == "mock-senses-v1"
    assert status["loaded"] is False

    payload = b"\x89PNG\r\n\x1a\nimage"
    resp = client.post(
        "/v1/cascade/senses",
        files={"file": ("figure.png", payload, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["modality"] == "image"
    assert body["bucket"] == "vision"
    assert body["path"] == "P6"
    assert body["plan"]["path"] == "P6"
    assert body["routed"] is True
    assert body["dimensions"] == 8
    assert Path(body["stored_path"]).read_bytes() == payload


def test_senses_text_upload_routes_to_standard_qa(client):
    resp = client.post(
        "/v1/cascade/senses",
        files={"file": ("notes.txt", b"hello hive", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["modality"] == "text"
    assert body["bucket"] == "qa"
    assert body["path"] == "P2"


def test_senses_audio_is_stored_but_not_routed(client):
    resp = client.post(
        "/v1/cascade/senses",
        files={"file": ("clip.wav", b"RIFFfake", "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["modality"] == "audio"
    assert body["path"] is None
    assert body["routed"] is False
    assert "no cascade path" in body["note"]
    assert Path(body["stored_path"]).is_file()


def test_senses_rejects_unsupported_empty_and_oversized_uploads(client):
    unsupported = client.post(
        "/v1/cascade/senses",
        files={"file": ("blob.bin", b"data", "application/x-unknown")},
    )
    assert unsupported.status_code == 415

    empty = client.post(
        "/v1/cascade/senses",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert empty.status_code == 422

    oversized = client.post(
        "/v1/cascade/senses",
        files={"file": ("big.txt", b"x" * 65, "text/plain")},
    )
    assert oversized.status_code == 413


def test_senses_is_503_when_no_encoder_is_attached(tmp_path):
    router = cascade_api.create_router(runs_root=tmp_path / "runs", tasks_root=tmp_path)
    app = FastAPI()
    app.include_router(router)
    bare = TestClient(app)

    status = bare.get("/v1/cascade/senses/status").json()
    assert status["configured"] is False
    resp = bare.post(
        "/v1/cascade/senses",
        files={"file": ("x.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 503


def test_frontier_empty_and_measured(client):
    empty = client.get("/v1/cascade/frontier").json()
    assert empty["measurements"] == []
    assert empty["frontiers"] == {}
    oracle = client.runs_root.parent / "experiments" / "oracle.json"  # type: ignore[attr-defined]
    oracle.write_text(
        json.dumps(
            [
                {"candidate": "a", "role": "C1", "device": "igpu", "quality": 0.9, "latency_ms": 10},
                {"candidate": "b", "role": "C1", "device": "igpu", "quality": 0.8, "latency_ms": 12},
            ]
        ),
        encoding="utf-8",
    )
    data = client.get("/v1/cascade/frontier", params={"file": "oracle.json"}).json()
    assert len(data["measurements"]) == 2
    assert [r["candidate"] for r in data["frontiers"]["C1"]] == ["a"]


def test_frontier_rejects_non_json_and_malformed(client):
    assert client.get("/v1/cascade/frontier", params={"file": "x.txt"}).status_code == 422
    bad = client.runs_root.parent / "experiments" / "bad.json"  # type: ignore[attr-defined]
    bad.write_text(json.dumps([{"candidate": "a"}]), encoding="utf-8")
    assert client.get("/v1/cascade/frontier", params={"file": "bad.json"}).status_code == 400


def test_run_launches_the_runner(client):
    resp = client.post("/v1/cascade/run", json={"run_name": "cascade-test", "limit": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] and body["pid"] == 4242 and body["run"] == "cascade-test"
    cmd = client.spawned[-1]  # type: ignore[attr-defined]
    assert cmd[0] == sys.executable
    assert "--tasks" in cmd and cmd[cmd.index("--tasks") + 1].endswith("tasks-hard.json")
    assert "--run-name" in cmd and "cascade-test" in cmd
    assert "--limit" in cmd
    assert (client.runs_root / "cascade-test").is_dir()  # type: ignore[attr-defined]


def test_run_defaults_name_and_refuses_duplicates(client):
    first = client.post("/v1/cascade/run", json={}).json()
    assert first["run"].startswith("cascade-")
    dup = client.post("/v1/cascade/run", json={"run_name": first["run"]})
    assert dup.status_code == 409


def test_run_input_validation(client):
    assert client.post("/v1/cascade/run", json={"tasks": "../secret.json"}).status_code == 422
    assert client.post("/v1/cascade/run", json={"tasks": "missing.json"}).status_code == 404
    assert client.post("/v1/cascade/run", json={"run_name": "..bad"}).status_code == 422
    assert client.post("/v1/cascade/run", json={"limit": -2}).status_code == 422


def test_runs_index_and_report(client):
    root = client.runs_root  # type: ignore[attr-defined]
    done = root / "cascade-done"
    done.mkdir(parents=True)
    (done / "report.json").write_text(
        json.dumps(
            {
                "spec": "cascade-pilot-hard",
                "summary": {
                    "tasks": 14, "scion_accuracy": 0.64, "api_accuracy": 1.0,
                    "cascade_accuracy": 0.93, "escalation_rate": 0.29,
                    "usage": {"total_cost_usd": 0.0071},
                },
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    (root / "cascade-running").mkdir()
    index = client.get("/v1/cascade/runs").json()["runs"]
    assert {r["run"] for r in index} == {"cascade-done", "cascade-running"}
    row = next(r for r in index if r["run"] == "cascade-done")
    assert row["status"] == "done"
    assert row["cascade_accuracy"] == 0.93
    assert row["cost_usd"] == 0.0071
    running = next(r for r in index if r["run"] == "cascade-running")
    assert running["status"] == "running"
    report = client.get("/v1/cascade/report/cascade-done").json()
    assert report["summary"]["tasks"] == 14


def test_report_validation(client):
    assert client.get("/v1/cascade/report/nope").status_code == 404
    assert client.get("/v1/cascade/report/..bad").status_code == 422


# ---------------------------------------------------------------------------
# validator service
# ---------------------------------------------------------------------------


class _FakeValidator:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, str, float | None]] = []

    def status(self) -> dict:
        return {"model": "fake-validator", "device": "cpu", "threshold": 0.5,
                "loaded": True, "load_ms": 1.0, "checks": len(self.calls)}

    def verdict(self, task, candidate, *, threshold=None):
        self.calls.append((task, candidate, threshold))
        return {"model": "fake-validator", "device": "cpu", "p_correct": 0.9,
                "accept": True, "threshold": threshold or 0.5, "ms": 2.0,
                "checks": len(self.calls)}


def _validator_client(tmp_path, validator=None):
    router = cascade_api.create_router(
        runs_root=tmp_path / "runs", tasks_root=tmp_path, validator=validator)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_validator_status_and_validate_route(tmp_path):
    validator = _FakeValidator()
    client = _validator_client(tmp_path, validator)

    status = client.get("/v1/cascade/validator").json()
    assert status["configured"] is True and status["model"] == "fake-validator"

    out = client.post("/v1/cascade/validate", json={
        "task": {"prompt": "2+2?"}, "candidate": "4", "threshold": 0.7}).json()
    assert out["p_correct"] == 0.9 and out["accept"] is True
    assert validator.calls == [({"prompt": "2+2?"}, "4", 0.7)]


def test_validate_rejects_malformed_input(tmp_path):
    client = _validator_client(tmp_path, _FakeValidator())
    assert client.post("/v1/cascade/validate", json={}).status_code == 422
    assert client.post(
        "/v1/cascade/validate",
        json={"task": {}, "candidate": "x"}).status_code == 422
    assert client.post(
        "/v1/cascade/validate",
        json={"task": {"prompt": "p"}, "candidate": "  "}).status_code == 422


def test_validate_without_a_validator_is_503(tmp_path):
    client = _validator_client(tmp_path)
    status = client.get("/v1/cascade/validator").json()
    assert status["configured"] is False
    assert client.post(
        "/v1/cascade/validate",
        json={"task": {"prompt": "p"}, "candidate": "a"}).status_code == 503
